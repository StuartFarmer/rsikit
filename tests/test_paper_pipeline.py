"""Offline checks for evaluator pruning and bounded asynchronous search."""

import asyncio
import unittest

import numpy as np

from research.alphaevolve.paper import EvaluationResult
from research.alphaevolve.paper.evaluation import (
    EvaluationStage,
    evaluate_cascade,
)
from research.alphaevolve.paper.pipeline import search
from rsikit.evaluation import InfrastructureError
from tests.helpers import episodes


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutated_callback_results_and_thresholds_are_revalidated(self):
        result = EvaluationResult(metrics={"reward": 2})
        result.accepted = 1

        async def measured(policy):
            return result

        with self.assertRaises(ValueError):
            await evaluate_cascade(object(), [EvaluationStage(measured)])
        result.accepted = True
        stage = EvaluationStage(measured, {"reward": 1})
        stage.thresholds["reward"] = True
        with self.assertRaises(ValueError):
            await evaluate_cascade(object(), [stage])

    async def test_stage_failure_skips_thresholds_and_remaining_stages(self):
        async def broken(policy):
            return EvaluationResult(failure="invalid action")

        async def expensive(policy):
            self.fail("failed candidate reached expensive stage")

        result = await evaluate_cascade(
            object(), [EvaluationStage(broken, {"reward": 1}), EvaluationStage(expensive)]
        )
        self.assertEqual(result.failure, "invalid action")
        self.assertFalse(result.accepted)

    async def test_pruning_combines_feedback_and_skips_expensive_stage(self):
        calls = []

        async def cheap(policy):
            calls.append("cheap")
            return EvaluationResult(
                metrics={"reward": 2},
                features={"size": 3},
                feedback="cheap result",
                seed_scores={7: 2},
            )

        async def expensive(policy):
            self.fail("pruned candidate reached expensive stage")

        result = await evaluate_cascade(
            object(), [EvaluationStage(cheap, {"reward": 3}), EvaluationStage(expensive)]
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics, {"reward": 2})
        self.assertEqual(result.features, {"size": 3})
        self.assertEqual(result.seed_scores, {7: 2})
        self.assertIn("cheap result", result.feedback)
        self.assertEqual(calls, ["cheap"])

    async def test_later_measurement_overwrites_estimate_and_grader_can_reject(self):
        async def cheap(policy):
            return EvaluationResult(feedback="estimate", metrics={"reward": 2})

        async def full(policy):
            return EvaluationResult(feedback="measured", metrics={"reward": 4})

        async def grade(policy, measured):
            self.assertEqual(measured.metrics["reward"], 4)
            return EvaluationResult(feedback="grader", accepted=False, metrics={"quality": 0.2})

        result = await evaluate_cascade(
            object(), [EvaluationStage(cheap), EvaluationStage(full)], feedback_evaluator=grade
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics, {"reward": 4, "quality": 0.2})
        self.assertEqual(result.feedback, "estimate\nmeasured\ngrader")

    async def test_validation_and_infrastructure_errors(self):
        for kwargs in [
            {"metrics": {"x": np.bool_(True)}},
            {"metrics": {"x": float("nan")}},
            {"metrics": {"": 2}},
            {"metrics": {4: 2}},
            {"metrics": {}, "features": {"x": float("inf")}},
            {"metrics": {}, "seed_scores": {"one": 2}},
            {"metrics": {}, "feedback": 3},
            {"metrics": {}, "accepted": "yes"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationResult(**kwargs)
        with self.assertRaises(ValueError):
            await evaluate_cascade(object(), [])
        with self.assertRaises(ValueError):
            EvaluationStage(None)
        with self.assertRaises(ValueError):
            EvaluationStage(lambda policy: None, {"x": float("nan")})

        async def unavailable(policy):
            raise InfrastructureError("offline")

        with self.assertRaisesRegex(InfrastructureError, "offline"):
            await evaluate_cascade(object(), [EvaluationStage(unavailable)])


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pathlib import Path
        from unittest.mock import patch

        from slick import prompts

        from research import alphaevolve

        root = patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, outputs, **options):
        from research.alphaevolve.paper import AlphaEvolve, Config
        from tests.providers import ScriptedProvider

        agent = AlphaEvolve(
            "task",
            ScriptedProvider(outputs),
            config=Config(islands=1, meta_interval=0, mode="rewrite", **options),
        )
        self.addCleanup(agent.close)
        return agent

    async def test_complete_rounds_measure_founders_before_descendants(self):
        from tests.test_alphaevolve import program

        agent = self.agent([program(i) for i in range(4)])
        counts = []

        async def evaluate(policies):
            counts.append((agent.generation_calls, agent.completed))
            return {p.id: episodes({0: i + 1}) for i, p in enumerate(policies)}

        await search(agent, evaluate, proposals=4, evaluation_batch_size=2)
        self.assertEqual(counts, [(2, 0), (4, 2)])
        self.assertTrue(all(row["parent"] is not None for row in agent.attempts[2:]))
        self.assertTrue(agent.done)

    async def test_failures_repair_rejections_discard_and_successes_stay(self):
        from tests.test_alphaevolve import program

        agent = self.agent([program(i) for i in range(4)], max_repairs=1)
        panels = []

        async def evaluate(policies):
            panels.append([p.name for p in policies])
            if len(panels) == 1:
                return {
                    policies[0].id: episodes(failure="bad action"),
                    policies[1].id: episodes({0: 8}, accepted=False),
                    policies[2].id: episodes({0: 3}),
                }
            return {p.id: episodes({0: 9}) for p in policies}

        await search(agent, evaluate, proposals=3, evaluation_batch_size=3)
        self.assertEqual([len(p) for p in panels], [3, 1])
        self.assertEqual((agent.completed, agent.repair_calls, len(agent.attempts)), (2, 1, 3))
        self.assertEqual(agent.attempts[1]["status"], "discarded")

    async def test_all_invalid_generations_exhaust_original_attempt_budget(self):
        agent = self.agent(["bad"] * 3, max_repairs=0)

        async def evaluate(policies):
            self.fail("invalid candidates reached evaluation")

        await search(agent, evaluate, proposals=3, evaluation_batch_size=2)
        self.assertEqual(len(agent.attempts), 3)
        self.assertIsNone(agent.best)
        self.assertTrue(agent.done)

    async def test_cancellation_drains_generation_and_checkpoints_evidence(self):
        from unittest.mock import patch

        from tests.test_alphaevolve import program

        agent = self.agent([program(0)])
        entered = asyncio.Event()
        active = 0

        async def waiting(*args, **kwargs):
            nonlocal active
            active += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1

        with patch.object(agent.models[0][0], "acall", side_effect=waiting):
            task = asyncio.create_task(
                search(agent, None, proposals=3, generation_concurrency=2, evaluation_batch_size=3)
            )
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(active, 0)
        self.assertFalse(agent._proposing)
        self.assertIsNotNone(agent.database.load_state("optimizer"))

    async def test_generation_and_repairs_obey_concurrency(self):
        from unittest.mock import patch

        from tests.test_alphaevolve import program

        agent = self.agent([program(i) for i in range(8)], max_repairs=1)
        provider = agent.models[0][0]
        original = provider.acall
        active = peak = 0

        async def tracked(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return await original(*args, **kwargs)
            finally:
                active -= 1

        calls = 0

        async def evaluate(policies):
            nonlocal calls
            calls += 1
            return {
                p.id: episodes(failure="broken") if calls == 1 else episodes({0: 1})
                for p in policies
            }

        with patch.object(provider, "acall", side_effect=tracked):
            await search(
                agent, evaluate, proposals=4, evaluation_batch_size=4, generation_concurrency=2
            )
        self.assertEqual((peak, active, agent.repair_calls), (2, 0, 4))
