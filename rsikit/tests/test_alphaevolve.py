"""Deterministic checks of generation, selection, evaluation, and failure budgets."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from pydantic import ValidationError
from slick import prompts
from slick.providers import ProviderError

import rsikit.alphaevolve as alphaevolve
from rsikit.alphaevolve import AlphaEvolve, Config, Evaluation, EvaluationStage, InvalidCandidate
from rsikit.alphaevolve.agent import Guidance
from rsikit.alphaevolve.edits import Edit, Mutation, Program, apply_edits, check_rewrite
from rsikit.tests.providers import ScriptedProvider

ROOT = Path(alphaevolve.__file__).parent / "prompts"
SOURCE = """from rsikit import Policy
# EVOLVE-BLOCK-START
class Solution(Policy):
    async def act(self, observation):
        return 0
# EVOLVE-BLOCK-END
"""


def program(number):
    return SOURCE.replace("return 0", f"return {number}")


def rewrite(number):
    return Program(source=program(number))


class AlphaEvolveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_typed_edits_feedback_and_rejection_budget(self):
        provider = ScriptedProvider(
            [
                Mutation(edits=[Edit(search="return 0", replacement="return 1")]),
                "not json",
                Mutation(edits=[Edit(search="from rsikit", replacement="from other")]),
                '{"edits": []}',
            ]
        )

        async def evaluate(source):
            return Evaluation({"reward": float("return 1" in source)}, feedback="measured result")

        agent = AlphaEvolve("Improve the policy", provider, evaluate, config=Config(islands=1))
        best = await agent.run(SOURCE, attempts=4, concurrency=1)
        self.assertEqual(best.content, program(1))
        self.assertEqual((agent.generation_calls, agent.evaluations, agent.completed), (4, 2, 4))
        self.assertEqual(agent.attempts[1]["raw"], "not json")
        self.assertEqual([r["status"] for r in agent.attempts], ["evaluated"] + ["rejected"] * 3)
        self.assertIn("measured result", provider.calls[1])
        self.assertIn("not json", provider.calls[-1])
        self.assertIn('"properties"', provider.calls[0])

        edits = [
            Edit(search="return 0", replacement="return 1"),
            Edit(search="return 1", replacement="return 2"),
        ]
        self.assertEqual(apply_edits(SOURCE, edits), program(2))
        for source, edit in (
            (SOURCE, Edit(search="absent", replacement="x")),
            (SOURCE, Edit(search="return 0", replacement="return 0")),
            ("aaa", Edit(search="aa", replacement="b")),
            (SOURCE, Edit(search="from rsikit", replacement="from elsewhere")),
        ):
            with self.assertRaises(InvalidCandidate):
                apply_edits(source, [edit])
        with self.assertRaises(InvalidCandidate):
            check_rewrite(SOURCE, program(1).replace("EVOLVE-BLOCK-END", "changed"))
        with self.assertRaises(ValidationError):
            Mutation.model_validate({"edits": [{"search": "", "replacement": "x"}]})

    async def test_multiobjective_diversity_and_reset(self):
        values = {
            SOURCE: (0, 0, (0,)),
            program(1): (5, 1, (0,)),
            program(2): (1, 5, (0,)),
            program(3): (0, 0, (1,)),
        }

        async def evaluate(source):
            speed, size, cell = values[source]
            return Evaluation({"speed": speed, "size": size}, cell=cell)

        agent = AlphaEvolve(
            "task",
            ScriptedProvider([rewrite(1), rewrite(2), rewrite(3)]),
            evaluate,
            config=Config(islands=1, mode="rewrite"),
        )
        best = await agent.run(SOURCE, attempts=3, concurrency=1, target_metric="speed")
        self.assertEqual(best.content, program(1))
        self.assertEqual(agent.best_by_metric["size"].content, program(2))
        self.assertEqual(
            {p.content for p in agent.islands[0].values()}, {program(1), program(2), program(3)}
        )
        agent.islands.append(dict(agent.islands[0]))
        agent.reset_islands()
        self.assertEqual(len(agent.events), 1)
        self.assertEqual(agent.best_by_metric["speed"], best)

    async def test_cascade_and_program_checks_avoid_expensive_evaluations(self):
        expensive = []

        async def cheap(source):
            return Evaluation({"valid": float(source != program(1))}, feedback="cheap diagnostic")

        async def final(source):
            expensive.append(source)
            return Evaluation({"reward": float("nan") if source == program(2) else 1})

        provider = ScriptedProvider(
            [
                rewrite(1),
                rewrite(2),
                rewrite(3),
                Program(source=SOURCE.replace("class Solution(Policy):", "class Solution(:")),
                Program(source=SOURCE.replace("class Solution(Policy):", "class Other(Policy):")),
            ]
        )
        agent = AlphaEvolve(
            "task",
            provider,
            final,
            stages=(EvaluationStage(cheap, {"valid": 1}),),
            config=Config(mode="rewrite"),
        )
        result = await agent.run(SOURCE, attempts=5, concurrency=1)
        self.assertEqual(result.content, SOURCE)  # Ties preserve the incumbent.
        self.assertEqual(expensive, [SOURCE, program(2), program(3)])
        self.assertEqual(agent.evaluations, 7)
        self.assertEqual(len(agent.programs), 2)
        self.assertIn("cheap diagnostic", provider.calls[1])
        self.assertTrue(all(agent.attempts[i]["status"] == "rejected" for i in (0, 1, 3, 4)))

    async def test_ensemble_meta_guidance_and_failures_have_no_hidden_retries(self):
        unused = ScriptedProvider([])
        selected = ScriptedProvider(
            [
                ProviderError("offline"),
                '{"instruction": " "}',
                rewrite(1),
                rewrite(2),
                Guidance(instruction="Try another representation"),
                rewrite(3),
            ]
        )

        async def evaluate(source):
            return Evaluation({"reward": float(source == program(1)) + 2 * (source == program(2))})

        agent = AlphaEvolve(
            "task",
            unused,
            evaluate,
            ensemble=((unused, 0), (selected, 1)),
            config=Config(mode="rewrite", meta_interval=2, reset_interval=1),
        )
        await agent.run(SOURCE, attempts=4, concurrency=1)
        self.assertEqual(unused.calls, [])
        self.assertEqual((agent.generation_calls, agent.meta_calls), (4, 2))
        self.assertEqual(agent.attempts[1]["meta_raw"], '{"instruction": " "}')
        self.assertIn("blank", agent.attempts[1]["meta_error"])
        self.assertEqual(agent.prompt_ideas[-1].uses, 1)
        self.assertIn("Try another representation", selected.calls[-1])
        self.assertEqual(len(agent.events), 8)

    async def test_async_completion_and_unexpected_errors_cancel_workers(self):
        gate, started = asyncio.Event(), asyncio.Event()

        async def evaluate(source):
            if source == program(1):
                started.set()
                await gate.wait()
            if source == program(2):
                await started.wait()
            if source == program(3):
                gate.set()
            return Evaluation({"reward": 1})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider([rewrite(1), rewrite(2), rewrite(3)]),
            evaluate,
            config=Config(mode="rewrite"),
        )
        await asyncio.wait_for(agent.run(SOURCE, attempts=3, concurrency=2), 2)
        self.assertEqual(agent.evaluations, 4)
        self.assertEqual([p.id for p in agent.programs], [0, 2, 3, 1])

        started, cancelled = asyncio.Event(), asyncio.Event()

        async def broken(source):
            if source == program(1):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if source == program(2):
                await started.wait()
                raise RuntimeError("evaluator bug")
            return Evaluation({"reward": 0})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider([rewrite(1), rewrite(2)]),
            broken,
            config=Config(mode="rewrite"),
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator bug"):
            await agent.run(SOURCE, attempts=2, concurrency=2)
        self.assertTrue(cancelled.is_set())

    async def test_timeouts_seed_failure_and_prompt_rendering(self):
        async def evaluate(source):
            if source == program(1):
                await asyncio.Event().wait()
            return Evaluation({"changed" if source == program(2) else "reward": 1})

        agent = AlphaEvolve(
            "task",
            ScriptedProvider([rewrite(1), rewrite(2)]),
            evaluate,
            config=Config(mode="rewrite", evaluation_timeout=0.02),
        )
        best = await agent.run(SOURCE, attempts=2, concurrency=1)
        self.assertEqual(best.content, SOURCE)
        self.assertIn("TimeoutError", agent.attempts[0]["error"])
        self.assertIn("objective names", agent.attempts[1]["error"])
        for method in (AlphaEvolve.mutate, AlphaEvolve.rewrite):
            text = await method.render(agent, best, [], "guidance", [])
            self.assertIn("guidance", text)
            self.assertIn('"properties"', text)
            self.assertIn("Solution", text)
        self.assertIn('"properties"', await AlphaEvolve.evolve_prompt.render(agent, best, [], []))
        for path in ROOT.glob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )
        provider = ScriptedProvider([])
        with self.assertRaises(InvalidCandidate):
            await AlphaEvolve("task", provider, evaluate).run("not a program", attempts=0)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
