"""Offline checks for evaluator pruning and bounded asynchronous search."""

import asyncio
import unittest
from types import SimpleNamespace

from research.alphaevolve.paper.evaluation import (
    EvaluationResult,
    EvaluationStage,
    evaluate_cascade,
)
from research.alphaevolve.paper.pipeline import search
from rsikit.evaluation import InfrastructureError, PolicyError


class Generator:
    def __init__(self):
        self.attempts = []
        self.results = {}
        self.discarded = []
        self.repairs = []
        self.active = 0
        self.peak = 0

    async def generate(self, n=1, *, concurrency=1):
        policies = []
        for _ in range(n):
            policy = SimpleNamespace(id=str(len(self.attempts)))
            self.attempts.append(policy)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                policies.append(policy)
            finally:
                self.active -= 1
        return policies

    async def repair(self, policy, diagnostic):
        self.repairs.append((policy.id, diagnostic))
        return SimpleNamespace(id=policy.id + "r") if policy.id == "0" else None

    def update_results(self, results):
        self.results.update(results)

    def discard(self, policy, reason):
        self.discarded.append((policy.id, reason))


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
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
            return EvaluationResult({"reward": 2}, {"size": 3}, "cheap result", {7: 2})

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
            return EvaluationResult({"reward": 2}, feedback="estimate")

        async def full(policy):
            return EvaluationResult({"reward": 4}, feedback="measured")

        async def grade(policy, measured):
            self.assertEqual(measured.metrics["reward"], 4)
            return EvaluationResult({"quality": 0.2}, feedback="grader", accepted=False)

        result = await evaluate_cascade(
            object(), [EvaluationStage(cheap), EvaluationStage(full)], feedback_evaluator=grade
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics, {"reward": 4, "quality": 0.2})
        self.assertEqual(result.feedback, "estimate\nmeasured\ngrader")

    async def test_validation_and_infrastructure_errors(self):
        for kwargs in [
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
    async def test_result_failures_are_repaired_but_screened_candidates_are_discarded(self):
        generator = Generator()
        batches = []

        async def evaluate(policies):
            batches.append([p.id for p in policies])
            return {
                p.id: EvaluationResult(failure="bad action")
                if p.id == "0"
                else EvaluationResult({"reward": 2}, accepted=p.id != "1")
                for p in policies
            }

        await search(generator, evaluate, proposals=3, evaluation_batch_size=3)
        self.assertEqual(generator.repairs, [("0", "bad action")])
        self.assertEqual(set(generator.results), {"0r", "2"})
        self.assertEqual([id for id, _ in generator.discarded], ["1"])
        self.assertEqual(batches, [["0", "1", "2"], ["0r"]])

    async def test_generation_and_runtime_repairs_share_concurrency_limit(self):
        generator = Generator()
        generate = generator.generate
        active = peak = repairs = 0

        async def produce(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return await generate(*args, **kwargs)
            finally:
                active -= 1

        async def repair(policy, diagnostic):
            nonlocal active, peak, repairs
            active += 1
            repairs += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return SimpleNamespace(id=policy.id + "r")
            finally:
                active -= 1

        async def evaluate(policies):
            failures = {
                p.id: "bad action"
                for p in policies
                if p.id not in {"0", "1", "2"} and not p.id.endswith("r")
            }
            if failures:
                error = PolicyError("bad action")
                error.failures = failures
                raise error
            return {p.id: EvaluationResult({"reward": 1}) for p in policies}

        generator.generate, generator.repair = produce, repair
        await search(
            generator, evaluate, proposals=15, evaluation_batch_size=3, generation_concurrency=3
        )
        self.assertEqual(repairs, 12)
        self.assertEqual((peak, active), (3, 0))
        self.assertEqual(len(generator.results), 15)

    async def test_repair_failure_and_cancellation_drain_siblings_before_failed_event(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                generator = Generator()
                started, release = asyncio.Event(), asyncio.Event()
                active = 0
                events = []

                async def repair(policy, diagnostic):
                    nonlocal active
                    active += 1
                    if active == 2:
                        started.set()
                    try:
                        await release.wait()
                        if policy.id == "0":
                            raise RuntimeError("repair provider offline")
                        await asyncio.Future()
                    finally:
                        active -= 1

                async def evaluate(policies):
                    error = PolicyError("bad action")
                    error.failures = {p.id: "bad action" for p in policies}
                    raise error

                def event(name, policies):
                    events.append(name)
                    if name == "failed":
                        self.assertEqual(active, 0)

                generator.repair = repair
                task = asyncio.create_task(
                    search(
                        generator,
                        evaluate,
                        proposals=3,
                        evaluation_batch_size=3,
                        generation_concurrency=2,
                        on_event=event,
                    )
                )
                try:
                    await asyncio.wait_for(started.wait(), 1)
                    if cancelled:
                        task.cancel()
                    else:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError if cancelled else RuntimeError):
                        await asyncio.wait_for(task, 1)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(active, 0)
                self.assertEqual(events[-1], "failed")

    async def test_bootstrap_overlap_and_bounded_backpressure(self):
        generator = Generator()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        generated = set()

        def event(name, policies):
            if name == "generated":
                generated.update(p.id for p in policies)
            if name == "evaluated":
                self.assertTrue(all(p.id in generator.results for p in policies))

        async def evaluate(policies):
            nonlocal calls
            calls += 1
            self.assertTrue(all(p.id in generated for p in policies))
            self.assertLessEqual(len(policies), 2)
            if calls == 1:
                self.assertEqual(len(generator.attempts), 2)
            else:
                entered.set()
                await release.wait()
            return {p.id: EvaluationResult({"reward": 1}) for p in policies}

        task = asyncio.create_task(
            search(
                generator,
                evaluate,
                proposals=30,
                generation_concurrency=3,
                evaluation_batch_size=2,
                on_event=event,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            for _ in range(30):
                await asyncio.sleep(0)
            self.assertGreater(len(generator.attempts), 4)
            self.assertLessEqual(len(generator.attempts), 2 + 2 + 2 + 3)
            release.set()
            await asyncio.wait_for(task, 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(generator.results), 30)
        self.assertLessEqual(generator.peak, 3)

    async def test_empty_generations_finish_and_preserve_attempt_budget(self):
        generator = Generator()
        original = generator.generate

        async def empty(*args, **kwargs):
            await original(*args, **kwargs)
            return []

        generator.generate = empty

        async def evaluate(policies):
            self.fail("empty proposals must not be evaluated")

        events = []
        await asyncio.wait_for(
            search(
                generator,
                evaluate,
                proposals=9,
                evaluation_batch_size=2,
                on_event=lambda name, policies: events.append(name),
            ),
            2,
        )
        self.assertEqual(len(generator.attempts), 9)
        self.assertIn("discarded", events)

    async def test_policy_repairs_rejections_and_survivors(self):
        generator = Generator()
        batches = []

        async def evaluate(policies):
            ids = [p.id for p in policies]
            batches.append(ids)
            failed = {id: "bad action" for id in ids if id in {"0", "1"}}
            if failed:
                error = PolicyError("bad action")
                error.failures = failed
                raise error
            return {p.id: EvaluationResult({"reward": 1}, accepted=p.id != "2") for p in policies}

        events = []
        await search(
            generator,
            evaluate,
            proposals=4,
            evaluation_batch_size=4,
            on_event=lambda name, policies: events.append((name, [p.id for p in policies])),
        )
        self.assertEqual(batches, [["0", "1", "2", "3"], ["0r", "2", "3"]])
        self.assertEqual(set(generator.results), {"0r", "3"})
        self.assertEqual(generator.repairs, [("0", "bad action"), ("1", "bad action")])
        self.assertIn("2", [id for id, _ in generator.discarded])
        self.assertIn(("repair", ["0r"]), events)
        self.assertEqual(len(generator.attempts), 4)

    async def test_failure_and_cancellation_clean_up_workers_before_saving(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                generator = Generator()
                entered = asyncio.Event()
                events = []
                calls = 0

                def event(name, policies):
                    events.append(name)
                    if name == "failed":
                        self.assertEqual(generator.active, 0)

                async def evaluate(policies):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return {p.id: EvaluationResult({"reward": 1}) for p in policies}
                    entered.set()
                    await asyncio.sleep(0)
                    if cancelled:
                        await asyncio.Event().wait()
                    raise InfrastructureError("backend stopped")

                task = asyncio.create_task(
                    search(
                        generator,
                        evaluate,
                        proposals=100,
                        evaluation_batch_size=2,
                        generation_concurrency=3,
                        on_event=event,
                    )
                )
                await asyncio.wait_for(entered.wait(), 2)
                if cancelled:
                    task.cancel()
                with self.assertRaises(
                    asyncio.CancelledError if cancelled else InfrastructureError
                ):
                    await asyncio.wait_for(task, 2)
                self.assertEqual(events[-1], "failed")
                self.assertEqual(generator.active, 0)

    async def test_generation_fault_does_not_deadlock_waiting_consumer(self):
        generator = Generator()
        original = generator.generate

        async def generate(*args, **kwargs):
            if len(generator.attempts) >= 2:
                raise RuntimeError("provider offline")
            return await original(*args, **kwargs)

        generator.generate = generate

        async def evaluate(policies):
            return {p.id: EvaluationResult({"reward": 1}) for p in policies}

        with self.assertRaisesRegex(RuntimeError, "provider offline"):
            await asyncio.wait_for(
                search(generator, evaluate, proposals=10, evaluation_batch_size=2), 2
            )

    async def test_duplicate_pending_programs_share_results_without_stale_queue_updates(self):
        generator = Generator()
        original = generator.generate
        pending = {}
        completed = 0

        async def generate(*args, **kwargs):
            policies = await original(*args, **kwargs)
            for policy in policies:
                policy.id = "identical"
                pending[policy.id] = pending.get(policy.id, 0) + 1
            return policies

        def update(results):
            nonlocal completed
            for id in results:
                completed += pending.pop(id)

        generator.generate = generate
        generator.update_results = update

        async def evaluate(policies):
            for _ in range(5):
                await asyncio.sleep(0)
            return {p.id: EvaluationResult({"reward": 1}) for p in policies}

        await search(
            generator, evaluate, proposals=12, evaluation_batch_size=2, generation_concurrency=3
        )
        self.assertEqual(completed, 12)
        self.assertEqual(pending, {})


if __name__ == "__main__":
    unittest.main()
