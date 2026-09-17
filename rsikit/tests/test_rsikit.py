"""Exercise search decisions, real evaluation, and persistent failure evidence."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

import rsikit
from rsikit.tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1]
SEED = "def solve(x):\n    return x + 1\n"


class StrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposer_requires_an_implementation(self):
        with self.assertRaises(TypeError):
            rsikit.Proposer()

        class Replacement(rsikit.Proposer):
            async def __call__(self, parent, history):
                return parent.source.replace("x + 1", "x + 2")

        strategy = rsikit.HillClimb(
            SEED, rsikit.Evaluation(valid=True, metrics={"score": 1}), Replacement()
        )
        candidates = await strategy.generate()
        self.assertEqual(candidates[0].source, SEED.replace("x + 1", "x + 2"))
        self.assertIsInstance(rsikit.SlickProposer("Revise", ScriptedProvider([])), rsikit.Proposer)

    async def test_generate_update_owns_state_and_keeps_valid_improvements(self):
        provider = ScriptedProvider(["wrong", "better", "worse", "tie"])
        with patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"):
            strategy = rsikit.HillClimb(
                SEED,
                rsikit.Evaluation(valid=True, metrics={"loss": 3}),
                rsikit.SlickProposer("Reduce error", provider),
                objective="loss",
                maximize=False,
            )
            for text, score, valid in [
                ("wrong", -100, False),
                ("better", 1, True),
                ("worse", 2, True),
                ("tie", 1, True),
            ]:
                candidates = await strategy.generate()
                self.assertEqual(candidates[0].source, text)
                evaluation = rsikit.Evaluation(
                    valid=valid, metrics={"loss": score}, feedback=f"measured {text}"
                )
                await strategy.update(candidates, [evaluation])
        self.assertEqual(strategy.best.source, "better")
        self.assertEqual([c.parent_id for c in strategy.history], [None, 0, 0, 2, 2])
        self.assertEqual(len(strategy.history), 5)
        self.assertIsNone(strategy.pending)
        self.assertIn("measured wrong", provider.calls[1])

    async def test_invalid_edits_are_remembered_without_requesting_evaluation(self):
        source = "fixed\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
        proposals = iter([source.replace("fixed", "changed"), "", source, source.replace("1", "2")])

        async def propose(parent, history):
            return next(proposals)

        strategy = rsikit.HillClimb(
            source, rsikit.Evaluation(valid=True, metrics={"score": 0}), propose
        )
        for _ in range(3):
            self.assertEqual(await strategy.generate(), [])
            await strategy.update([], [])
        self.assertTrue(all(not candidate.evaluation.valid for candidate in strategy.history[1:]))
        candidates = await strategy.generate()
        await strategy.update(candidates, [rsikit.Evaluation(valid=True, metrics={"score": 1})])
        self.assertEqual(strategy.best.source, source.replace("1", "2"))

    async def test_mismatched_updates_leave_pending_state_intact(self):
        async def propose(parent, history):
            return "new"

        strategy = rsikit.HillClimb(
            SEED, rsikit.Evaluation(valid=True, metrics={"score": 1}), propose
        )
        candidates = await strategy.generate()
        with self.assertRaises(RuntimeError):
            await strategy.generate()
        for submitted, results in [
            ([], []),
            (candidates, []),
            (
                [candidates[0].model_copy(update={"id": 99})],
                [rsikit.Evaluation(valid=True, metrics={"score": 2})],
            ),
            (candidates, [rsikit.Evaluation(valid=True, metrics={"other": 2})]),
        ]:
            with self.subTest(submitted=submitted, results=results):
                with self.assertRaises((ValueError, rsikit.EvaluationError)):
                    await strategy.update(submitted, results)
                self.assertEqual(strategy.pending, candidates[0])
                self.assertEqual(len(strategy.history), 1)
        await strategy.update(candidates, [rsikit.Evaluation(valid=True, metrics={"score": 2})])
        self.assertEqual(strategy.best.source, "new")
        with self.assertRaises(ValueError):
            await strategy.update(candidates, [rsikit.Evaluation(valid=True, metrics={"score": 3})])

    async def test_caller_can_retry_failed_evaluation_then_update(self):
        async def propose(parent, history):
            return "new"

        async def unavailable(candidate):
            raise RuntimeError("worker unavailable")

        strategy = rsikit.HillClimb(
            SEED, rsikit.Evaluation(valid=True, metrics={"score": 1}), propose
        )
        candidates = await strategy.generate()
        with self.assertRaisesRegex(RuntimeError, "worker unavailable"):
            await unavailable(candidates[0])
        self.assertEqual(strategy.best.source, SEED)
        self.assertEqual(strategy.pending, candidates[0])
        await strategy.update(
            candidates, [rsikit.Evaluation(valid=False, feedback="evaluation abandoned")]
        )
        self.assertIsNone(strategy.pending)
        self.assertEqual(strategy.history[-1].evaluation.feedback, "evaluation abandoned")

    async def test_caller_cancellation_leaves_search_state_unchanged(self):
        started = asyncio.Event()

        async def propose(parent, history):
            started.set()
            await asyncio.Event().wait()

        strategy = rsikit.HillClimb(
            SEED, rsikit.Evaluation(valid=True, metrics={"score": 1}), propose
        )
        job = asyncio.create_task(strategy.generate())
        await asyncio.wait_for(started.wait(), 2)
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(strategy.best.source, SEED)
        self.assertEqual(len(strategy.history), 1)
        self.assertIsNone(strategy.pending)

    async def test_invalid_baseline_fails_before_generation(self):
        async def propose(parent, history):
            self.fail("must not generate from an invalid baseline")

        for evaluation in [
            rsikit.Evaluation(valid=False),
            rsikit.Evaluation(valid=True, metrics={"other": 1}),
        ]:
            with self.subTest(evaluation=evaluation):
                with self.assertRaises((rsikit.InvalidCandidate, rsikit.EvaluationError)):
                    rsikit.HillClimb(SEED, evaluation, propose)


class ScriptEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_programs_use_evaluation_json_and_keep_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text("def approximate(x):\n    return x\n")
            evaluator = rsikit.LocalEvaluator(ROOT / "examples/sine/evaluate.py", timeout=5)
            result = await evaluator(candidate)
            self.assertTrue(result.valid)
            self.assertGreater(result.metrics["mse"], 0)
            self.assertTrue((root / "stdout.log").exists())
            self.assertTrue((root / "evaluation.json").exists())

    async def test_timeout_and_bad_protocol_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text(SEED)
            script = root / "evaluate.py"
            script.write_text("import time\ntime.sleep(30)\n")
            result = await rsikit.LocalEvaluator(script, timeout=0.1)(candidate)
            self.assertFalse(result.valid)
            self.assertIn("timeout", result.feedback.lower())
            script.write_text("pass\n")
            with self.assertRaises(rsikit.EvaluationError):
                await rsikit.LocalEvaluator(script)(candidate)

    async def test_sine_evaluator_rejects_direct_reference_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text("import math\ndef approximate(x):\n    return math.sin(x)\n")
            evaluation = await rsikit.LocalEvaluator(ROOT / "examples/sine/evaluate.py")(candidate)
            self.assertFalse(evaluation.valid)

    async def test_invalid_encoding_is_an_evaluator_protocol_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text(SEED)
            script = root / "evaluate.py"
            script.write_text(
                "import sys\nfrom pathlib import Path\n"
                "Path(sys.argv[-1]).write_bytes(bytes([255]))\n"
            )
            with self.assertRaises(rsikit.EvaluationError):
                await rsikit.LocalEvaluator(script)(candidate)

    async def test_evaluator_measures_current_source_when_path_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            evaluator = rsikit.LocalEvaluator(ROOT / "examples/sine/evaluate.py")
            candidate.write_text("def approximate(x):\n    return 0\n")
            stamp = candidate.stat().st_mtime
            first = await evaluator(candidate)
            candidate.write_text("def approximate(x):\n    return 1\n")
            os.utime(candidate, (stamp, stamp))
            second = await evaluator(candidate)
            self.assertAlmostEqual(second.metrics["mse"] - first.metrics["mse"], 1)

    async def test_cancellation_stops_evaluator_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text(SEED)
            child = "import time; from pathlib import Path; time.sleep(0.5); Path('leaked').touch()"
            script = root / "evaluate.py"
            script.write_text(
                "import subprocess, sys, time\nfrom pathlib import Path\n"
                f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                "Path('started').touch()\ntime.sleep(30)\n"
            )
            job = asyncio.create_task(rsikit.LocalEvaluator(script)(candidate))
            try:

                async def wait_started():
                    while not (root / "started").exists():
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_started(), 2)
            finally:
                job.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await job
            await asyncio.sleep(0.6)
            self.assertFalse((root / "leaked").exists())


if __name__ == "__main__":
    unittest.main()
