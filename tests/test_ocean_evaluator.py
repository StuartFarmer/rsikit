"""Real native rollouts catch seed, batch, action and process-boundary regressions."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from rsikit import Policy


class OceanEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_seed_order_and_reference_preserve_actions_and_scores(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator, rollout

        policy = policies()[1]
        seeds = [19, 0, 11]
        scalar = await rollout(policy._implementation, seeds, 3, 40, trace=True)
        batch = await rollout(policy._implementation, seeds, 3, 40, trace=True)
        reverse = await rollout(policy._implementation, seeds[::-1], 3, 40, trace=True)
        self.assertEqual(scalar["results"], batch["results"])
        self.assertEqual(scalar["results"], reverse["results"][::-1])
        self.assertEqual(scalar["steps"], sum(r["steps"] for r in scalar["results"]))
        self.assertTrue(all(len(r["actions"]) == r["vector_steps"] for r in scalar["results"]))
        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(
                directory, mode="reference", batch_size=3, max_steps=40
            ) as evaluator:
                reference = await evaluator.submit(policy, seeds)
            self.assertEqual(reference["status"], "ok", reference["error"])
            for expected, actual in zip(scalar["results"], reference["results"]):
                self.assertEqual(actual, {k: v for k, v in expected.items() if k != "actions"})

    async def test_failure_is_not_a_partial_panel_and_next_candidate_recovers(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator

        invalid = Policy.from_text("""
import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.full(len(observation), np.nan)
""")
        hanging = Policy.from_text("""
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        while True: pass
""")
        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(directory, max_steps=3, timeout=5) as evaluator:
                bad = await evaluator.submit(invalid, [0, 1])
                self.assertEqual(bad["status"], "failed")
                self.assertEqual(bad["results"], [])
                timed = await evaluator.submit(hanging, [0])
                self.assertEqual(timed["status"], "failed")
                self.assertIn("exceeded", timed["error"])
                good = await evaluator.submit(policies()[0], [0, 1])
                self.assertEqual(good["status"], "ok", good["error"])
                self.assertEqual(len(good["results"]), 2)
                self.assertEqual((evaluator.queued, evaluator.running), (0, 0))
                self.assertGreaterEqual(good["persisted"], good["received"])
                self.assertGreaterEqual(good["received"], good["started"])
                events = [
                    json.loads(s)
                    for s in Path(directory, "evaluations.jsonl").read_text().splitlines()
                ]
                self.assertEqual([e["status"] for e in events], ["failed", "failed", "ok"])
                measurement = (await evaluator.evaluate([policies()[0]], seeds=[7]))[
                    policies()[0].id
                ]
                self.assertEqual(set(measurement), {7})

    async def test_episode_feedback_retains_trajectory_before_candidate_crash(self):
        from research.ocean.evaluator import PanelEvaluator
        from rsikit.episode import Episode

        policy = Policy.from_text("""
import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def reset(self, *, seed=None):
        await super().reset(seed=seed)
        self.calls = 0
    async def act(self, observation):
        if self.calls:
            raise ValueError("second action failed")
        self.calls += 1
        return np.zeros(len(observation), dtype=np.int32)
""")
        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(directory, max_steps=3, timeout=10) as evaluator:
                result = await evaluator.evaluate([policy], seeds=[0, 1])
        for episode in result[policy.id].values():
            self.assertEqual(len(episode), 1)
            self.assertEqual(len(episode.observations), 2)
            self.assertIn("second action failed", episode.error)
            self.assertEqual(Episode.from_data(episode.encode()).error, episode.error)

    async def test_backend_failure_propagates_instead_of_becoming_an_episode_error(self):
        from unittest.mock import AsyncMock, patch

        from research.ocean.evaluator import PanelEvaluator

        policy = Policy.from_text(
            "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation): return 0\n"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("research.ocean.evaluator.metadata", return_value={"protocol": "test"}),
        ):
            async with PanelEvaluator(directory) as evaluator:
                evaluator._panel = AsyncMock(side_effect=RuntimeError("backend unavailable"))
                with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                    await evaluator.evaluate([policy], seeds=[0])

    async def test_invalid_inputs_and_cancellation_release_capacity(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator, rollout

        for seeds in ([], [1, 1], [-1], [2**32], [True]):
            with self.assertRaises(ValueError):
                await rollout(policies()[0]._implementation, seeds, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(directory, timeout=5, max_steps=2000) as evaluator:
                task = asyncio.create_task(evaluator.submit(policies()[0], list(range(32))))
                await asyncio.sleep(0.02)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual((evaluator.queued, evaluator.running), (0, 0))

    async def test_queued_cancellation_is_recorded_without_claiming_service(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator

        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(directory, max_steps=1000) as evaluator:
                first = asyncio.create_task(evaluator.submit(policies()[0], [0], job_id="first"))
                while evaluator.running == 0:
                    await asyncio.sleep(0)
                queued = asyncio.create_task(evaluator.submit(policies()[0], [1], job_id="queued"))
                while evaluator.queued == 0:
                    await asyncio.sleep(0)
                queued.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await queued
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                event = next((e for e in evaluator.events if e["job_id"] == "queued"), None)
                self.assertIsNotNone(event)
                self.assertEqual(event["status"], "cancelled")
                self.assertIsNone(event["started"])
                self.assertEqual(event["service_seconds"], 0)
                self.assertTrue(Path(directory, "measurements", "queued.json").exists())


if __name__ == "__main__":
    unittest.main()
