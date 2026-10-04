"""Native Maze map isolation, episode memory and terminal accounting."""

import sys
import unittest
from pathlib import Path

import numpy as np

from research.ocean.baselines import policies
from research.ocean.evaluator import rollout
from research.ocean.maze import Maze, verify_splits


@unittest.skipUnless(
    (Path(sys.prefix) / "share/rsikit-maze/maze.so").exists(), "run scripts/install_maze.py"
)
class MazeTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_policy_is_independent_of_batch_width_and_seed_order(self):
        source = policies("maze")[-1]._implementation
        kwargs = dict(env_name="maze", score_key="return", max_steps=450)
        scalar = await rollout(source, [7, 9, 11], batch_size=1, **kwargs)
        batch = await rollout(source, [7, 9, 11], batch_size=2, **kwargs)
        reverse = await rollout(source, [11, 9, 7], batch_size=32, **kwargs)
        self.assertEqual(scalar, batch)
        self.assertEqual(scalar["results"], reverse["results"][::-1])
        self.assertTrue(
            all(row["score"] == 1 and row["ending"] == "terminated" for row in batch["results"])
        )
        self.assertEqual(len({row["metrics"]["map_id"] for row in batch["results"]}), 3)
        capped = await rollout(
            source, [7], batch_size=32, max_steps=5, env_name="maze", score_key="return"
        )
        self.assertEqual(capped["steps"], 5)
        self.assertEqual(capped["results"][0]["ending"], "truncated")

    async def test_native_timeout_freezes_without_autoreset_and_reset_clears_state(self):
        env = Maze(num_envs=2)
        try:
            env.reset(seed=[7, 9])
            with self.assertRaises(ValueError):
                env.step([1.5, 2])
            for _ in range(450):
                observation, reward, done, _, _ = env.step([0, 0])
            self.assertTrue(done.all())
            self.assertEqual(reward.tolist(), [0, 0])
            frozen, stats = observation.copy(), env.episode_stats()
            for _ in range(3):
                env.step([1, 1])
            np.testing.assert_array_equal(env.observations, frozen)
            self.assertEqual(env.episode_stats(), stats)
            env.reset(seed=[9, 7])
            self.assertFalse(env.terminals.any())
            self.assertEqual(env.steps.tolist(), [0, 0])
            self.assertEqual(
                [r["map_id"] for r in env.episode_stats()], [r["map_id"] for r in stats][::-1]
            )
        finally:
            env.close()

    async def test_all_default_panels_use_distinct_map_identities(self):
        from research.meta_ocean.experiment import Config

        report = verify_splits(Config())
        rows = [row for panel in report.values() for row in panel]
        self.assertEqual(len(rows), len({row["map_id"] for row in rows}))
        self.assertEqual(len(report["calibration"]), 64)
        self.assertEqual(len(report["test/audit"]), 2560)
