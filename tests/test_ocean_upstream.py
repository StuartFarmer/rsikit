"""The evaluator must preserve upstream batch rollouts and their accounting."""

import unittest

CONSTANT = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.zeros(len(observation), dtype=np.int64)
"""


class UpstreamOceanTests(unittest.IsolatedAsyncioTestCase):
    async def test_fractional_discrete_actions_are_rejected_before_native_step(self):
        from research.ocean.evaluator import rollout
        from rsikit.evaluation import PolicyError

        source = CONSTANT.replace(
            "np.zeros(len(observation), dtype=np.int64)", "np.full(len(observation), 1.5)"
        )
        with self.assertRaises(PolicyError):
            await rollout(source, [0], batch_size=2, max_steps=1)

    async def test_each_seed_runs_a_full_fixed_width_batch(self):
        from research.ocean.evaluator import rollout

        result = await rollout(CONSTANT, [7, 9], batch_size=3, max_steps=5, trace=True)
        self.assertEqual(result["steps"], 30)
        self.assertEqual([r["seed"] for r in result["results"]], [7, 9])
        for row in result["results"]:
            self.assertEqual(row["batch_size"], 3)
            self.assertEqual(row["vector_steps"], 5)
            self.assertEqual(row["actions"], [[0, 0, 0]] * 5)

    async def test_breakout_uses_the_same_evaluator_and_counts_all_rewards(self):
        from pufferlib.ocean.breakout.breakout import Breakout

        from research.ocean.evaluator import rollout

        env = Breakout(num_envs=3, seed=7, log_interval=5)
        try:
            env.reset(seed=7)
            expected = 0.0
            for _ in range(5):
                _, rewards, _, _, _ = env.step([0, 0, 0])
                expected += float(rewards.sum()) / 3
        finally:
            env.close()
        result = await rollout(
            CONSTANT, [7], batch_size=3, max_steps=5, env_name="breakout", score_key="return"
        )
        self.assertEqual(result["results"][0]["score"], expected)
        self.assertEqual(result["steps"], 15)

    async def test_fresh_batches_repeat_and_seed_order_does_not_change_results(self):
        from research.ocean.evaluator import rollout

        first = await rollout(CONSTANT, [7, 9], 3, 20, trace=True)
        again = await rollout(CONSTANT, [9, 7], 3, 20, trace=True)
        self.assertEqual(first["results"], again["results"][::-1])

    async def test_completed_episode_scores_match_upstream_log(self):
        from pufferlib.ocean.g2048.g2048 import G2048

        from research.ocean.evaluator import rollout

        env = G2048(num_envs=2, seed=7, log_interval=1100)
        try:
            env.reset(seed=7)
            for _ in range(1100):
                _, _, _, _, info = env.step([0, 0])
            expected = info[0]
        finally:
            env.close()
        result = await rollout(CONSTANT, [7], batch_size=2, max_steps=1100)
        row = result["results"][0]
        self.assertGreater(row["episodes"], 0)
        self.assertEqual(row["score"], expected["merge_score"])
        self.assertEqual(row["metrics"], expected)


if __name__ == "__main__":
    unittest.main()
