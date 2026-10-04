"""Native episode boundaries, independent seeds, and exact workload accounting."""

import unittest

CONSTANT = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.zeros(len(observation), dtype=np.int64)
"""


class UpstreamOceanTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_progress_reports_completed_games_without_extra_episodes(self):
        from research.ocean.evaluator import rollout

        snapshots = []
        result = await rollout(
            CONSTANT,
            [0, 1, 2],
            batch_size=2,
            max_steps=5,
            _on_batch=lambda rows: snapshots.append((len(rows), sum(r["steps"] for r in rows))),
        )
        self.assertEqual(snapshots, [(2, 10), (3, 15)])
        self.assertEqual(result["steps"], 15)

    async def test_fractional_discrete_actions_are_rejected_before_native_step(self):
        from research.ocean.evaluator import rollout
        from rsikit.evaluation import PolicyError

        source = CONSTANT.replace(
            "np.zeros(len(observation), dtype=np.int64)", "np.full(len(observation), 1.5)"
        )
        with self.assertRaises(PolicyError):
            await rollout(source, [0], batch_size=2, max_steps=1)

    async def test_batch_size_does_not_multiply_episode_budget(self):
        from research.ocean.evaluator import rollout

        result = await rollout(CONSTANT, list(range(10)), batch_size=32, max_steps=5, trace=True)
        self.assertEqual(result["steps"], 50)
        self.assertEqual([r["seed"] for r in result["results"]], list(range(10)))
        for row in result["results"]:
            self.assertEqual(row["steps"], 5)
            self.assertEqual(row["ending"], "truncated")
            self.assertEqual(row["actions"], [0] * 5)

    async def test_terminal_signal_survives_upstream_autoreset(self):
        from research.ocean.environment import factory

        env = factory()(num_envs=1, log_interval=1)
        try:
            env.reset(seed=0)
            for _ in range(1100):
                _, _, done, _, infos = env.step([0])
                if infos and infos[0].get("n", 0):
                    self.assertTrue(done[0])
                    break
            else:
                self.fail("Expected the upstream internal time limit to end the game")
        finally:
            env.close()

    async def test_episode_stops_and_final_board_is_preserved(self):
        import numpy as np

        from research.ocean.environment import factory
        from research.ocean.evaluator import rollout

        env = factory()(num_envs=1, log_interval=1)
        try:
            env.reset(seed=[7])
            for step in range(1, 1101):
                obs, _, done, _, _ = env.step([0])
                if done[0]:
                    break
            self.assertLess(step, 1100)
            stats = env.episode_stats()[0]
            final = obs.copy()
            for _ in range(3):
                obs, reward, done, _, _ = env.step([0])
                np.testing.assert_array_equal(obs, final)
                self.assertTrue(done[0])
                self.assertEqual(reward[0], 0)
                self.assertEqual(env.episode_stats()[0], stats)
            env.reset(seed=[7])
            self.assertFalse(env.terminals[0])
            for _ in range(20):
                env.step([0])
            capped_score = env.episode_stats()[0]["merge_score"]
        finally:
            env.close()
        row = (await rollout(CONSTANT, [7], batch_size=32, max_steps=1100))["results"][0]
        self.assertEqual(row["steps"], step)
        self.assertEqual(row["episodes"], 1)
        self.assertEqual(row["ending"], "terminated")
        self.assertEqual(row["score"], stats["merge_score"])
        capped = (await rollout(CONSTANT, [7], max_steps=20))["results"][0]
        self.assertEqual(capped["ending"], "truncated")
        self.assertEqual(capped["score"], capped_score)

    async def test_width_order_and_truncated_scores_match_individual_episodes(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import rollout

        for name, score in (("g2048", "merge_score"), ("breakout", "score")):
            with self.subTest(env=name):
                source = policies(name)[0]._implementation
                kwargs = dict(max_steps=1100, trace=True, env_name=name, score_key=score)
                scalar = await rollout(source, [7, 9, 11], batch_size=1, **kwargs)
                batch = await rollout(source, [7, 9, 11], batch_size=2, **kwargs)
                reverse = await rollout(source, [11, 9, 7], batch_size=32, **kwargs)
                self.assertEqual(scalar, batch)
                self.assertEqual(scalar["results"], reverse["results"][::-1])
                if name == "g2048":
                    self.assertGreater(sum(r["score"] for r in batch["results"]), 0)
                    capped = await rollout(source, [7, 9, 11], batch_size=2, max_steps=20)
                    self.assertTrue(all(r["ending"] == "truncated" for r in capped["results"]))
                    self.assertGreater(sum(r["score"] for r in capped["results"]), 0)

    async def test_breakout_terminal_freezes_before_next_frameskip(self):
        import numpy as np

        from research.ocean.environment import factory

        env = factory("breakout")(num_envs=2, frameskip=32)
        try:
            for seeds in ([7, 9], [9, 7]):
                env.reset(seed=seeds)
                self.assertFalse(env.terminals.any())
                for _ in range(2000):
                    obs, _, done, _, _ = env.step([0, 0])
                    if done.all():
                        break
                self.assertTrue(done.all())
                stats, final = env.episode_stats(), obs.copy()
                obs, rewards, done, _, _ = env.step([0, 0])
                np.testing.assert_array_equal(obs, final)
                np.testing.assert_array_equal(rewards, [0, 0])
                self.assertEqual(env.episode_stats(), stats)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
