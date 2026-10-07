"""Episode histories preserve transitions without owning the caller's instances."""

import unittest

import gymnasium as gym
import numpy as np

from rsikit import Policy
from rsikit.episode import Episode
from rsikit.evaluation import Evaluator


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_snapshots_and_caller_owned_lifecycle(self):
        class ReusingEnv(gym.Env):
            observation_space = gym.spaces.Box(0, 10, (1,), dtype=np.int64)
            action_space = gym.spaces.Box(0, 10, (1,), dtype=np.int64)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.observation = np.array([0])
                self.info = {"count": [0]}
                return self.observation, self.info

            def step(self, action):
                self.observation += 1
                self.info["count"][0] += 1
                action[0] = 9  # Even mutation during step must not rewrite history.
                return (
                    self.observation,
                    float(self.observation[0]),
                    self.observation[0] == 2,
                    False,
                    self.info,
                )

            def close(self):
                self.observation[:] = -1

        class ReusingPolicy(Policy):
            async def reset(self, *, seed=None):
                await super().reset(seed=seed)
                self.action = np.array([0])

            async def act(self, observation):
                self.action[:] = observation + 1
                observation[:] = 10
                return self.action

            async def close(self):
                self.action[:] = -1

        env = ReusingEnv()
        policy = ReusingPolicy(env.observation_space, env.action_space)
        observation, info = env.reset(seed=42)
        await policy.reset(seed=43)

        def unexpected_reset(*args, **kwargs):
            raise AssertionError("Evaluator must not reset caller-owned instances")

        env.reset = policy.reset = unexpected_reset
        episode = await Evaluator(env, policy).run(observation, info=info)
        self.assertIsInstance(episode, Episode)
        self.assertEqual([obs.tolist() for obs in episode.observations], [[0], [1], [2]])
        self.assertEqual([action.tolist() for action in episode.actions], [[1], [2]])
        self.assertEqual(episode.rewards, [1.0, 2.0])
        self.assertEqual(episode.terminations, [False, True])
        self.assertEqual(episode.truncations, [False, False])
        self.assertEqual(episode.infos, [{"count": [0]}, {"count": [1]}, {"count": [2]}])
        self.assertEqual((len(episode), episode.total_reward), (2, 3.0))
        self.assertEqual(episode.final_step[1:4], (2.0, True, False))
        self.assertEqual(env.observation.tolist(), [2])
        self.assertEqual(policy.action.tolist(), [9])
        env.close()
        await policy.close()
        self.assertEqual(episode.observations[-1].tolist(), [2])
        self.assertEqual(episode.actions[-1].tolist(), [2])

    async def test_failed_attempt_preserves_completed_steps_and_zero_step_errors(self):
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        for steps in (0, 1):

            class Broken(CounterPolicy):
                async def act(self, observation):
                    if self.calls == steps:
                        raise ValueError("candidate broke")
                    return await super().act(observation)

            env = CounterEnv()
            observation, info = env.reset()
            policy = Broken(env.observation_space, env.action_space)
            await policy.reset()
            episode = await Evaluator(env, policy).run(observation, info=info)
            self.assertEqual(len(episode), steps)
            self.assertEqual(len(episode.observations), steps + 1)
            self.assertIn("candidate broke", episode.error)
            restored = Episode.from_data(episode.encode())
            self.assertEqual(restored.error, episode.error)
            self.assertEqual(restored.rewards, episode.rewards)

    async def test_initialization_failure_is_an_empty_episode(self):
        from rsikit.evaluation import _run_episode
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        class Broken(CounterPolicy):
            def __init__(self, *args, **kwargs):
                raise ValueError("construction failed")

        env = CounterEnv()
        episode = await _run_episode(lambda: env, Broken)
        self.assertIn("construction failed", episode.error)
        self.assertEqual(episode.rewards, [])

        self.assertEqual(Episode.from_data(episode.encode()).error, episode.error)
        self.assertTrue(env.closed)

    async def test_candidate_cleanup_failure_preserves_its_trajectory(self):
        from rsikit.evaluation import _run_episode
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        class BrokenClose(CounterPolicy):
            async def close(self):
                raise ValueError("close failed")

        episode = await _run_episode(CounterEnv, BrokenClose)
        self.assertEqual(episode.rewards, [1.0, 2.0])
        self.assertIn("close failed", episode.error)

    async def test_step_cap_and_validation(self):
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        env = CounterEnv()
        policy = CounterPolicy(env.observation_space, env.action_space)
        observation, _ = env.reset()
        await policy.reset()
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                Evaluator(env, policy, max_steps=limit)
        episode = await Evaluator(env, policy, max_steps=1).run(observation)
        self.assertEqual(episode.rewards, [1.0])
        self.assertEqual(episode.terminations, [False])
        self.assertEqual(episode.truncations, [True])
        self.assertEqual(episode.infos, [{}, {}])
        self.assertFalse(env.closed or policy.closed)


if __name__ == "__main__":
    unittest.main()
