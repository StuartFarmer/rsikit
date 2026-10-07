"""Episode histories preserve transitions without owning the caller's instances."""

import unittest

import gymnasium as gym
import numpy as np

from rsikit import Policy, evaluate
from rsikit.episode import Episode
from rsikit.evaluation import Evaluator


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_evaluations_reset_once_with_shared_seed(self):
        resets = []

        class Env(gym.Env):
            observation_space = gym.spaces.Discrete(2)
            action_space = gym.spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                resets.append(("environment", seed))
                self.steps = 0
                return 0, {"seed": seed}

            def step(self, action):
                self.steps += 1
                return 1, 1.0, self.steps == 2, False, {}

            def close(self):
                raise AssertionError("Caller owns cleanup")

        class Agent(Policy):
            async def reset(self, *, seed=None):
                resets.append(("policy", seed))
                self.calls = 0

            async def act(self, observation):
                self.calls += 1
                return 0

            async def close(self):
                raise AssertionError("Caller owns cleanup")

        env = Env()
        policy = Agent(env.observation_space, env.action_space)
        evaluator = Evaluator(max_steps=1)
        for seed in (0, 42, None):
            episode = await evaluator.evaluate(policy, env, seed=seed)
            self.assertEqual(episode.observations, [0, 1])
            self.assertEqual(episode.truncations, [True])
            self.assertEqual((env.steps, policy.calls), (1, 1))
            self.assertEqual(episode.infos[-1]["episode"]["l"], 1)
        self.assertEqual(
            resets,
            [
                ("environment", 0),
                ("policy", 0),
                ("environment", 42),
                ("policy", 42),
                ("environment", None),
                ("policy", None),
            ],
        )

    async def test_function_matches_class_and_rejects_invalid_seeds(self):
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        env = CounterEnv()
        policy = CounterPolicy(env.observation_space, env.action_space)
        direct = await evaluate(policy, env, seed=42, max_steps=1)
        configured = await Evaluator(max_steps=1).evaluate(policy, env, seed=42)
        self.assertEqual(direct.observations, configured.observations)
        self.assertEqual(direct.rewards, [1.0])
        self.assertEqual(direct.truncations, [True])
        self.assertFalse(env.closed or policy.closed)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                await evaluate(policy, env, seed=seed)

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
        episode = await Evaluator().evaluate(policy, env, seed=42)
        self.assertIsInstance(episode, Episode)
        self.assertEqual([obs.tolist() for obs in episode.observations], [[0], [1], [2]])
        self.assertEqual([action.tolist() for action in episode.actions], [[1], [2]])
        self.assertEqual(episode.rewards, [1.0, 2.0])
        self.assertEqual(episode.terminations, [False, True])
        self.assertEqual(episode.truncations, [False, False])
        self.assertEqual([info["count"] for info in episode.infos], [[0], [1], [2]])
        self.assertEqual(episode.infos[-1]["episode"]["r"], 3.0)
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
            policy = Broken(env.observation_space, env.action_space)
            episode = await Evaluator().evaluate(policy, env)
            self.assertEqual(len(episode), steps)
            self.assertEqual(len(episode.observations), steps + 1)
            self.assertIn("candidate broke", episode.error)
            restored = Episode.from_data(episode.encode())
            self.assertEqual(restored.error, episode.error)
            self.assertEqual(restored.rewards, episode.rewards)

    async def test_policy_reset_failure_retains_initial_observation(self):
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        class Broken(CounterPolicy):
            async def reset(self, *, seed=None):
                raise ValueError("reset failed")

        env = CounterEnv()
        policy = Broken(env.observation_space, env.action_space)
        episode = await Evaluator().evaluate(policy, env)
        self.assertIn("reset failed", episode.error)
        self.assertEqual(episode.rewards, [])
        self.assertEqual(len(episode.observations), 1)
        self.assertEqual(Episode.from_data(episode.encode()).error, episode.error)
        self.assertFalse(env.closed)

    async def test_step_cap_and_validation(self):
        from tests.test_inner_loop import CounterEnv, CounterPolicy

        env = CounterEnv()
        policy = CounterPolicy(env.observation_space, env.action_space)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                Evaluator(max_steps=limit)
        episode = await Evaluator(max_steps=1).evaluate(policy, env)
        self.assertEqual(episode.rewards, [1.0])
        self.assertEqual(episode.terminations, [False])
        self.assertEqual(episode.truncations, [True])
        self.assertEqual(episode.infos[0], {"private": "not for policy"})
        self.assertEqual(episode.infos[-1]["episode"]["r"], 1.0)
        self.assertFalse(env.closed or policy.closed)


if __name__ == "__main__":
    unittest.main()
