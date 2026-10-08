"""Small runnable checks of the new inner-loop contract."""

import asyncio
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from examples.cartpole import Solution as CartPolePolicy
from examples.circle_packing.initial import Solution as PackingPolicy
from rsikit.envs import CirclePackingEnv
from rsikit.evaluation import Evaluator, InfrastructureError
from rsikit.policy import Policy
from tests.helpers import run_episode


class CounterEnv(gym.Env):
    instructions = "Count up.\nPublic task: café."

    def __init__(self):
        self.observation_space = spaces.Box(0, 10, shape=(1,), dtype=np.int64)
        self.action_space = spaces.Discrete(3)
        self.closed = False
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.value = np.array([0], dtype=np.int64)
        return self.value, {"private": "not for policy"}

    def step(self, action):
        self.steps += 1
        self.value[0] += 1
        return self.value, float(self.value[0]), self.steps == 2, False, {}

    def close(self):
        self.closed = True


class CounterPolicy(Policy):
    async def reset(self, *, seed=None):
        await super().reset(seed=seed)
        self.calls = 0
        self.seen_instructions = self.instructions
        self.closed = False

    async def act(self, observation):
        observation[0] = 9  # The runner protects the environment from policy mutation.
        self.instructions = "Policy-owned prompt changed"
        action = self.calls
        self.calls += 1
        return action

    async def close(self):
        self.closed = True


class InnerLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_counter(self, policy=CounterPolicy, **kwargs):
        return await run_episode(
            CounterEnv,
            policy,
            seed=1,
            max_steps=kwargs.pop("max_steps", 10),
            **kwargs,
        )

    async def test_lifecycle_instructions_and_limits(self):
        captured = []

        def factory(*args, **kwargs):
            policy = CounterPolicy(*args, **kwargs)
            captured.append(policy)
            return policy

        observation, reward, terminated, truncated, info = await self.run_counter(factory)
        self.assertEqual(observation[0], 2)
        self.assertEqual((reward, terminated, truncated), (2.0, True, False))
        self.assertEqual((info["episode"]["r"], info["episode"]["l"]), (3.0, 2))
        self.assertEqual(captured[0].calls, 2)
        self.assertEqual(captured[0].seen_instructions, CounterEnv.instructions)
        self.assertTrue(captured[0].closed)
        for text in ("Override\nλ", ""):
            await self.run_counter(factory, instructions=text)
            self.assertEqual(captured[-1].seen_instructions, text)
            self.assertEqual(captured[-1].calls, 2)
        _, _, terminated, truncated, info = await self.run_counter(max_steps=1)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["episode"]["l"], 1)
        wrapped = gym.Wrapper(CounterEnv())
        wrapped.instructions = "Wrapper-owned instructions"
        await run_episode(lambda: wrapped, factory)
        self.assertEqual(captured[-1].seen_instructions, wrapped.instructions)

    async def test_policy_environment_and_cleanup_failures(self):
        class Invalid(CounterPolicy):
            async def act(self, observation):
                return 50

        env = CounterEnv()
        episode = await Evaluator().evaluate(Invalid(env.observation_space, env.action_space), env)
        self.assertIn("outside action_space", episode.error)
        self.assertEqual(env.steps, 0)
        self.assertFalse(env.closed)

        class Broken(CounterPolicy):
            async def act(self, observation):
                if self.calls:
                    raise RuntimeError("policy failed")
                return await super().act(observation)

        env = CounterEnv()
        episode = await Evaluator().evaluate(Broken(env.observation_space, env.action_space), env)
        self.assertIn("policy failed", episode.error)
        self.assertEqual(episode.rewards, [1.0])

        class BrokenEnv(CounterEnv):
            def step(self, action):
                raise RuntimeError("environment failed")

        with self.assertRaisesRegex(RuntimeError, "environment failed"):
            await run_episode(BrokenEnv, CounterPolicy)

        class BackendFailure(CounterPolicy):
            async def reset(self, *, seed=None):
                raise InfrastructureError("worker unavailable")

        with self.assertRaisesRegex(InfrastructureError, "worker unavailable"):
            await self.run_counter(BackendFailure)

    async def test_environment_invalid_action_is_a_policy_failure(self):
        class MaskedEnv(CounterEnv):
            def step(self, action):
                raise gym.error.InvalidAction("Action masked in this state")

        env = MaskedEnv()
        episode = await Evaluator().evaluate(
            CounterPolicy(env.observation_space, env.action_space), env
        )
        self.assertIn("Action masked", episode.error)
        self.assertFalse(env.closed)

        class BrokenEnv(CounterEnv):
            def step(self, action):
                raise ValueError("Broken environment calculation")

        with self.assertRaisesRegex(ValueError, "Broken environment calculation"):
            await run_episode(BrokenEnv, CounterPolicy)

    async def test_cancellation_leaves_cleanup_with_caller(self):
        env = CounterEnv()
        started = asyncio.Event()

        class Waiting(CounterPolicy):
            async def act(self, observation):
                started.set()
                await asyncio.Event().wait()

        policy = Waiting(env.observation_space, env.action_space)
        task = asyncio.create_task(Evaluator().evaluate(policy, env, seed=1))
        try:
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(env.closed or policy.closed)
        finally:
            await policy.close()
            env.close()

    async def test_custom_and_builtin_gymnasium_environments(self):
        check_env(CirclePackingEnv(), skip_render_check=True)
        _, _, terminated, _, info = await run_episode(CirclePackingEnv, PackingPolicy)
        self.assertTrue(terminated)
        self.assertEqual(info["episode"]["l"], 1)
        self.assertAlmostEqual(info["episode"]["r"], 1.0)
        self.assertTrue(info["feasible"])

        class Overlapping(PackingPolicy):
            async def act(self, observation):
                return np.tile([0.5, 0.5, 0.1], (10, 1))

        _, _, _, _, info = await run_episode(CirclePackingEnv, Overlapping)
        self.assertEqual(info["episode"]["r"], 0.0)
        self.assertFalse(info["feasible"])
        episodes = [await run_episode("CartPole-v1", CartPolePolicy, seed=1) for _ in range(2)]
        self.assertEqual(episodes[0][4]["episode"]["r"], episodes[1][4]["episode"]["r"])
        self.assertTrue(episodes[0][2] or episodes[0][3])
        self.assertTrue(1 < episodes[0][4]["episode"]["l"] <= 500)

        class RandomPolicy(Policy):
            async def act(self, observation):
                assert self.instructions == ""
                return self.action_space.sample()

        _, _, terminated, truncated, info = await run_episode("FrozenLake-v1", RandomPolicy, seed=1)
        self.assertTrue(terminated or truncated)
        self.assertTrue(0 < info["episode"]["l"] <= 100)


if __name__ == "__main__":
    unittest.main()
