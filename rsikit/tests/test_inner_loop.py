"""Small runnable checks of the new inner-loop contract."""

import asyncio
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from rsikit.envs import CirclePackingEnv
from rsikit.episode import EpisodeError, InfrastructureError, run_episode
from rsikit.examples.cartpole import INSTRUCTIONS
from rsikit.examples.cartpole import Solution as CartPolePolicy
from rsikit.examples.circle_packing.initial import Solution as PackingPolicy
from rsikit.policy import Policy


class CounterEnv(gym.Env):
    instructions = "Count up.\nPublic task: café."

    def __init__(self):
        self.observation_space = spaces.Box(0, 10, shape=(1,), dtype=np.int64)
        self.action_space = spaces.Discrete(3)
        self.closed = False
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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
        observation[0] = 9  # The runner protects the environment and saved observations.
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
            env_seed=1,
            policy_seed=2,
            max_steps=kwargs.pop("max_steps", 10),
            **kwargs,
        )

    async def test_lifecycle_instructions_snapshots_and_limits(self):
        captured = []

        def factory(*args, **kwargs):
            policy = CounterPolicy(*args, **kwargs)
            captured.append(policy)
            return policy

        episode = await self.run_counter(factory)
        self.assertEqual((episode.status, episode.length, episode.return_), ("completed", 2, 3.0))
        self.assertEqual([t.action for t in episode.transitions], [0, 1])
        self.assertEqual([t.observation[0] for t in episode.transitions], [0, 1])
        self.assertEqual(episode.initial_observation[0], 0)
        self.assertTrue(episode.transitions[-1].terminated)
        self.assertEqual(episode.instructions, captured[0].seen_instructions)
        self.assertTrue(captured[0].closed)
        for text in ("Override\nλ", ""):
            result = await self.run_counter(factory, instructions=text)
            self.assertEqual(result.instructions, text)
            self.assertEqual(captured[-1].seen_instructions, text)
            self.assertEqual(result.transitions[0].action, 0)
        limited = await self.run_counter(max_steps=1)
        self.assertTrue(limited.transitions[-1].truncated)
        self.assertFalse(limited.transitions[-1].terminated)
        with self.assertRaisesRegex(ValueError, "instructions"):
            await run_episode(
                lambda: gym.make("CartPole-v1"),
                CounterPolicy,
                env_seed=1,
                policy_seed=2,
                max_steps=1,
            )

    async def test_policy_environment_and_cleanup_failures(self):
        class Invalid(CounterPolicy):
            async def act(self, observation):
                return 50

        env = CounterEnv()
        invalid = await run_episode(lambda: env, Invalid, env_seed=1, policy_seed=2, max_steps=3)
        self.assertEqual((invalid.status, env.steps), ("invalid_action", 0))
        self.assertEqual(invalid.attempted_action, 50)
        self.assertTrue(env.closed)

        class Broken(CounterPolicy):
            async def act(self, observation):
                if self.calls:
                    raise RuntimeError("policy failed")
                return await super().act(observation)

            async def close(self):
                raise RuntimeError("cleanup also failed")

        failed = await self.run_counter(Broken)
        self.assertEqual((failed.status, failed.length), ("policy_error", 1))
        self.assertIn("policy failed", failed.failure)
        self.assertEqual(len(failed.cleanup_errors), 1)

        class BrokenEnv(CounterEnv):
            def step(self, action):
                raise RuntimeError("environment failed")

        with self.assertRaisesRegex(EpisodeError, "environment failed") as raised:
            await run_episode(BrokenEnv, CounterPolicy, env_seed=1, policy_seed=2, max_steps=3)
        self.assertEqual(raised.exception.episode.length, 0)

        class BackendFailure(CounterPolicy):
            async def reset(self, *, seed=None):
                raise InfrastructureError("worker unavailable")

        with self.assertRaisesRegex(EpisodeError, "worker unavailable"):
            await self.run_counter(BackendFailure)

    async def test_cancellation_closes_both_sides(self):
        env, policies = CounterEnv(), []
        started = asyncio.Event()

        class Waiting(CounterPolicy):
            async def act(self, observation):
                started.set()
                await asyncio.Event().wait()

        def factory(*args, **kwargs):
            policies.append(Waiting(*args, **kwargs))
            return policies[-1]

        task = asyncio.create_task(
            run_episode(lambda: env, factory, env_seed=1, policy_seed=2, max_steps=3)
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(env.closed and policies[0].closed)

        close_started = asyncio.Event()

        class SlowClose(CounterPolicy):
            async def close(self):
                close_started.set()
                await asyncio.Event().wait()

        env = CounterEnv()
        task = asyncio.create_task(
            run_episode(lambda: env, SlowClose, env_seed=1, policy_seed=2, max_steps=3)
        )
        await close_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(env.closed)

    async def test_packing_and_native_cartpole_use_same_runner(self):
        check_env(CirclePackingEnv(), skip_render_check=True)
        packing = await run_episode(
            CirclePackingEnv, PackingPolicy, env_seed=1, policy_seed=2, max_steps=1
        )
        self.assertEqual((packing.status, packing.length), ("completed", 1))
        self.assertAlmostEqual(packing.return_, 1.0)
        self.assertTrue(packing.transitions[-1].info["feasible"])

        class Overlapping(PackingPolicy):
            async def act(self, observation):
                return np.tile([0.5, 0.5, 0.1], (10, 1))

        bad_geometry = await run_episode(
            CirclePackingEnv, Overlapping, env_seed=1, policy_seed=2, max_steps=1
        )
        self.assertEqual((bad_geometry.status, bad_geometry.return_), ("completed", 0.0))
        self.assertFalse(bad_geometry.transitions[-1].info["feasible"])
        episodes = [
            await run_episode(
                lambda: gym.make("CartPole-v1"),
                CartPolePolicy,
                env_seed=1,
                policy_seed=2,
                max_steps=50,
                instructions=INSTRUCTIONS,
            )
            for _ in range(2)
        ]
        self.assertEqual(episodes[0].return_, episodes[1].return_)
        self.assertEqual(episodes[0].status, "completed")
        self.assertTrue(1 < episodes[0].length <= 50)
        self.assertEqual(episodes[0].instructions, INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
