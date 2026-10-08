"""The interactive-to-worker contract, including ownership and native episode output."""

import os
import threading
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from rsikit import Executor, Job, Policy, PolicyDefinition, evaluate, execute
from rsikit.evaluation import InfrastructureError
from tests.test_execution import SOURCE, ProcessEnv


def malformed_result(*args):
    return {"not": "an episode"}


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_instances_round_trip_without_mutating_templates(self):
        class Env(ProcessEnv):
            def __init__(self):
                self.shared = []
                self.closed = False

            def reset(self, *, seed=None, options=None):
                self.shared.clear()
                self.shared.append(("env", seed))
                return super().reset(seed=seed, options=options)

            def close(self):
                self.closed = True

        class Agent(Policy):
            async def reset(self, *, seed=None):
                await super().reset(seed=seed)
                self.shared.append(("policy", seed))

            async def act(self, observation):
                assert self.shared == [("env", 42), ("policy", 42)]
                return 0

            async def close(self):
                self.shared.append("closed")

        env = Env()
        policy = Agent(env.observation_space, env.action_space)
        policy.shared = env.shared
        direct = await evaluate(policy, env, seed=42)
        jobs = [Job(policy, gym.Wrapper(env), seed=42) for _ in range(2)]
        self.assertTrue(all(not job.done for job in jobs))
        completed = await execute(jobs, concurrency=2)
        self.assertEqual({id(job) for job in completed}, {id(job) for job in jobs})
        for job in completed:
            self.assertTrue(job.done)
            self.assertIsNone(job.result.error)
            self.assertEqual(job.result.observations, direct.observations)
            self.assertEqual(job.result.rewards, [1.0, 1.0])
            self.assertNotEqual(job.result.infos[0]["pid"], os.getpid())
        # Workers are reused; the queue may schedule both short jobs on one worker.
        self.assertLessEqual(len({job.result.infos[0]["pid"] for job in jobs}), 2)
        self.assertEqual(env.shared, [("env", 42), ("policy", 42)])
        self.assertFalse(env.closed)

    async def test_validation_and_single_submission(self):
        policy, env = PolicyDefinition(source=SOURCE), ProcessEnv()
        for kwargs in ({"seed": -1}, {"seed": True}, {"seed": 1.5}, {"max_steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Job(policy, env, **kwargs)
        with self.assertRaises(TypeError):
            Job("source", env)
        with self.assertRaises(TypeError):
            Job(policy, object())
        job = Job(policy, env)
        self.assertEqual(await execute([]), [])
        with self.assertRaises(ValueError):
            await execute([job, job])
        self.assertFalse(job.done)
        self.assertIs((await execute([job]))[0], job)
        with self.assertRaises(ValueError):
            await execute([job])

    async def test_unserializable_job_preserves_successful_sibling(self):
        env = ProcessEnv()
        env.lock = threading.Lock()
        bad = Job(PolicyDefinition(source=SOURCE), env)
        good = Job(PolicyDefinition(source=SOURCE), ProcessEnv(), seed=2)
        completed = []
        async with Executor(concurrency=2) as executor:
            with self.assertRaisesRegex(InfrastructureError, "serialize"):
                async for job in executor.iterate([bad, good]):
                    completed.append(job)
        self.assertEqual(completed, [good])
        self.assertFalse(bad.done)
        self.assertTrue(good.done)

    async def test_pickle_arrays_artifacts_and_malformed_response(self):
        class ArrayEnv(ProcessEnv):
            def step(self, action):
                obs, reward, done, truncated, info = super().step(action)
                info.update(
                    array=np.array([1, 2], dtype=np.int16),
                    detail=(np.int16(7), b"\x00\xff"),
                    artifacts={"blob": b"\x00\xff"},
                )
                return obs, reward, done, truncated, info

        job = Job(PolicyDefinition(source=SOURCE), ArrayEnv())
        await execute([job])
        self.assertEqual(job.result.artifacts["blob"], b"\x00\xff")
        np.testing.assert_array_equal(
            job.result.infos[-1]["array"], np.array([1, 2], dtype=np.int16)
        )
        self.assertEqual(job.result.infos[-1]["array"].dtype, np.int16)
        self.assertIsInstance(job.result.infos[-1]["detail"], tuple)
        self.assertIsInstance(job.result.infos[-1]["detail"][0], np.int16)
        with patch("rsikit.execution.worker.run_episode", malformed_result):
            with self.assertRaisesRegex(InfrastructureError, "result"):
                await execute([Job(PolicyDefinition(source=SOURCE), ProcessEnv())])
        with patch("rsikit.execution.worker.MAX_RESULT", 256):
            with self.assertRaises(InfrastructureError):
                await execute([Job(PolicyDefinition(source=SOURCE), ProcessEnv())])

    async def test_source_initialization_is_inside_deadline(self):
        for source in (
            "while True: pass",
            SOURCE + "\n    def __init__(self, *a, **k):\n        while True: pass\n",
        ):
            job = Job(PolicyDefinition(source=source), ProcessEnv())
            await execute([job], episode_timeout=1)
            self.assertTrue(job.done)
            self.assertIn("timeout", job.result.error)

    async def test_worker_bounds_encoded_episode_and_error_responses(self):
        # Exercise the task result bound inside the worker.
        prefix = "import rsikit.execution.worker\nrsikit.execution.worker.MAX_RESULT = 1024\n"
        for body in (
            SOURCE.replace("return 0", "raise ValueError('x' * 2000)"),
            SOURCE + "\nfrom pathlib import Path\nPath('large.bin').write_bytes(b'x' * 2000)",
        ):
            with self.subTest(body=body), self.assertRaisesRegex(InfrastructureError, "exceed"):
                await execute([Job(PolicyDefinition(source=prefix + body), ProcessEnv())])
