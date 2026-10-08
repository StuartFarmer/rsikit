"""Huey evaluation, trajectories, artifacts, and signal-based timeouts."""

import asyncio
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym

from rsikit import Executor, Job, PolicyDefinition
from rsikit.evaluation import InfrastructureError, PolicyError


class ProcessEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)
    instructions = "default instructions"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return 0, {"pid": os.getpid(), "seed": seed}

    def step(self, action):
        self.steps += 1
        return 1, 1.0, self.steps == 2, False, {}


SOURCE = """import os, json
from pathlib import Path
from rsikit import Policy
count = 0
class Solution(Policy):
    async def reset(self, *, seed=None):
        global count
        count += 1
        Path("state.json").write_text(json.dumps([os.getpid(), count, seed, self.instructions]))
    async def act(self, observation):
        print("candidate output")
        return 0
"""


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_darwin_cleanup_waits_for_exiting_worker(self):
        from research.ocean.evaluator import PanelEvaluator

        # The separate OCEAN panel backend still owns child processes.
        for exits in (True, False):
            with (
                self.subTest(exits=exits),
                tempfile.TemporaryDirectory() as directory,
                patch("research.ocean.evaluator.metadata", return_value={}),
            ):
                executor = PanelEvaluator(directory)
                process = Mock(pid=12345)
                process.is_alive.return_value = True

                def join(timeout=None):
                    process.is_alive.return_value = not exits

                process.join.side_effect = join
                receiving, sending = Mock(), Mock()
                receiving.recv_bytes.return_value = pickle.dumps({"results": [], "steps": 0})
                executor._context = Mock()
                executor._context.Process.return_value = process
                executor._context.Pipe.return_value = receiving, sending
                with (
                    patch("sys.platform", "darwin"),
                    patch("os.killpg", side_effect=PermissionError(1, "Operation not permitted")),
                ):
                    if exits:
                        await executor._panel(SOURCE, [1])
                        process.close.assert_called_once()
                        receiving.close.assert_called_once()
                    else:
                        with self.assertRaises(PermissionError):
                            await executor._panel(SOURCE, [1])

    async def test_generated_traceback_preserves_chain_without_locals(self):
        import io

        from rich.console import Console

        from rsikit import Run

        source = SOURCE.replace(
            "return 0",
            'secret = "do-not-print-locals"\n        raise RuntimeError("act failed " + "x" * 5000)',
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            Run.create(
                name="traceback", path=Path(directory) / "run", console=Console(file=io.StringIO())
            ) as run,
        ):
            async with Executor(episode_timeout=10) as executor:
                results = await self.collect(executor, source)
            diagnostic = results[0][2].error
            self.assertIn("candidate.py", diagnostic)
            self.assertIn("in act", diagnostic)
            self.assertIn("x" * 5000, diagnostic)
            self.assertNotIn("do-not-print-locals", diagnostic)
            self.assertIn("candidate.py", (run.path / "run.log").read_text())

    async def collect(self, executor, source=SOURCE, seeds=(1,), environment=None):
        return [
            (job.policy.id, job.seed, job.result)
            async for job in executor.iterate(
                [
                    Job(PolicyDefinition(source=source), environment or ProcessEnv(), seed=seed)
                    for seed in seeds
                ]
            )
        ]

    async def test_reused_workers_trajectory_and_large_result(self):
        import json

        async with Executor(concurrency=2, episode_timeout=10) as executor:
            results = await self.collect(executor, seeds=range(4))
            pids = []
            for _, seed, episode in results:
                pid, count, policy_seed, instructions = json.loads(episode.artifacts["state.json"])
                pids.append(pid)
                self.assertEqual(
                    (count, policy_seed, instructions), (1, seed, ProcessEnv.instructions)
                )
                self.assertEqual(episode.infos[0], {"pid": pid, "seed": seed})
                self.assertNotEqual(pid, os.getpid())
                self.assertEqual(episode.observations, [0, 1, 1])
                self.assertEqual(episode.rewards, [1.0, 1.0])
                self.assertIn(b"candidate output", episode.artifacts["episode.log"])
            self.assertLessEqual(len(set(pids)), 2)
            large = SOURCE + '\nPath("large.bin").write_bytes(b"x" * 2_000_000)\n'
            result = await asyncio.wait_for(self.collect(executor, large), 10)
            self.assertEqual(len(result[0][2].artifacts["large.bin"]), 2_000_000)

    async def test_seeds_instructions_and_cap(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.py"
            path.write_text(SOURCE)
            policy = PolicyDefinition.from_file(path)
            with ProcessEnv() as environment:
                async with Executor() as executor:
                    for seed, instructions in ((5, "override"), (None, "")):
                        results = [
                            job.result
                            async for job in executor.iterate(
                                [
                                    Job(
                                        policy,
                                        environment,
                                        seed=seed,
                                        instructions=instructions,
                                        max_steps=1,
                                    )
                                ]
                            )
                        ]
                        (episode,) = results
                        self.assertEqual(
                            json.loads(episode.artifacts["state.json"])[1:],
                            [1, seed, instructions],
                        )
                        self.assertEqual(episode.infos[0]["seed"], seed)
                        self.assertEqual(episode.truncations, [True])

    async def test_errors_and_successful_siblings(self):
        async with Executor(concurrency=2, episode_timeout=3) as executor:
            for source, error in [
                (SOURCE.replace("return 0", "return 99"), PolicyError),
                ("raise ValueError('load failed')", PolicyError),
                (SOURCE.replace("return 0", "raise RuntimeError('act failed')"), PolicyError),
            ]:
                with self.subTest(source=source):
                    if error is InfrastructureError:
                        with self.assertRaises(error):
                            await self.collect(executor, source)
                    else:
                        results = await self.collect(executor, source)
                        self.assertIsNotNone(results[0][2].error)
            good = []
            bad = Job(
                PolicyDefinition(source=SOURCE.replace("return 0", "while True: pass")),
                ProcessEnv(),
                seed=0,
            )
            good_job = Job(PolicyDefinition(source=SOURCE), ProcessEnv(), seed=1)
            async for job in executor.iterate([bad, good_job]):
                good.append(job)
            self.assertIs(good[0], good_job)
            self.assertIn("timeout", bad.result.error)
            self.assertEqual(len(await self.collect(executor)), 1)

    async def test_invalid_policy_methods_are_repairable(self):
        async with Executor(episode_timeout=10) as executor:
            for source in (
                SOURCE.replace("async def act", "def act"),
                SOURCE.replace("act(self, observation)", "act(self)"),
                SOURCE.replace("reset(self, *, seed=None)", "reset(self)"),
            ):
                with self.subTest(source=source):
                    results = await self.collect(executor, source)
                    self.assertIsNotNone(results[0][2].error)

    async def test_constructor_and_cleanup_failures_preserve_evidence(self):
        class ClosingEnv(ProcessEnv):
            def close(self):
                Path("environment.closed").write_text("closed")

        constructor_failure = (
            SOURCE
            + "\n    def __init__(self, *args, **kwargs):\n        raise ValueError('construction failed')\n"
        )
        cleanup_failure = (
            SOURCE + "\n    async def close(self):\n        raise ValueError('cleanup failed')\n"
        )
        async with Executor(episode_timeout=10) as executor:
            for source, error, length in (
                (constructor_failure, "construction failed", 0),
                (cleanup_failure, "cleanup failed", 2),
                (
                    cleanup_failure.replace("return 0", "raise ValueError('act failed')"),
                    "act failed",
                    0,
                ),
            ):
                with self.subTest(error=error):
                    ((_, _, episode),) = await self.collect(
                        executor, source, environment=ClosingEnv()
                    )
                    self.assertIn(error, episode.error)
                    self.assertEqual(len(episode), length)
                    self.assertEqual(episode.artifacts["environment.closed"], b"closed")
                    episode.encode()
                    if error == "act failed":
                        self.assertNotIn("cleanup failed", episode.error)
                        self.assertIn(b"cleanup failed", episode.artifacts["episode.log"])

            class BrokenEnv(ProcessEnv):
                def step(self, action):
                    raise RuntimeError("environment failed")

            with self.assertRaisesRegex(InfrastructureError, "environment failed"):
                await self.collect(executor, cleanup_failure, environment=BrokenEnv())

    async def test_worker_exit_raises_instead_of_waiting_forever(self):
        async with Executor() as executor:
            with self.assertRaisesRegex(InfrastructureError, "worker exited"):
                await asyncio.wait_for(self.collect(executor, "import os; os._exit(17)"), 5)

    def test_options_validated(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Executor(episode_timeout=value)


if __name__ == "__main__":
    unittest.main()
