"""Fresh local episodes, real deadlines and ordinary process cleanup."""

import asyncio
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym

from rsikit import Executor, PolicyDefinition
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
        from rsikit.episode import Episode

        for ocean in (False, True):
            for exits in (True, False):
                with (
                    self.subTest(ocean=ocean, exits=exits),
                    tempfile.TemporaryDirectory() as directory,
                    patch("research.ocean.evaluator.metadata", return_value={}),
                ):
                    executor = PanelEvaluator(directory) if ocean else Executor()
                    process = Mock(pid=12345)
                    process.is_alive.return_value = True

                    def join(timeout=None):
                        process.is_alive.return_value = not exits

                    process.join.side_effect = join
                    receiving, sending = Mock(), Mock()
                    receiving.recv_bytes.return_value = (
                        b'{"results": [], "steps": 0}' if ocean else pickle.dumps(Episode())
                    )
                    executor._context = Mock()
                    executor._context.Process.return_value = process
                    executor._context.Pipe.return_value = receiving, sending
                    with (
                        patch("sys.platform", "darwin"),
                        patch(
                            "os.killpg", side_effect=PermissionError(1, "Operation not permitted")
                        ),
                    ):
                        evaluation = (
                            executor._panel(SOURCE, [1])
                            if ocean
                            else executor._evaluate(SOURCE, b"", 1)
                        )
                        if exits:
                            await evaluation
                            process.close.assert_called_once()
                            receiving.close.assert_called_once()
                        else:
                            with self.assertRaises(PermissionError):
                                await evaluation

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
            r
            async for r in executor.evaluate(
                [("policy", source, seed) for seed in seeds], environment or ProcessEnv()
            )
        ]

    async def test_fresh_processes_trajectory_and_large_result(self):
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
            self.assertEqual(len(set(pids)), 4)
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
                    for policy_seed, instructions in ((5, "override"), (None, "")):
                        results = [
                            episode
                            async for _, _, episode in executor.evaluate(
                                [(policy.id, policy.source, 3)],
                                environment,
                                policy_seed=policy_seed,
                                instructions=instructions,
                                max_steps=1,
                            )
                        ]
                        (episode,) = results
                        self.assertEqual(
                            json.loads(episode.artifacts["state.json"])[1:],
                            [1, policy_seed, instructions],
                        )
                        self.assertEqual(episode.infos[0]["seed"], 3)
                        self.assertEqual(episode.truncations, [True])

    async def test_errors_and_successful_siblings(self):
        async with Executor(concurrency=2, episode_timeout=3) as executor:
            for source, error in [
                (SOURCE.replace("return 0", "return 99"), PolicyError),
                ("raise ValueError('load failed')", PolicyError),
                (SOURCE.replace("return 0", "raise RuntimeError('act failed')"), PolicyError),
                ("import os; os._exit(17)", InfrastructureError),
            ]:
                with self.subTest(source=source):
                    if error is InfrastructureError:
                        with self.assertRaises(error):
                            await self.collect(executor, source)
                    else:
                        results = await self.collect(executor, source)
                        self.assertIsNotNone(results[0][2].error)
            good = []
            async for result in executor.evaluate(
                [
                    ("bad", SOURCE.replace("return 0", "while True: pass"), 0),
                    ("good", SOURCE, 1),
                ],
                ProcessEnv(),
            ):
                good.append(result)
            self.assertEqual(good[0][0], "good")
            self.assertIn("exceeded", next(ep.error for id, _, ep in good if id == "bad"))
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

    async def test_cancel_and_timeout_kill_ordinary_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pids"
            source = SOURCE.replace(
                "return 0",
                f"""import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path({str(marker)!r}).write_text(str(os.getpid()) + " " + str(child.pid))
        while True: pass""",
            )
            for cancel in (False, True):
                marker.unlink(missing_ok=True)
                async with Executor(episode_timeout=3) as executor:
                    task = asyncio.create_task(self.collect(executor, source))

                    async def wait_for_marker():
                        while not marker.exists():
                            await asyncio.sleep(0.01)

                    await asyncio.wait_for(wait_for_marker(), 10)
                    pids = [int(p) for p in marker.read_text().split()]
                    if cancel:
                        task.cancel()
                    if cancel:
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        results = await task
                        self.assertIn("exceeded", results[0][2].error)
                    for pid in pids:
                        # Linux init may need a moment to reap an adopted grandchild.
                        for _ in range(100):
                            try:
                                os.kill(pid, 0)
                            except ProcessLookupError:
                                break
                            await asyncio.sleep(0.02)
                        else:
                            self.fail(f"process {pid} survived")

    def test_options_validated(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Executor(episode_timeout=value)


if __name__ == "__main__":
    unittest.main()
