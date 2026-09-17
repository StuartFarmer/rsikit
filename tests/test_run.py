"""Run persistence and executor scheduling without a Docker dependency."""

import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import gymnasium as gym
from slick import prompts

import rsikit.generation as generation
from rsikit import Executor, Policy, Run, generate
from rsikit.episode import InfrastructureError, PolicyError
from tests.providers import ScriptedProvider

RESPONSE = {
    "name": "Model chose this name",
    "implementation": "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n",
}


class FakeSandbox:
    def __init__(self):
        self.start = AsyncMock()
        self.evaluate = AsyncMock(return_value=(7.0, {}))
        self.close = AsyncMock()


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run"
        self.env = gym.make("CartPole-v1", max_episode_steps=3)
        self.addCleanup(self.env.close)
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts")
        root.start()
        self.addCleanup(root.stop)
        self.provider = ScriptedProvider([json.dumps(RESPONSE)])
        self.policy = await generate("test task", provider=self.provider)
        self.sandbox = FakeSandbox()
        self.executor = Executor(sandbox=self.sandbox, call_timeout=4)

    def create(self, **kwargs):
        return Run.create(
            name="test", environment=self.env, path=self.path, executor=self.executor, **kwargs
        )

    def reopen(self, path=None):
        return Run.open(path or self.path, environment=self.env, executor=self.executor)

    async def test_policy_scores_exports_and_reuse(self):
        self.assertTrue(issubclass(self.policy, Policy))
        self.assertEqual(self.policy.name, RESPONSE["name"])
        with self.create() as run:
            self.assertIs(run.environment, self.env)
            self.assertIs(run.executor, self.executor)
            self.assertEqual(
                await run.evaluate(self.policy, seeds=iter([0, 1, 0])),
                {self.policy.id: {0: 7.0, 1: 7.0}},
            )
            await run.evaluate(self.policy, seeds=[0, 1])
            self.assertEqual(self.sandbox.evaluate.await_count, 2)
            self.sandbox.start.assert_awaited_once_with(1)
            self.assertEqual(self.sandbox.evaluate.call_args.args[3], 4)
            (export,) = (self.path / "exports").glob("*.py")
            self.assertEqual(export.read_text(), RESPONSE["implementation"])
        export.unlink()
        with self.reopen() as run:
            (restored,) = run.policies()
            await run.resume()
            self.assertEqual(restored.id, self.policy.id)
            self.assertEqual(run.scores(restored), {0: 7.0, 1: 7.0})
            self.assertTrue(export.exists())
        self.assertEqual(self.sandbox.start.await_count, 1)
        with closing(sqlite3.connect(self.path / "run.sqlite")) as db:
            self.assertEqual(
                [r[1] for r in db.execute("PRAGMA table_info(settings)")], ["name", "export"]
            )
            self.assertEqual(
                [r[1] for r in db.execute("PRAGMA table_info(policy)")],
                ["id", "name", "implementation", "scores"],
            )
        self.assertEqual(len(self.provider.calls), 1)

    async def test_interruption_and_resume_after_move(self):
        started = asyncio.Event()

        async def blocked(implementation, environment, seed, call_timeout):
            if seed == 1:
                started.set()
                await asyncio.Event().wait()
            return 7.0, {}

        self.sandbox.evaluate.side_effect = blocked
        with self.create() as run:
            task = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
            await started.wait()
            # Allow the completed seed's result to reach the persistence consumer.
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: None, 2: None})
        self.sandbox.close.assert_awaited_once()
        moved = self.path.with_name("moved")
        shutil.move(self.path, moved)
        self.sandbox.evaluate = AsyncMock(return_value=(9.0, {}))
        with self.reopen(moved) as run:
            await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 9.0, 2: 9.0})
            self.assertEqual([c.args[2] for c in self.sandbox.evaluate.call_args_list], [1, 2])

    async def test_group_failure_preserves_other_scores_and_artifacts(self):
        async def evaluate(implementation, environment, seed, call_timeout):
            if seed == 1:
                raise PolicyError("bad action")
            return seed, {"nested/result.txt": str(seed).encode()}

        self.sandbox.evaluate.side_effect = evaluate
        self.executor = Executor(sandbox=self.sandbox, concurrency=2)
        with self.create() as run:
            with self.assertRaisesRegex(PolicyError, "bad action"):
                await run.evaluate(self.policy, seeds=[0, 1, 2])
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: None, 2: 2.0})
            self.assertEqual(
                (run.path / "artifacts" / self.policy.id / "2/nested/result.txt").read_bytes(), b"2"
            )
        self.sandbox.evaluate = AsyncMock(return_value=(3.0, {}))
        with self.reopen() as run:
            await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: 3.0, 2: 2.0})
            self.assertEqual(self.sandbox.evaluate.await_count, 1)

    async def test_executor_bounds_workers_and_closes_once(self):
        other = await generate(
            "task", provider=ScriptedProvider([json.dumps({**RESPONSE, "name": "Other"})])
        )
        started, release = asyncio.Event(), asyncio.Event()
        active = peak = calls = 0

        async def evaluate(*args):
            nonlocal active, peak, calls
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
            try:
                await release.wait()
                return 7.0, {}
            finally:
                active -= 1

        self.sandbox.evaluate.side_effect = evaluate
        self.executor = Executor(sandbox=self.sandbox, concurrency=2)
        with self.create() as run:
            task = asyncio.create_task(run.evaluate(self.policy, other, seeds=[0, 1]))
            await asyncio.wait_for(started.wait(), 2)
            self.assertEqual(calls, 2)
            release.set()
            result = await task
            self.assertEqual(result, {self.policy.id: {0: 7.0, 1: 7.0}, other.id: {0: 7.0, 1: 7.0}})
        self.assertEqual(peak, 2)
        self.sandbox.start.assert_awaited_once_with(2)
        self.sandbox.close.assert_awaited_once()

    async def test_cancellation_waits_for_workers_before_closing_sandbox(self):
        started = asyncio.Event()
        active = 0

        async def evaluate(*args):
            nonlocal active
            active += 1
            if active == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                active -= 1

        async def close():
            self.assertEqual(active, 0)

        self.sandbox.evaluate.side_effect = evaluate
        self.sandbox.close.side_effect = close
        self.executor = Executor(sandbox=self.sandbox, concurrency=2)
        with self.create() as run:
            task = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(run.scores(self.policy), {0: None, 1: None, 2: None})
        self.sandbox.close.assert_awaited_once()

    async def test_invalid_results_artifact_paths_and_startup_failure_remain_pending(self):
        with self.create() as run:
            for outcome in [(float("nan"), {}), (1.0, {"../../../../outside": b"bad"})]:
                self.sandbox.evaluate.return_value = outcome
                with self.assertRaises(ValueError):
                    await run.evaluate(self.policy)
                self.assertEqual(run.scores(self.policy), {0: None})
            self.sandbox.start.side_effect = InfrastructureError("start failed")
            with self.assertRaisesRegex(InfrastructureError, "start failed"):
                await run.resume()
            self.assertEqual(self.sandbox.close.await_count, 3)

            entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def start(workers):
                entered.set()
                await release.wait()
                finished.set()

            async def close():
                self.assertTrue(finished.is_set())

            self.sandbox.start.side_effect = start
            self.sandbox.close.side_effect = close
            pending = asyncio.create_task(run.resume())
            await entered.wait()
            pending.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertFalse((self.path.parent / "outside").exists())

    async def test_run_ownership_identity_and_export_opt_out(self):
        with self.create(export=False) as run:
            with self.assertRaises(BlockingIOError):
                self.reopen()
            await run.evaluate(self.policy)
            with patch.object(self.policy, "name", "changed"):
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    await run.evaluate(self.policy)
            self.assertFalse((self.path / "exports").exists())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await run.resume()

    async def test_generation_never_executes_code_and_export_name_is_safe(self):
        response = {
            **RESPONSE,
            "name": "../../outside",
            "implementation": "raise AssertionError('host execution')\n"
            + RESPONSE["implementation"],
        }
        policy = await generate("task", provider=ScriptedProvider([json.dumps(response)]))
        with self.create() as run:
            await run.evaluate(policy)
            (export,) = (self.path / "exports").glob("*.py")
            self.assertEqual(export.parent, self.path / "exports")
        with self.assertRaisesRegex(ValueError, "Solution"):
            await generate(
                "task",
                provider=ScriptedProvider([json.dumps({**RESPONSE, "implementation": "pass"})]),
            )


if __name__ == "__main__":
    unittest.main()
