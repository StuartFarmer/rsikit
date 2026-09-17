"""Generated policies act; storage keeps scores and resumes unfinished work."""

import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from slick import prompts

import rsikit.generation as generation
from rsikit import Policy, Run, generate
from rsikit.episode import InfrastructureError, PolicyError
from rsikit.sandbox import SandboxPolicy
from tests.providers import ScriptedProvider

RESPONSE = {
    "name": "Model chose this name",
    "implementation": "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n",
}
RESULT = 7.0


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run"
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts")
        root.start()
        self.addCleanup(root.stop)
        self.provider = ScriptedProvider([json.dumps(RESPONSE)])
        self.policy = await generate("test task", provider=self.provider)

    async def test_policy_class_scores_exports_and_reuse(self):
        self.assertTrue(issubclass(self.policy, Policy))
        self.assertEqual(self.policy.name, RESPONSE["name"])
        self.assertFalse(issubclass(self.policy, SandboxPolicy))
        self.assertFalse(hasattr(self.policy, "source"))
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.execution.DockerExecutor.evaluate", runner):
            with Run.create(name="demo", environment="CartPole-v1", path=self.path) as run:
                self.assertEqual(
                    await run.evaluate(self.policy, seeds=iter([0, 1, 0])),
                    {self.policy.id: {0: 7.0, 1: 7.0}},
                )
                await run.evaluate(self.policy, seeds=[0, 1])
                self.assertEqual(runner.await_count, 2)
                self.assertEqual(runner.call_args.args[0], RESPONSE["implementation"])
                self.assertEqual(runner.call_args.kwargs["environment"], "CartPole-v1")
                (export,) = (self.path / "exports").glob("*.py")
                self.assertEqual(export.read_text(), RESPONSE["implementation"])
            export.unlink()
            with Run.open(self.path) as run:
                (restored,) = run.policies()
                self.assertTrue(issubclass(restored, Policy))
                self.assertEqual(restored.id, self.policy.id)
                await run.resume()
                self.assertEqual(run.scores(restored), {0: 7.0, 1: 7.0})
                self.assertEqual(runner.await_count, 2)
                self.assertTrue(export.exists())
        with closing(sqlite3.connect(self.path / "run.sqlite")) as db:
            self.assertEqual(
                {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                {"settings", "policy"},
            )
            self.assertEqual(
                [row[1] for row in db.execute("PRAGMA table_info(policy)")],
                ["id", "name", "implementation", "scores"],
            )
        self.assertEqual(len(self.provider.calls), 1)

    async def test_interruption_preserves_scores_and_resumes_after_move(self):
        started = asyncio.Event()

        async def blocked(*args, seed, **kwargs):
            if seed == 1:
                started.set()
                await asyncio.Event().wait()
            return RESULT

        with Run.create(name="resume", environment="CartPole-v1", path=self.path) as run:
            with patch("rsikit.execution.DockerExecutor.evaluate", blocked):
                task = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: None, 2: None})
        moved = self.path.with_name("moved")
        shutil.move(self.path, moved)
        runner = AsyncMock(return_value=RESULT)
        with Run.open(moved, executor=SimpleNamespace(evaluate=runner)) as run:
            await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 7.0, 2: 7.0})
            self.assertEqual([call.kwargs["seed"] for call in runner.call_args_list], [1, 2])

    async def test_errors_propagate_without_becoming_scores(self):
        with Run.create(name="errors", environment="CartPole-v1", path=self.path) as run:
            for error in (PolicyError("bad action"), InfrastructureError("worker unavailable")):
                with patch(
                    "rsikit.execution.DockerExecutor.evaluate", AsyncMock(side_effect=error)
                ):
                    with self.assertRaises(type(error)):
                        await run.evaluate(self.policy, seeds=[0, 1])
                self.assertEqual(run.scores(self.policy), {0: None, 1: None})
            with patch(
                "rsikit.execution.DockerExecutor.evaluate", AsyncMock(return_value=float("nan"))
            ):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    await run.resume()
            self.assertEqual(run.scores(self.policy), {0: None, 1: None})
            with patch("rsikit.execution.DockerExecutor.evaluate", AsyncMock(return_value=RESULT)):
                await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 7.0})

    async def test_run_ownership_export_opt_out_and_identity(self):
        with Run.create(
            name="private", environment="CartPole-v1", path=self.path, export=False
        ) as run:
            with self.assertRaises(BlockingIOError):
                Run.open(self.path)
            with patch("rsikit.execution.DockerExecutor.evaluate", AsyncMock(return_value=RESULT)):
                await run.evaluate(self.policy)
                with patch.object(self.policy, "name", "changed"):
                    with self.assertRaisesRegex(ValueError, "cannot change"):
                        await run.evaluate(self.policy)
            self.assertFalse((self.path / "exports").exists())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await run.resume()
        with self.assertRaises(FileNotFoundError):
            Run.open(self.path / "missing")

    async def test_generated_code_is_not_executed_on_host_and_name_is_safe(self):
        response = {
            **RESPONSE,
            "name": "../../outside",
            "implementation": "raise AssertionError('host execution')\n"
            + RESPONSE["implementation"],
        }
        policy = await generate("task", provider=ScriptedProvider([json.dumps(response)]))
        with Run.create(name="paths", environment="CartPole-v1", path=self.path) as run:
            with patch("rsikit.execution.DockerExecutor.evaluate", AsyncMock(return_value=RESULT)):
                await run.evaluate(policy)
            (export,) = (self.path / "exports").glob("*.py")
            self.assertEqual(export.parent, self.path / "exports")
        with self.assertRaisesRegex(ValueError, "Solution"):
            await generate(
                "task",
                provider=ScriptedProvider([json.dumps({**RESPONSE, "implementation": "pass"})]),
            )

    async def test_run_bounds_group_execution_and_preserves_scores(self):
        other = await generate(
            "task",
            provider=ScriptedProvider(
                [
                    json.dumps(
                        {
                            "name": "Other",
                            "implementation": RESPONSE["implementation"].replace(
                                "return 0", "return 1"
                            ),
                        }
                    )
                ]
            ),
        )
        both_started, release = asyncio.Event(), asyncio.Event()
        active = peak = calls = 0

        async def evaluate(implementation, **options):
            nonlocal active, peak, calls
            # The boundary needs no dynamic Python classes or database session.
            json.dumps({"implementation": implementation, **options})
            self.assertEqual(len(run.policies()), 2)
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            try:
                await release.wait()
                await asyncio.sleep(0)
                return options["seed"] + (10 if "return 1" in implementation else 0)
            finally:
                active -= 1

        executor = SimpleNamespace(evaluate=evaluate)
        with Run.create(
            name="batch",
            environment="CartPole-v1",
            path=self.path,
            executor=executor,
            concurrency=2,
        ) as run:
            self.assertIs(run.executor, executor)
            pending = asyncio.create_task(
                run.evaluate(self.policy, other, self.policy, seeds=[1, 2, 3])
            )
            await asyncio.wait_for(both_started.wait(), 2)
            self.assertEqual(calls, 2)
            release.set()
            self.assertEqual(
                await pending,
                {
                    self.policy.id: {1: 1.0, 2: 2.0, 3: 3.0},
                    other.id: {1: 11.0, 2: 12.0, 3: 13.0},
                },
            )
            self.assertEqual(peak, 2)
            self.assertEqual(calls, 6)
            await run.evaluate(other, self.policy, seeds=[1, 2, 3])
            self.assertEqual(calls, 6)

    async def test_group_failure_preserves_other_scores_and_resume_uses_new_executor(self):
        async def evaluate(implementation, *, seed, **options):
            if seed == 1:
                raise PolicyError("bad action")
            return seed

        with Run.create(
            name="partial",
            environment="CartPole-v1",
            path=self.path,
            executor=SimpleNamespace(evaluate=evaluate),
            concurrency=2,
        ) as run:
            with self.assertRaisesRegex(PolicyError, "bad action"):
                await run.evaluate(self.policy, seeds=[0, 1, 2])
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: None, 2: 2.0})
        replacement = SimpleNamespace(evaluate=AsyncMock(return_value=9))
        with Run.open(self.path, executor=replacement) as run:
            await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: 9.0, 2: 2.0})
            self.assertEqual(replacement.evaluate.await_count, 1)
            self.assertEqual(replacement.evaluate.call_args.kwargs["seed"], 1)

    async def test_cancellation_waits_for_all_active_executor_calls(self):
        both_started = asyncio.Event()
        active = 0

        async def evaluate(*args, **kwargs):
            nonlocal active
            active += 1
            if active == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                active -= 1

        with Run.create(
            name="cancel",
            environment="CartPole-v1",
            path=self.path,
            executor=SimpleNamespace(evaluate=evaluate),
            concurrency=2,
        ) as run:
            pending = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
            await asyncio.wait_for(both_started.wait(), 2)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual(active, 0)
            self.assertEqual(run.scores(self.policy), {0: None, 1: None, 2: None})


if __name__ == "__main__":
    unittest.main()
