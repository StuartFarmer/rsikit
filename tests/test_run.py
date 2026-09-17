"""Durable jobs survive interruptions without regenerating or importing policies."""

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from slick import prompts
from sqlmodel import Session, select

import rsikit.generation as generation
from rsikit import Execution, Policy, Run, generate
from rsikit.episode import InfrastructureError, PolicyError
from tests.providers import ScriptedProvider

RESPONSE = {
    "name": "Model chose this name",
    "summary": "A generated controller",
    "implementation": """from rsikit import Controller
class Solution(Controller):
    async def act(self, observation):
        return 0
""",
}
RESULT = (0, 1.0, True, False, {"episode": {"r": 7.0, "l": 7, "t": 0.1}})


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

    async def test_generated_object_exports_storage_and_reuse(self):
        self.assertIsInstance(self.policy, Policy)
        self.assertEqual(self.policy.name, RESPONSE["name"])
        self.assertFalse(hasattr(self.policy, "source"))
        self.assertNotIn("implementation", self.policy.model_dump())
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.run.run_policy", runner):
            with Run.create(name="demo", environment="CartPole-v1", path=self.path) as run:
                results = await run.evaluate(self.policy, seeds=iter([0, 1, 0]))
                self.assertEqual(len(results), 2)
                self.assertEqual([r.status for r in results], ["completed", "completed"])
                await run.evaluate(self.policy, seeds=[0, 1])
                self.assertEqual(runner.await_count, 2)
                exports = list((self.path / "exports").glob("*.py"))
                self.assertEqual(len(exports), 1)
                self.assertEqual(exports[0].read_text(), RESPONSE["implementation"])
                self.assertIsInstance(runner.call_args.args[0], Policy)
            exports[0].unlink()
            with Run.open(self.path) as reopened:
                self.assertEqual(reopened.name, "demo")
                self.assertEqual(reopened.policies()[0].id, self.policy.id)
                self.assertEqual(len(await reopened.resume()), 2)
                self.assertEqual(runner.await_count, 2)
                self.assertTrue(exports[0].exists())
        self.assertEqual(len(self.provider.calls), 1)

    async def test_interrupted_episode_restarts_and_completed_episode_is_retained(self):
        started = asyncio.Event()

        async def blocked(policy, make_env, *, env_seed, **kwargs):
            if env_seed == 1:
                started.set()
                await asyncio.Event().wait()
            return RESULT

        with Run.create(name="resume", environment="CartPole-v1", path=self.path) as run:
            with patch("rsikit.run.run_policy", blocked):
                task = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(
                [r.status for r in run.executions()], ["completed", "pending", "pending"]
            )
            # Simulate a process dying after it commits 'running'.
            with Session(run._engine) as session:
                job = session.exec(select(Execution).where(Execution.env_seed == 1)).one()
                job.status = "running"
                session.add(job)
                session.commit()
        moved = self.path.with_name("moved")
        shutil.move(self.path, moved)
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.run.run_policy", runner), Run.open(moved) as run:
            result = await run.resume()
            self.assertEqual([r.status for r in result], ["completed"] * 3)
            self.assertEqual([call.kwargs["env_seed"] for call in runner.call_args_list], [1, 2])
            self.assertEqual([r.attempts for r in result], [1, 2, 1])
            self.assertEqual(result[0].reward, 7)

    async def test_failed_policies_are_terminal_infrastructure_errors_are_resumable(self):
        with Run.create(name="failures", environment="CartPole-v1", path=self.path) as run:
            with patch(
                "rsikit.run.run_policy",
                AsyncMock(
                    side_effect=[
                        PolicyError("bad action"),
                        InfrastructureError("worker unavailable"),
                    ]
                ),
            ):
                with self.assertRaisesRegex(InfrastructureError, "worker unavailable"):
                    await run.evaluate(self.policy, seeds=[0, 1, 2])
            self.assertEqual([r.status for r in run.executions()], ["failed", "pending", "pending"])
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.run.run_policy", runner), Run.open(self.path) as run:
            result = await run.resume()
            self.assertEqual([r.status for r in result], ["failed", "completed", "completed"])
            self.assertEqual(runner.await_count, 2)
            self.assertIsNone(result[0].reward)
            self.assertIn("bad action", result[0].error)

    async def test_run_ownership_export_opt_out_and_policy_identity(self):
        with Run.create(
            name="private", environment="CartPole-v1", path=self.path, export=False
        ) as run:
            with self.assertRaises(BlockingIOError):
                Run.open(self.path)
            with patch("rsikit.run.run_policy", AsyncMock(return_value=RESULT)):
                await run.evaluate(self.policy, seeds=[0])
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    await run.evaluate(
                        self.policy.model_copy(update={"name": "changed"}), seeds=[0]
                    )
            self.assertFalse((self.path / "exports").exists())
        with Run.open(self.path) as run:
            self.assertFalse((self.path / "exports").exists())
            self.assertEqual(run.policies()[0].name, self.policy.name)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await run.resume()
        with self.assertRaises(FileNotFoundError):
            Run.open(self.path / "missing")

    async def test_untrusted_generated_name_cannot_choose_export_path(self):
        provider = ScriptedProvider([json.dumps({**RESPONSE, "name": "../../outside"})])
        policy = await generate("task", provider=provider)
        with Run.create(name="paths", environment="CartPole-v1", path=self.path) as run:
            with patch("rsikit.run.run_policy", AsyncMock(return_value=RESULT)):
                await run.evaluate(policy, seeds=[0])
            files = list((self.path / "exports").glob("*.py"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].parent, self.path / "exports")
            self.assertEqual(run.policies()[0].name, "../../outside")


if __name__ == "__main__":
    unittest.main()
