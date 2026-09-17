"""Generated policies act; storage keeps scores and resumes unfinished work."""

import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from gymnasium.spaces import Discrete
from slick import prompts

import rsikit.generation as generation
from rsikit import Policy, Run, generate
from rsikit.episode import InfrastructureError, PolicyError
from tests.providers import ScriptedProvider

RESPONSE = {
    "name": "Model chose this name",
    "implementation": "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n",
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

    async def test_policy_class_scores_exports_and_reuse(self):
        self.assertTrue(issubclass(self.policy, Policy))
        self.assertEqual(self.policy.name, RESPONSE["name"])
        agent = self.policy(Discrete(2), Discrete(2))
        self.assertIsInstance(agent, Policy)
        self.assertEqual(agent.name, RESPONSE["name"])
        self.assertFalse(hasattr(agent, "source"))
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.run.run_episode", runner):
            with Run.create(name="demo", environment="CartPole-v1", path=self.path) as run:
                self.assertEqual(
                    await run.evaluate(self.policy, seeds=iter([0, 1, 0])), {0: 7.0, 1: 7.0}
                )
                await run.evaluate(self.policy, seeds=[0, 1])
                self.assertEqual(runner.await_count, 2)
                self.assertIs(runner.call_args.args[1].func, self.policy)
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

        async def blocked(*args, env_seed, **kwargs):
            if env_seed == 1:
                started.set()
                await asyncio.Event().wait()
            return RESULT

        with Run.create(name="resume", environment="CartPole-v1", path=self.path) as run:
            with patch("rsikit.run.run_episode", blocked):
                task = asyncio.create_task(run.evaluate(self.policy, seeds=[0, 1, 2]))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: None, 2: None})
        moved = self.path.with_name("moved")
        shutil.move(self.path, moved)
        runner = AsyncMock(return_value=RESULT)
        with patch("rsikit.run.run_episode", runner), Run.open(moved) as run:
            await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 7.0, 2: 7.0})
            self.assertEqual([call.kwargs["env_seed"] for call in runner.call_args_list], [1, 2])

    async def test_errors_propagate_without_becoming_scores(self):
        with Run.create(name="errors", environment="CartPole-v1", path=self.path) as run:
            for error in (PolicyError("bad action"), InfrastructureError("worker unavailable")):
                with patch("rsikit.run.run_episode", AsyncMock(side_effect=error)):
                    with self.assertRaises(type(error)):
                        await run.evaluate(self.policy, seeds=[0, 1])
                self.assertEqual(run.scores(self.policy), {0: None, 1: None})
            with patch("rsikit.run.run_episode", AsyncMock(return_value=RESULT)):
                await run.resume()
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 7.0})

    async def test_run_ownership_export_opt_out_and_identity(self):
        with Run.create(
            name="private", environment="CartPole-v1", path=self.path, export=False
        ) as run:
            with self.assertRaises(BlockingIOError):
                Run.open(self.path)
            with patch("rsikit.run.run_episode", AsyncMock(return_value=RESULT)):
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
            with patch("rsikit.run.run_episode", AsyncMock(return_value=RESULT)):
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
