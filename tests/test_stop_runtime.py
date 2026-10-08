"""Offline Docker/RPC checks; opt in with RSIKIT_TEST_IMAGE after building it."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class STOPExecutionCountTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_counts_failed_episode_retries_and_excludes_cached_success(self):
        from research.stop_optimizer import _evaluate
        from rsikit import Episode, Run
        from tests.helpers import fake_executor
        from tests.search_helpers import policy
        from tests.test_episode_storage import trajectory
        from tests.test_execution import ProcessEnv
        from tests.test_run import FakeEvaluation

        candidate = policy(1)
        backend = FakeEvaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            with Run.create(name="retry", path=path) as run:
                run.save_episode(candidate, 0, trajectory(3))
                run.save_episode(candidate, 1, Episode(error="previous crash"))
            request = dict(
                source=candidate.to_text(),
                environment="CartPole-v1",
                seeds=[0, 1, 2, 1],
                max_steps=2,
            )
            with (
                patch.object(_evaluate, "Path", return_value=path),
                patch.object(_evaluate, "make_environment", return_value=ProcessEnv()),
                patch.object(_evaluate, "Executor", return_value=fake_executor(evaluation=backend)),
            ):
                report = await _evaluate.evaluate(request)
                self.assertEqual(report["executed_episodes"], 2)
                self.assertEqual(report["executed_episodes"], backend.evaluate.await_count)
                self.assertTrue(report["accepted"])
                report = await _evaluate.evaluate(request)
                self.assertEqual(report["executed_episodes"], 0)
                self.assertEqual(backend.evaluate.await_count, 2)


@unittest.skipUnless(os.environ.get("RSIKIT_TEST_IMAGE"), "requires built Docker test image")
class STOPRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_improver_uses_host_capabilities_and_policy_run(self):
        from research.stop_optimizer import Capabilities
        from research.stop_optimizer.runtime import DockerRuntime
        from tests.search_helpers import policy

        runtime = DockerRuntime(image=os.environ["RSIKIT_TEST_IMAGE"])
        with tempfile.TemporaryDirectory() as directory:
            source = policy(1).to_text()

            async def suggest(initial, guidance):
                return source

            async def evaluate(candidate):
                report = await runtime.evaluate_policy(
                    candidate,
                    path=Path(directory) / "task",
                    environment="CartPole-v1",
                    seeds=[0],
                    max_steps=2,
                )
                return report["scores"]["0"]

            code = """async def improve(initial, capabilities):
    candidate = await capabilities.suggest(initial, "change")
    score = await capabilities.evaluate(candidate)
    assert score == 2
    return candidate
"""
            caps = Capabilities(suggest, evaluate, 1, 1)
            self.assertEqual(await runtime.execute(code, source, caps), source)
            self.assertEqual((caps.generations_left, caps.evaluations_left), (0, 0))
            self.assertTrue((Path(directory) / "task/run/run.sqlite").exists())

    async def test_budget_timeout_bad_rpc_and_infrastructure_errors(self):
        from research.stop_optimizer import Capabilities, ImproverExecutionError
        from research.stop_optimizer.runtime import DockerRuntime

        runtime = DockerRuntime(image=os.environ["RSIKIT_TEST_IMAGE"])

        async def forbidden(*args):
            self.fail("zero budget must not reach host callback")

        caps = Capabilities(forbidden, forbidden, 0, 0)
        for code in (
            "async def improve(initial, capabilities): return await capabilities.suggest(initial)",
            'import inspect\ninspect.currentframe().f_back.f_locals["protocol"].write("{}\\n")\nasync def improve(initial, capabilities): return initial',
            'async def improve(initial, capabilities): return "x" * 1100000',
        ):
            with self.assertRaises(ImproverExecutionError):
                await runtime.execute(code, "initial", caps)
        runtime.timeout = 0.2
        with self.assertRaisesRegex(ImproverExecutionError, "timed out"):
            await runtime.execute("while True: pass", "initial", caps)
        runtime.timeout = 10

        async def broken(*args):
            raise RuntimeError("provider offline")

        with self.assertRaisesRegex(RuntimeError, "provider offline"):
            await runtime.execute(
                "async def improve(initial, capabilities): return await capabilities.suggest(initial)",
                "initial",
                Capabilities(broken, forbidden, 1, 0),
            )
        task = asyncio.create_task(runtime.execute("while True: pass", "initial", caps))
        await asyncio.sleep(0.3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_real_self_improvement_round(self):
        from research import stop_optimizer
        from research.stop_optimizer.runtime import DockerRuntime
        from tests.providers import ScriptedProvider
        from tests.search_helpers import templates

        runtime = DockerRuntime(image=os.environ["RSIKIT_TEST_IMAGE"])

        async def evaluate(source):
            return 0.75 if source == "solution" else 0.25

        improver = 'async def improve(initial, capabilities): return "solution"'
        provider = ScriptedProvider([json.dumps(dict(code=improver))])
        with templates(stop_optimizer):
            agent = stop_optimizer.STOP(
                "task",
                provider,
                runtime.execute,
                [stop_optimizer.Problem("initial solution", "return solution", evaluate)],
            )
            result = await agent.run(
                rounds=1, generations=1, evaluations=1, inner_generations=1, inner_evaluations=1
            )
        self.assertEqual(result["improver"], improver)
        self.assertEqual(result["history"][0]["checked_score"], 0.75)

    async def test_optimizer_interface_consumes_real_policy_episodes(self):
        from research.stop_optimizer import STOP, Problem
        from research.stop_optimizer.runtime import DockerRuntime
        from rsikit import Run
        from tests.providers import ScriptedProvider
        from tests.search_helpers import policy

        runtime = DockerRuntime(image=os.environ["RSIKIT_TEST_IMAGE"])

        async def utility(source):
            return 0.5

        agent = STOP(
            "task",
            ScriptedProvider([]),
            runtime.execute,
            [Problem(policy(1).to_text(), "reward", utility)],
            rounds=0,
            initial="async def improve(initial, capabilities): return initial",
        )
        with tempfile.TemporaryDirectory() as directory:
            (proposed,) = await agent.propose()
            path = Path(directory) / "task"
            await runtime.evaluate_policy(
                proposed.to_text(), path=path, environment="CartPole-v1", seeds=[0], max_steps=2
            )
            with Run.open(path / "run") as run:
                agent.update({proposed.id: {0: run.load_episode(proposed, 0)}})
            self.assertEqual(agent.best.id, proposed.id)
            self.assertEqual(agent._best_score, 2)
            await agent.aclose()
