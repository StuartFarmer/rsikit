"""Check that Rich output is emitted before a generation or evaluation batch ends."""

import asyncio
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from rich.progress import Progress
from slick import prompts

import examples.alphaevolve as example
from research import alphaevolve
from research.alphaevolve.generation import _PolicyResponse
from research.alphaevolve.improved import AlphaEvolve, Config
from rsikit import Executor
from rsikit.evaluation import PolicyError
from rsikit.progress import ProgressHandler
from tests.helpers import recorded_run
from tests.providers import ScriptedProvider
from tests.test_episode_storage import trajectory
from tests.test_run import RESPONSE, FakeSandbox


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    def test_overlapping_evaluation_submissions_accumulate_progress(self):
        progress = Progress(console=Console(file=io.StringIO()))
        handler = ProgressHandler(progress, overlap=True)
        self.addCleanup(handler.close)
        for total in (3, 4):
            record = logging.makeLogRecord(
                dict(
                    msg="evaluating",
                    levelno=logging.INFO,
                    levelname="INFO",
                    event="evaluation_started",
                    total=total,
                )
            )
            handler.emit(record)
        handler.emit(
            logging.makeLogRecord(
                dict(msg="done", levelno=logging.INFO, levelname="INFO", event="policy_evaluated")
            )
        )
        task = progress.tasks[handler.evaluation]
        self.assertEqual((task.total, task.completed), (7, 1))

    async def test_live_policy_summaries_scores_and_failure_log(self):
        output = io.StringIO()
        console = Console(file=output, width=160, force_terminal=False)
        provider = ScriptedProvider(
            [
                _PolicyResponse(
                    name="First [bold]",
                    description="Push left as a baseline.",
                    implementation=RESPONSE["implementation"],
                ),
                _PolicyResponse(
                    name="Second",
                    description="Try a different strategy.",
                    implementation=RESPONSE["implementation"] + "\n# second\n",
                ),
            ]
        )
        acall = provider.acall
        calls = 0
        generation_started, generation_release = asyncio.Event(), asyncio.Event()

        async def generate(*args, **kwargs):
            nonlocal calls
            calls += 1
            response = await acall(*args, **kwargs)
            if calls == 2:
                generation_started.set()
                await generation_release.wait()
            return response

        provider.acall = generate
        sandbox = FakeSandbox()
        second_started, release = asyncio.Event(), asyncio.Event()

        async def evaluate(implementation, environment, seed, call_timeout):
            if "# second" in implementation:
                second_started.set()
                await release.wait()
                raise PolicyError("bad action")
            return trajectory(7.0, {})

        sandbox.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=3) as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            recorded_run(
                name="progress",
                path=Path(directory) / "run",
                environment=env,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as (run, rollouts),
        ):
            agent = AlphaEvolve("task", provider, config=Config(max_repairs=0))
            task = asyncio.create_task(
                example.run_search(
                    agent, run, rollouts, generations=1, batch_size=2, console=console
                )
            )
            try:
                await asyncio.wait_for(generation_started.wait(), 2)
                for _ in range(100):
                    if "Push left as a baseline." in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("First [bold]", output.getvalue())
                self.assertIn("Push left as a baseline.", output.getvalue())
                self.assertFalse(task.done())
                generation_release.set()
                await asyncio.wait_for(second_started.wait(), 2)
                for _ in range(100):
                    if "score=7" in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("score=7", output.getvalue())
                self.assertFalse(task.done())
            finally:
                generation_release.set()
                release.set()
                await task
            text = output.getvalue()
            self.assertIn("Second", text)
            self.assertIn("bad action", text)
            saved = (run.path / "run.log").read_text()
            self.assertIn("Push left as a baseline.", saved)
            self.assertIn("score=7", saved)
            self.assertIn("bad action", saved)
            self.assertEqual(run.policies()[0].description, "Push left as a baseline.")
            self.assertIn("Discarded Second", saved)
            self.assertEqual(agent.completed, 1)
            self.assertEqual(agent._pending, {})


if __name__ == "__main__":
    unittest.main()
