"""Check that Rich output is emitted before a generation or evaluation batch ends."""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts

import examples.alphaevolve as example
import rsikit.alphaevolve as alphaevolve
from rsikit import Executor, Run
from rsikit.alphaevolve import AlphaEvolve, Config
from rsikit.alphaevolve.edits import Program
from rsikit.episode import PolicyError
from tests.providers import ScriptedProvider
from tests.test_run import RESPONSE, FakeSandbox


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_policy_summaries_scores_and_failure_log(self):
        output = io.StringIO()
        console = Console(file=output, width=160, force_terminal=False)
        provider = ScriptedProvider(
            [
                Program(
                    name="First [bold]",
                    description="Push left as a baseline.",
                    implementation=RESPONSE["implementation"],
                ),
                Program(
                    name="Second",
                    description="Try a different strategy.",
                    implementation=RESPONSE["implementation"] + "\n# second\n",
                ),
            ]
        )
        acall = provider.acall
        calls = 0

        async def generate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertIn("First [bold]", output.getvalue())
                self.assertIn("Push left as a baseline.", output.getvalue())
            return await acall(*args, **kwargs)

        provider.acall = generate
        sandbox = FakeSandbox()
        second_started, release = asyncio.Event(), asyncio.Event()

        async def evaluate(implementation, environment, seed, call_timeout):
            if "# second" in implementation:
                second_started.set()
                await release.wait()
                raise PolicyError("bad action")
            return 7.0, {}

        sandbox.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=3) as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent / "prompts"),
            Run.create(
                name="progress",
                path=Path(directory) / "run",
                environment=env,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as run,
        ):
            agent = AlphaEvolve("task", provider, config=Config(max_repairs=0))
            task = asyncio.create_task(
                example.run_search(agent, run, generations=1, batch_size=2, console=console)
            )
            try:
                await asyncio.wait_for(second_started.wait(), 2)
                for _ in range(100):
                    if "score=7" in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("score=7", output.getvalue())
                self.assertFalse(task.done())
            finally:
                release.set()
                with self.assertRaises(PolicyError):
                    await task
            text = output.getvalue()
            self.assertIn("Second", text)
            self.assertIn("bad action", text)
            saved = (run.path / "run.log").read_text()
            self.assertIn("Push left as a baseline.", saved)
            self.assertIn("score=7", saved)
            self.assertIn("bad action", saved)
            self.assertEqual(run.policies()[0].description, "Push left as a baseline.")
            self.assertEqual(agent.completed, 0)


if __name__ == "__main__":
    unittest.main()
