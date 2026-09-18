"""Keep the experimental comparison limited to founding and prompt feedback."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts

import alphaevolve
import examples.alphaevolve as example
from alphaevolve import improved, original
from examples.alphaevolve import run_search
from rsikit import Executor, Run
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program
from tests.test_run import FakeSandbox


class VariantTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_selects_variant_and_records_experiment(self):
        for variant in ("original", "improved"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = [
                    "alphaevolve",
                    "--variant",
                    variant,
                    "--generations",
                    "1",
                    "--batch-size",
                    "2",
                    "--seeds",
                    "0",
                    "1",
                    "--search-seed",
                    "7",
                    "--output",
                    str(output),
                ]
                with (
                    patch("sys.argv", args),
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                    patch.object(
                        example,
                        "OpenRouterAPI",
                        return_value=ScriptedProvider([program(0), program(1)]),
                    ),
                    patch.object(example, "Executor", return_value=Executor(sandbox=FakeSandbox())),
                    patch.object(example, "Console", return_value=Console(file=io.StringIO())),
                    patch.object(prompts, "TEMPLATE_ROOT"),
                ):
                    await example.main()
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertEqual(metadata["variant"], variant)
                self.assertEqual(metadata["search_seed"], 7)
                self.assertEqual(metadata["seeds"], [0, 1])
                self.assertIn(
                    f"Optimizer: alphaevolve.{variant}.agent", (output / "run.log").read_text()
                )
                with gym.make("CartPole-v1") as env, Run.open(output, environment=env) as run:
                    self.assertIn(variant, run.name)
                    self.assertEqual(len(run.policies()), 2)

    async def test_variants_share_mechanics_but_change_founding_and_feedback(self):
        rendered = {}
        for variant in (original, improved):
            with (
                self.subTest(variant=variant.__name__),
                patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
                tempfile.TemporaryDirectory() as directory,
                gym.make("CartPole-v1", max_episode_steps=3) as env,
                Run.create(
                    name="comparison",
                    path=Path(directory) / "run",
                    environment=env,
                    executor=Executor(sandbox=FakeSandbox()),
                ) as run,
            ):
                provider = ScriptedProvider([program(i) for i in range(9)])
                agent = variant.AlphaEvolve(
                    "task", provider, config=variant.Config(mode="rewrite", reset_interval=0)
                )
                policies = await agent.generate(n=8)
                scores = {p.id: float(i) for i, p in enumerate(policies)}
                details = {p.id: {0: 100.0 + i, 1: -100.0 + i} for i, p in enumerate(policies)}
                agent.update(scores, seed_scores=details)
                expected = [7] * 4 if variant is original else list(range(4, 8))
                self.assertEqual(
                    [p.policy.name for p in agent.islands], [f"Policy {i}" for i in expected]
                )
                self.assertEqual(agent.best, policies[7])
                parent = agent.islands[-1]
                texts = {}
                for operation in ("mutate", "rewrite"):
                    method = getattr(variant.AlphaEvolve, operation)
                    texts[operation] = await method.render(agent, parent, [parent], "", [])
                texts["guidance"] = await variant.AlphaEvolve.evolve_prompt.render(
                    agent, parent, [], []
                )
                for text in texts.values():
                    self.assertEqual("Per-seed rewards" in text, variant is improved)
                    self.assertEqual('"1": -93.0' in text, variant is improved)
                texts["initialize"] = provider.calls[0]
                texts["repair"] = await variant.AlphaEvolve.fix.render(
                    agent, "", "broken", "syntax"
                )
                rendered[variant.__name__] = texts
                output = io.StringIO()
                await run_search(
                    agent,
                    run,
                    generations=1,
                    batch_size=1,
                    seeds=(0, 1),
                    console=Console(file=output, force_terminal=False),
                )
                self.assertEqual(agent.completed, 9)
                self.assertEqual(len(provider.calls), 9)
                self.assertEqual(len(run.scores(run.policies()[0])), 2)
                saved = (run.path / "run.log").read_text()
                self.assertIn(variant.AlphaEvolve.__module__, saved)
                self.assertIn("Generated Policy 8", saved)
                self.assertIn("score=", saved)
                self.assertIn("Generated Policy 8", output.getvalue())
        for operation in ("initialize", "repair"):
            self.assertEqual(
                rendered[original.__name__][operation], rendered[improved.__name__][operation]
            )


if __name__ == "__main__":
    unittest.main()
