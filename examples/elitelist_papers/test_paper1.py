"""Offline checks for benchmark configuration and leakage-free reporting."""

import csv
import importlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import numpy as np
from rich.console import Console

from rsikit import Executor
from rsikit.evaluation import PolicyError
from tests.helpers import recorded_run
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeSandbox


class Paper1Tests(unittest.IsolatedAsyncioTestCase):
    def runner(self):
        self.assertIsNotNone(
            importlib.util.find_spec("examples.elitelist_papers.run"),
            "Paper 1 needs a runnable CLI with standard Gymnasium profiles",
        )
        return importlib.import_module("examples.elitelist_papers.run")

    def test_standard_environments_and_serialized_instructions(self):
        runner = self.runner()
        for name in runner.TASKS:
            with self.subTest(env=name), runner.make_environment(name) as env:
                restored = cloudpickle.loads(cloudpickle.dumps(env))
                try:
                    observation, _ = restored.reset(seed=0)
                    self.assertTrue(restored.observation_space.contains(observation))
                    self.assertIn("Action", restored.get_wrapper_attr("instructions"))
                    self.assertEqual(
                        restored.paper_settings, json.loads(json.dumps(restored.paper_settings))
                    )
                    self.assertGreater(restored.spec.max_episode_steps, 0)
                    restored.step(restored.action_space.sample())
                    if name == "LunarLander-v3":
                        self.assertEqual(restored.action_space.n, 4)
                        self.assertFalse(restored.unwrapped.enable_wind)
                    if name == "LunarLanderContinuous-v3":
                        self.assertEqual(restored.action_space.shape, (2,))
                    if name == "CarRacing-v3":
                        self.assertEqual(observation.shape, (96, 96, 3))
                    if name == "Blackjack-v1":
                        self.assertIsInstance(observation, tuple)
                    if name.startswith("BipedalWalker"):
                        self.assertIn("[13] second lower-leg ground contact", restored.instructions)
                        self.assertIn("[14:24]", restored.instructions)
                        self.assertIn("torque", restored.instructions)
                finally:
                    restored.close()

    async def test_invalid_inputs_fail_before_provider_or_output(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "untouched"
            for flags in (
                ["--seeds", "0", "--heldout-seeds", "0"],
                ["--population", "0"],
                ["--max-steps", "0"],
                ["--episode-timeout", "nan"],
                ["--policy-timeout", "0"],
                ["--policy-timeout", "nan"],
                ["--seeds", "-1"],
                ["--target-score", "nan"],
            ):
                with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit):
                        await runner.main(["--output", str(destination), *flags])
                self.assertFalse(destination.exists())

    async def test_cli_exports_all_generations_without_heldout_selection(self):
        runner = self.runner()
        sandbox = FakeSandbox()

        async def evaluate(source, environment, seed, call_timeout):
            if "action_space.sample" in source:
                return trajectory(-2.0, {})
            value = int(source.split("return ")[-1].strip())
            # Better search candidates deliberately generalize worse.
            return trajectory(float(value if seed < 100 else 10 - value), {})

        sandbox.evaluate.side_effect = evaluate
        provider = ScriptedProvider([])
        next_value = 0
        contexts = []

        async def respond(context, **kwargs):
            nonlocal next_value
            contexts.append(context)
            value = next_value
            next_value += 1
            if "Return exact search/replacement edits" in context:
                parent_value = context.split("return ")[-1].split()[0]
                return json.dumps(
                    dict(
                        name=f"Policy {value}",
                        description="Edited",
                        edits=[
                            dict(search=f"return {parent_value}", replacement=f"return {value}")
                        ],
                    )
                ), []
            return program(value), []

        provider.acall = respond
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(runner, "LoggedOpenRouter", return_value=provider),
                patch.object(
                    runner, "Executor", return_value=Executor(sandbox=sandbox)
                ) as executor,
                patch.object(runner, "Console", return_value=Console(file=io.StringIO())),
            ):
                await runner.main(
                    [
                        "--env",
                        "CartPole-v1",
                        "--population",
                        "2",
                        "--elites",
                        "2",
                        "--generations",
                        "2",
                        "--seeds",
                        "0",
                        "--heldout-seeds",
                        "100",
                        "101",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(executor.call_args.kwargs["sandbox"].episode_timeout, 10)
            manifest = json.loads((output / "experiment.json").read_text())
            context = (output / "context.txt").read_text()
            self.assertEqual(context, manifest["instructions"])
            self.assertIn("10 seconds per episode", context)
            self.assertIn("10 seconds per policy call", context)
            self.assertIn("mean search reward >= 475", context)
            self.assertTrue(all(context in prompt for prompt in contexts))
            rows = list(csv.DictReader(io.StringIO((output / "curves.csv").read_text())))
            self.assertEqual([row["generation"] for row in rows], ["1", "2"])
            self.assertEqual([float(row["search_mean"]) for row in rows], [1.0, 3.0])
            self.assertEqual([float(row["heldout_mean"]) for row in rows], [9.0, 7.0])
            heldout = json.loads((output / "heldout.json").read_text())
            self.assertEqual(heldout["random"]["scores"], {"100": -2.0, "101": -2.0})
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "completed"
            )
            self.assertTrue((output / "source.zip").is_file())
            with zipfile.ZipFile(output / "source.zip") as archive:
                for name in (
                    "uv.lock",
                    "rsikit/sandbox/Dockerfile",
                    "rsikit/policy.py",
                    "rsikit/envs/tasks.py",
                    "rsikit/progress.py",
                    "research/__init__.py",
                    "research/elitesearch/agent.py",
                    "research/elitesearch/generation.py",
                    "research/elitesearch/prompts/new.j2",
                ):
                    self.assertEqual(archive.read(name), (runner.ROOT / name).read_bytes())
            with (
                runner.make_environment("CartPole-v1") as env,
                recorded_run(output, environment=env) as (run, rollouts),
            ):
                self.assertTrue(
                    all(np.isfinite(v) for p in run.policies() for v in run.scores(p).values())
                )

    async def test_failed_heldout_is_missing_not_zero(self):
        runner = self.runner()
        sandbox = FakeSandbox()

        async def evaluate(source, environment, seed, call_timeout):
            if seed >= 100 and "action_space.sample" not in source:
                raise PolicyError("unseen-state failure")
            return trajectory(7.0, {})

        sandbox.evaluate.side_effect = evaluate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    runner, "LoggedOpenRouter", return_value=ScriptedProvider([program(0)])
                ),
                patch.object(runner, "Executor", return_value=Executor(sandbox=sandbox)),
                patch.object(runner, "Console", return_value=Console(file=io.StringIO())),
            ):
                await runner.main(
                    [
                        "--population",
                        "1",
                        "--elites",
                        "1",
                        "--generations",
                        "1",
                        "--seeds",
                        "0",
                        "--heldout-seeds",
                        "100",
                        "--output",
                        str(output),
                    ]
                )
            row = next(csv.DictReader(io.StringIO((output / "curves.csv").read_text())))
            self.assertEqual(row["heldout_mean"], "")
            self.assertIn("unseen-state failure", row["heldout_error"])

    async def test_usage_logging_preserves_unknowns_and_errors(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calls.jsonl"
            provider = runner.LoggedOpenRouter(output=output, model="test")
            response = SimpleNamespace(
                model_dump=lambda **kwargs: dict(
                    id="generation-123",
                    model="resolved-model",
                    provider="provider-name",
                    usage={"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.01},
                )
            )
            with patch.object(runner.OpenRouterAPI, "_asend", return_value=response):
                await provider._asend({"messages": [{"role": "user", "content": "prompt"}]})
            with patch.object(runner.OpenRouterAPI, "_asend", side_effect=RuntimeError("offline")):
                with self.assertRaises(RuntimeError):
                    await provider._asend({"messages": []})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(records[0]["usage"]["cost"], 0.01)
            self.assertEqual(records[0]["provider"], "provider-name")
            self.assertIn("offline", records[1]["error"])
            self.assertNotIn("usage", records[1])

    async def test_early_stopping_uses_search_target_and_keeps_outputs(self):
        runner = self.runner()
        for env, flags, expected_generations, reason, reporting_target in (
            ("CartPole-v1", [], 1, "target_reached", 475),
            ("CartPole-v1", ["--target-score", "475"], 1, "target_reached", 475),
            ("CartPole-v1", ["--target-score", "476"], 3, "completed", 476),
            ("CartPole-v1", ["--no-early-stop"], 3, "completed", 475),
            ("CartPole-v1", ["--no-early-stop", "--target-score", "474"], 3, "completed", 474),
            ("Pendulum-v1", [], 3, "completed", None),
        ):
            with self.subTest(env=env, flags=flags), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                sandbox = FakeSandbox()

                async def evaluate(source, environment, seed, call_timeout):
                    # Search meets the target exactly; held-out results do not.
                    return trajectory(475.0 if seed == 0 else -100.0, {})

                sandbox.evaluate.side_effect = evaluate
                provider = ScriptedProvider(
                    [
                        program(0),
                        *[
                            json.dumps(
                                dict(
                                    name=f"Policy {i}",
                                    description="Edited",
                                    edits=[
                                        dict(search=f"return {i - 1}", replacement=f"return {i}")
                                    ],
                                )
                            )
                            for i in (1, 2)
                        ],
                    ]
                )
                with (
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                    patch.object(runner, "LoggedOpenRouter", return_value=provider),
                    patch.object(runner, "Executor", return_value=Executor(sandbox=sandbox)),
                    patch.object(runner, "Console", return_value=Console(file=io.StringIO())),
                ):
                    await runner.main(
                        [
                            "--env",
                            env,
                            "--population",
                            "1",
                            "--elites",
                            "1",
                            "--generations",
                            "3",
                            "--seeds",
                            "0",
                            "--heldout-seeds",
                            "100",
                            "--output",
                            str(output),
                            *flags,
                        ]
                    )
                summary = json.loads((output / "summary.json").read_text())
                manifest = json.loads((output / "experiment.json").read_text())
                self.assertEqual(manifest["reporting_target"], reporting_target)
                self.assertEqual(summary["reason"], reason)
                self.assertEqual(summary["generations"], expected_generations)
                self.assertEqual(len(provider.calls), expected_generations)
                context = (output / "context.txt").read_text()
                expected_target = (
                    "Early stopping is disabled"
                    if "--no-early-stop" in flags or env == "Pendulum-v1"
                    else f"mean search reward >= {flags[-1] if flags else '475'}"
                )
                self.assertIn(expected_target, context)
                self.assertTrue(all(context in prompt for prompt in provider.calls))
                self.assertEqual(summary["heldout"]["scores"], {"100": -100.0})
                rows = list(csv.DictReader(io.StringIO((output / "curves.csv").read_text())))
                self.assertEqual(len(rows), expected_generations)
                self.assertTrue((output / "best.py").is_file())
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"], "completed"
                )

    async def test_reporting_failure_does_not_mark_run_completed(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    runner, "LoggedOpenRouter", return_value=ScriptedProvider([program(0)])
                ),
                patch.object(runner, "Executor", return_value=Executor(sandbox=FakeSandbox())),
                patch.object(runner, "Console", return_value=Console(file=io.StringIO())),
                patch.object(runner, "export_curves", side_effect=RuntimeError("report failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "report failed"):
                    await runner.main(
                        [
                            "--population",
                            "1",
                            "--generations",
                            "1",
                            "--seeds",
                            "0",
                            "--heldout-seeds",
                            "100",
                            "--output",
                            str(output),
                        ]
                    )
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("report failed", status["error"])


if __name__ == "__main__":
    unittest.main()
