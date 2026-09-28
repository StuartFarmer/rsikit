"""Resume search state and cached episodes without paid calls or Docker."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from slick import prompts

from examples.elitelist_papers import run as runner
from research import elitesearch
from research.elitesearch import Config, EliteSearch, Generation, Measurement, Organism
from rsikit.evaluation import InfrastructureError
from tests.helpers import fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        template = patch.object(
            prompts, "TEMPLATE_ROOT", Path(elitesearch.__file__).parent / "prompts"
        )
        template.start()
        self.addCleanup(template.stop)

    def restorable(self):
        self.assertTrue(
            hasattr(EliteSearch, "restore"), "EliteSearch must restore persisted generations"
        )

    async def test_restore_keeps_finished_candidates_and_replays_parent_rng(self):
        self.restorable()
        config = Config(population_size=3, elite_size=2, generations=2)

        async def evaluate(policies):
            return {p.id: Measurement({0: float(p.name.split()[-1])}) for p in policies}

        original = EliteSearch(
            "test",
            ScriptedProvider([program(i) for i in range(3)]),
            evaluate,
            config=config,
            seed=41,
        )
        generation = Generation(number=1)
        original.generations.append(generation)
        rows = original._population(generation)
        await original._generate(rows[0])
        await original._measure([rows[0]])
        await original._generate(rows[1])
        rows[0].revisions.append(
            dict(
                implementation=json.loads(program(999))["implementation"],
                policy_id=None,
                status="rejected",
                error="Invalid edit boundary",
            )
        )
        # One evaluated candidate, one ready for evaluation, one not generated.
        saved_rows = [Organism.model_validate(r.model_dump()) for r in original.organisms]
        saved_generations = [
            Generation.model_validate(g.model_dump()) for g in original.generations
        ]
        resumed = EliteSearch(
            "test", ScriptedProvider([program(2)]), evaluate, config=config, seed=41
        )
        resumed.restore(saved_rows, saved_generations)
        self.assertEqual(resumed.rng.getstate(), original.rng.getstate())
        self.assertEqual(resumed._sources, original._sources)
        # Finish only generation one here; then compare the next allocation.
        await resumed._experiment(resumed.organisms)
        resumed._promote(resumed.generations[0], resumed.organisms)
        await original._generate(rows[2])
        await original._measure(rows[1:])
        original._promote(generation, rows)
        self.assertEqual([r.id for r in resumed.elites], [3, 2])
        self.assertEqual(len(resumed.provider.calls), 1)
        self.assertEqual(
            [(r.kind, r.parent_ids) for r in resumed._population(Generation(number=2))],
            [(r.kind, r.parent_ids) for r in original._population(Generation(number=2))],
        )

    async def test_interrupted_repair_keeps_budget_and_failed_material(self):
        self.restorable()
        for budget, expected in ((1, "discarded"), (2, "evaluated")):
            with self.subTest(budget=budget):
                config = Config(population_size=1, generations=1, max_repairs=budget)
                provider = ScriptedProvider([program(2)])

                async def evaluate(policies):
                    return {p.id: Measurement({0: 2.0}) for p in policies}

                agent = EliteSearch("test", provider, evaluate, config=config)
                row = Organism(
                    id=1,
                    generation=1,
                    kind="new",
                    status="cancelled",
                    repairs=1,
                    calls=[{"operation": "invent", "raw": "BROKEN_JSON"}, {"operation": "repair"}],
                    revisions=[
                        {
                            "implementation": "BROKEN_JSON",
                            "error": "invalid JSON",
                            "status": "rejected",
                        }
                    ],
                )
                agent.restore([row], [Generation(number=1, status="cancelled")])
                await agent.run()
                self.assertEqual(row.status, expected)
                self.assertEqual(row.repairs, budget)
                if budget == 2:
                    self.assertIn("BROKEN_JSON", provider.calls[0])
                else:
                    self.assertEqual(provider.calls, [])

    async def test_resume_reuses_cached_seeds_and_records_lower_timeout(self):
        first = FakeEvaluation()

        async def interrupted(source, environment, seed):
            if seed == 1:
                raise InfrastructureError("interrupted worker")
            return trajectory(7.0)

        first.evaluate.side_effect = interrupted
        second = FakeEvaluation()
        jobs = []

        async def finish(source, environment, seed):
            jobs.append(seed)
            return trajectory(7.0)

        second.evaluate.side_effect = finish
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    runner, "LoggedOpenRouter", return_value=ScriptedProvider([program(0)])
                ),
                patch.object(runner, "Executor", return_value=fake_executor(evaluation=first)),
                patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            ):
                with self.assertRaisesRegex(InfrastructureError, "interrupted worker"):
                    await runner.main(
                        [
                            "--env",
                            "Pendulum-v1",
                            "--population",
                            "1",
                            "--elites",
                            "1",
                            "--generations",
                            "1",
                            "--seeds",
                            "0",
                            "1",
                            "--heldout-seeds",
                            "100",
                            "--episode-timeout",
                            "6",
                            "--output",
                            str(output),
                        ]
                    )
            manifest = (output / "experiment.json").read_bytes()
            snapshot = (output / "source.zip").read_bytes()

            # Real Executor construction must receive the requested new timeout.
            deadlines = []

            def executor(**kwargs):
                deadlines.append(kwargs["episode_timeout"])
                return fake_executor(
                    evaluation=second,
                    concurrency=kwargs["concurrency"],
                )

            provider = ScriptedProvider([])
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(runner, "LoggedOpenRouter", return_value=provider),
                patch.object(runner, "Executor", side_effect=executor),
                patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            ):
                await runner.main(["--resume", str(output), "--episode-timeout", "0.25"])
            self.assertEqual(provider.calls, [])
            self.assertEqual(sorted(jobs), [1, 100, 100])
            self.assertEqual(deadlines, [0.25])
            self.assertEqual((output / "experiment.json").read_bytes(), manifest)
            self.assertEqual((output / "source.zip").read_bytes(), snapshot)
            attempts = json.loads((output / "attempts.json").read_text())
            self.assertEqual([a["episode_timeout"] for a in attempts], [6, 0.25])
            self.assertEqual(attempts[-1]["episode_timeout"], 0.25)
            self.assertEqual(attempts[-1]["status"], "completed")
            original_context = (output / attempts[0]["context"]).read_text()
            resumed_context = (output / attempts[-1]["context"]).read_text()
            self.assertIn("6 seconds per episode", original_context)
            self.assertIn("0.25 seconds per episode", resumed_context)
            self.assertNotIn("6 seconds per episode", resumed_context)
            self.assertIn("Early stopping is disabled", resumed_context)
            self.assertNotEqual(attempts[0]["context_sha256"], attempts[-1]["context_sha256"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["generations"], 1)
            self.assertTrue((output / "curves.csv").is_file())

    async def test_resume_rejects_changed_search_contract_and_active_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    runner, "LoggedOpenRouter", return_value=ScriptedProvider([program(0)])
                ),
                patch.object(
                    runner, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                ),
                patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            ):
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
            before = (output / "status.json").read_bytes()
            with (
                patch("sys.stderr", new=io.StringIO()),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
            ):
                with self.assertRaises(SystemExit):
                    await runner.main(["--resume", str(output), "--env", "Pendulum-v1"])
                with (
                    runner.make_environment("CartPole-v1") as env,
                    recorded_run(output, environment=env),
                ):
                    with self.assertRaises(SystemExit):
                        await runner.main(["--resume", str(output), "--episode-timeout", "1"])
            self.assertEqual((output / "status.json").read_bytes(), before)

    async def test_legacy_completed_run_extends_budget_then_resumes_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    runner, "LoggedOpenRouter", return_value=ScriptedProvider([program(0)])
                ),
                patch.object(
                    runner, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                ),
                patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            ):
                await runner.main(
                    [
                        "--env",
                        "Pendulum-v1",
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
            # This fixture represents the manifests written before resume and early stopping.
            (output / "attempts.json").unlink()
            manifest = json.loads((output / "experiment.json").read_text())
            manifest["policy_timeout"] = 10  # Old manifests may retain the removed option.
            manifest.pop("no_early_stop")
            manifest.pop("target_score")
            manifest["config"].pop("target_score")
            (output / "experiment.json").write_text(json.dumps(manifest))
            provider = ScriptedProvider(
                [
                    json.dumps(
                        dict(
                            name="Policy 1",
                            description="Edit",
                            edits=[dict(search="return 0", replacement="return 1")],
                        )
                    )
                ]
            )
            for flags in (["--generations", "2", "--episode-timeout", "0.3"], []):
                with (
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                    patch.object(runner, "LoggedOpenRouter", return_value=provider),
                    patch.object(
                        runner, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                    ),
                    patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
                ):
                    await runner.main(["--resume", str(output), *flags])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(json.loads((output / "summary.json").read_text())["generations"], 2)
            attempts = json.loads((output / "attempts.json").read_text())
            self.assertEqual([a["episode_timeout"] for a in attempts], [10, 0.3, 0.3])
            self.assertEqual([a["generations"] for a in attempts], [1, 2, 2])
            self.assertEqual(json.loads((output / "experiment.json").read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
