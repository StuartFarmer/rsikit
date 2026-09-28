"""Exercise baseline compatibility and the offline paper-variant Gym integration."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts, render
from sqlmodel import select

import examples.alphaevolve as example
from examples.alphaevolve import run_search
from research import alphaevolve
from research.alphaevolve import improved, original, paper
from research.alphaevolve.generation import Mutation, _PolicyResponse
from research.alphaevolve.history import Evaluation, Generation
from research.alphaevolve.original.agent import Guidance
from research.alphaevolve.paper.evaluation import assess
from research.rewards import mean_rewards
from rsikit.evaluation import PolicyError
from rsikit.policy import Policy
from rsikit.progress import show_scores
from tests.helpers import fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation


class VariantTests(unittest.IsolatedAsyncioTestCase):
    async def invoke_cli(self, arguments, provider, *, evaluation=None):
        with (
            patch("sys.argv", ["alphaevolve", *arguments]),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
            patch.object(example, "OpenRouterAPI", return_value=provider),
            patch.object(
                example,
                "Executor",
                return_value=fake_executor(evaluation=evaluation or FakeEvaluation()),
            ),
            patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
        ):
            await example.main()

    async def test_cli_selects_variant_and_records_experiment(self):
        for variant in ("original", "improved", "paper"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = [
                    "alphaevolve",
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
                if variant != "paper":
                    args.extend(["--variant", variant])
                with (
                    patch("sys.argv", args),
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                    patch.object(
                        example,
                        "OpenRouterAPI",
                        return_value=ScriptedProvider([program(0), program(1)]),
                    ),
                    patch.object(
                        example, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                    ),
                    patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
                    patch.object(prompts, "TEMPLATE_ROOT"),
                ):
                    await example.main()
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertEqual(metadata["variant"], variant)
                self.assertEqual(metadata["search_seed"], 7)
                self.assertEqual(metadata["seeds"], [0, 1])
                self.assertEqual(metadata["mode"], "diff")
                self.assertEqual(metadata["config"]["mode"], "diff")
                self.assertIn(
                    f"Optimizer: research.alphaevolve.{variant}.agent",
                    (output / "run.log").read_text(),
                )
                with (
                    gym.make("CartPole-v1") as env,
                    recorded_run(output, environment=env) as (run, rollouts),
                ):
                    self.assertIn(variant, run.name)
                    self.assertEqual(len(run.policies()), 2)

    async def test_cli_resume_restores_population_and_settings_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            config = root / "config.json"
            config.write_text(json.dumps({"islands": 1, "meta_interval": 0, "mode": "rewrite"}))
            initial = root / "initial.py"
            initial.write_text(program(0).implementation)
            await self.invoke_cli(
                [
                    "--output",
                    str(output),
                    "--generations",
                    "1",
                    "--batch-size",
                    "1",
                    "--search-config",
                    str(config),
                    "--initial-policy",
                    str(initial),
                    "--seeds",
                    "7",
                    "9",
                    "--max-steps",
                    "3",
                    "--model",
                    "saved-model",
                    "--ensemble",
                    "second-model",
                    "0.5",
                ],
                ScriptedProvider([program(1)]),
            )
            original_metadata = (output / "experiment.json").read_bytes()
            original_log = (output / "run.log").read_text()
            config.unlink()
            initial.unlink()
            provider = ScriptedProvider([program(2), program(3)])
            await self.invoke_cli(
                [
                    "--resume",
                    str(output),
                    "--generations",
                    "2",
                    "--generation-concurrency",
                    "1",
                    "--concurrency",
                    "2",
                ],
                provider,
            )
            self.assertEqual(len(provider.calls), 2)
            self.assertTrue(all("Current program to improve" in call for call in provider.calls))
            self.assertTrue(all("truncated after 3 steps" in call for call in provider.calls))
            self.assertEqual((output / "experiment.json").read_bytes(), original_metadata)
            self.assertTrue((output / "run.log").read_text().startswith(original_log))
            with example.make_environment("CartPole-v1", max_steps=3) as env:
                with recorded_run(output, environment=env) as (run, rollouts), run.database() as db:
                    self.assertEqual(len(run.policies()), 4)  # Initial seed plus three proposals.
                    self.assertTrue(all(set(run.scores(p)) == {7, 9} for p in run.policies()))
                    history = db.exec(select(Evaluation).order_by(Evaluation.attempt)).all()
                    self.assertEqual([row.attempt for row in history], [1, 2, 3])
                    self.assertTrue(all(row.parent_id for row in history))

    async def test_cli_resume_rejects_incompatible_or_missing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            await self.invoke_cli(
                ["--output", str(output), "--generations", "0"], ScriptedProvider([])
            )
            metadata_path = output / "experiment.json"
            metadata = metadata_path.read_bytes()
            for flags in (
                ["--seeds", "99"],
                ["--env", "LunarLander-v3"],
                ["--output", str(output)],
                ["--initial-policy", "missing.py"],
                ["--search-config", "missing.json"],
                ["--mode", "rewrite"],
            ):
                with self.subTest(flags=flags), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        await self.invoke_cli(
                            ["--resume", str(output), *flags], ScriptedProvider([])
                        )
                    self.assertEqual(caught.exception.code, 2)
                    self.assertEqual(metadata_path.read_bytes(), metadata)
            (output / "population.sqlite").unlink()
            error = io.StringIO()
            with patch("sys.stderr", error), self.assertRaises(SystemExit):
                await self.invoke_cli(["--resume", str(output)], ScriptedProvider([]))
            self.assertIn("population checkpoint", error.getvalue())
            self.assertFalse((output / "population.sqlite").exists())
            metadata_path.write_text(json.dumps({**json.loads(metadata), "variant": "improved"}))
            error = io.StringIO()
            with patch("sys.stderr", error), self.assertRaises(SystemExit):
                await self.invoke_cli(["--resume", str(output)], ScriptedProvider([]))
            self.assertIn("paper", error.getvalue())

    async def test_cli_mode_json_config_and_ensemble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"max_repairs": 0, "meta_interval": 0, "features": {}}))
            for mode in (None, "diff"):
                output = root / (mode or "default")
                arguments = [
                    "--env",
                    "BipedalWalker-v3",
                    "--generations",
                    "0",
                    "--output",
                    str(output),
                    "--search-config",
                    str(config),
                    "--ensemble",
                    "other-model",
                    "2.5",
                ]
                if mode:
                    arguments.extend(["--mode", mode, "--max-repairs", "1"])
                # Mode resolution uses the requested name; offline evaluation needs no Box2D.
                environment = example.make_environment("CartPole-v1")
                with patch.object(example, "make_environment", return_value=environment):
                    await self.invoke_cli(arguments, ScriptedProvider([]))
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertEqual(metadata["mode"], mode or "rewrite")
                self.assertEqual(metadata["config"]["max_repairs"], 1 if mode else 0)
                self.assertEqual(metadata["config"]["features"], {})
                self.assertEqual(metadata["ensemble"], [["other-model", "2.5"]])

    async def test_cli_rejects_invalid_config_and_screening(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"features": {"gait": [0, 1, 2]}}))
            objective = Path(directory) / "objective.json"
            objective.write_text(json.dumps({"objective": "foo"}))
            for arguments in (
                ["--search-config", str(config)],
                ["--search-config", str(objective)],
                ["--screening-seeds", "0"],
                ["--screening-min-reward", "0"],
                ["--screening-seeds", "0", "--screening-min-reward", "nan"],
                ["--ensemble", "model", "0"],
                ["--ensemble", "model", "nan"],
                ["--variant", "original", "--initial-policy", "source.py"],
            ):
                with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        await self.invoke_cli(arguments, ScriptedProvider([]))
                    self.assertEqual(caught.exception.code, 2)

    async def test_paper_pipeline_updates_population_and_persists_history(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            gym.make("CartPole-v1") as environment,
        ):
            evaluation = FakeEvaluation()

            async def evaluate(source, environment, seed):
                return trajectory(float(source.split("return ")[1].split()[0]) + seed, {})

            evaluation.evaluate.side_effect = evaluate
            generator = paper.AlphaEvolve(
                "task",
                ScriptedProvider([program(i) for i in range(8)]),
                config=paper.Config(
                    mode="rewrite",
                    islands=2,
                    meta_interval=0,
                    features=example.FEATURE_BOUNDS["CartPole-v1"],
                ),
            )
            self.addCleanup(generator.close)
            with recorded_run(
                name="paper",
                environment=environment,
                path=Path(directory) / "run",
                executor=fake_executor(evaluation=evaluation),
                console=Console(file=io.StringIO()),
            ) as (run, rollouts):
                await run_search(
                    generator,
                    run,
                    rollouts,
                    generations=4,
                    batch_size=2,
                    generation_concurrency=2,
                    seeds=(0, 2),
                )
                self.assertEqual(generator.completed, 8)
                self.assertEqual(generator.best.name, "Policy 7")
                self.assertTrue(
                    all((record["parent"] is not None for record in generator.attempts[2:]))
                )
                best = next(
                    (
                        c
                        for i in range(2)
                        for c in generator.database.members(i)
                        if c.policy.id == generator.best.id
                    )
                )
                self.assertEqual(best.metrics, {"reward": 8, "worst_reward": 7, "stability": -1})
                self.assertEqual(best.features, {"mean_reward": 8, "reward_std": 1})
                with run.database() as db:
                    attempts = db.exec(select(Evaluation)).all()
                    snapshots = db.exec(select(Generation)).all()
                    self.assertEqual(len(attempts), 8)
                    self.assertEqual({record.status for record in attempts}, {"evaluated"})
                    self.assertTrue(all((snapshot.complete for snapshot in snapshots)))
                    self.assertTrue(snapshots[-1].islands[0]["member_count"])

    async def test_screening_prunes_and_reuses_cached_seeds(self):
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as environment:
            evaluation = FakeEvaluation()

            async def evaluate(source, environment, seed):
                return trajectory(float(source.split("return ")[1].split()[0]) + seed, {})

            evaluation.evaluate.side_effect = evaluate
            policies = [Policy.from_text(program(i).implementation, name=str(i)) for i in (0, 10)]
            with recorded_run(
                name="cascade",
                environment=environment,
                path=Path(directory) / "run",
                executor=fake_executor(evaluation=evaluation),
            ) as (run, rollouts):
                results = await assess(
                    rollouts,
                    policies,
                    seeds=(0, 2),
                    features=("mean_reward", "reward_std"),
                    screening_seeds=(0,),
                    screening_min_reward=5,
                )
                self.assertFalse(results[policies[0].id].accepted)
                self.assertTrue(results[policies[1].id].accepted)
                self.assertEqual(results[policies[1].id].metrics["reward"], 11)
                self.assertEqual(evaluation.evaluate.await_count, 3)
                self.assertEqual(run.scores(policies[0]), {0: 0})

    async def test_screening_seed_is_excluded_from_displayed_final_scores(self):
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as environment:
            evaluation = FakeEvaluation()

            async def evaluate(source, environment, seed):
                return trajectory(float(seed), {})

            evaluation.evaluate.side_effect = evaluate
            policies = [Policy.from_text(program(i).implementation, name=str(i)) for i in (0, 1)]
            with recorded_run(
                name="screening",
                environment=environment,
                path=Path(directory) / "run",
                executor=fake_executor(evaluation=evaluation),
            ) as (run, rollouts):
                results = await assess(
                    rollouts,
                    policies[:1],
                    seeds=(0, 1),
                    features=(),
                    screening_seeds=(99,),
                    screening_min_reward=0,
                )
                await mean_rewards(rollouts, policies[1:], seeds=(99,))
                output = io.StringIO()
                show_scores(policies, run, Console(file=output), seeds=(0, 1))
                self.assertEqual(results[policies[0].id].metrics["reward"], 0.5)
                self.assertIn("0.5", output.getvalue())
                self.assertIn("unfinished", output.getvalue())
                self.assertNotIn("99.0", output.getvalue())

    async def test_paper_runner_rejects_evaluator_options_before_generation(self):
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as environment:
            provider = ScriptedProvider([])
            generator = paper.AlphaEvolve("task", provider)
            self.addCleanup(generator.close)
            with recorded_run(
                name="validation",
                environment=environment,
                path=Path(directory) / "run",
                executor=fake_executor(evaluation=FakeEvaluation()),
            ) as (run, rollouts):
                for options in (
                    {"screening_seeds": (99,)},
                    {"screening_min_reward": 0},
                    {"screening_seeds": (99,), "screening_min_reward": float("nan")},
                    {"seeds": ()},
                ):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        await example.run_paper_search(
                            generator, run, rollouts, generations=1, batch_size=1, **options
                        )
                self.assertEqual(provider.calls, [])

    async def test_paper_runtime_repair_and_cancellation_preserve_attempt_history(self):
        for cancel in (False, True):
            with (
                self.subTest(cancel=cancel),
                tempfile.TemporaryDirectory() as directory,
                patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
                gym.make("CartPole-v1") as environment,
            ):
                evaluation = FakeEvaluation()
                started = asyncio.Event()

                async def evaluate(source, environment, seed):
                    if "return 0" in source:
                        started.set()
                        if cancel:
                            await asyncio.Event().wait()
                        raise PolicyError("bad action")
                    return trajectory(7, {})

                evaluation.evaluate.side_effect = evaluate
                generator = paper.AlphaEvolve(
                    "task",
                    ScriptedProvider([program(0), program(1)]),
                    config=paper.Config(meta_interval=0),
                )
                self.addCleanup(generator.close)
                with recorded_run(
                    name="failure",
                    environment=environment,
                    path=Path(directory) / "run",
                    executor=fake_executor(evaluation=evaluation),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts):
                    task = asyncio.create_task(
                        run_search(generator, run, rollouts, generations=1, batch_size=1)
                    )
                    await asyncio.wait_for(started.wait(), 2)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await task
                    with run.database() as db:
                        rows = db.exec(select(Evaluation).order_by(Evaluation.revision)).all()
                        if cancel:
                            self.assertEqual([row.status for row in rows], ["generated"])
                            self.assertIsNone(generator.best)
                            self.assertFalse(db.exec(select(Generation)).one().complete)
                        else:
                            self.assertEqual([row.status for row in rows], ["failed", "evaluated"])
                            self.assertIn("bad action", rows[0].error)
                            self.assertEqual(generator.best.name, "Policy 1")

    async def test_initial_policy_is_validated_and_screened_before_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "initial.py"
            source.write_text(program(0).implementation)
            for threshold, expected in ((6, True), (8, False)):
                output = root / str(threshold)
                arguments = [
                    "--generations",
                    "0",
                    "--initial-policy",
                    str(source),
                    "--output",
                    str(output),
                    "--screening-seeds",
                    "0",
                    "--screening-min-reward",
                    str(threshold),
                ]
                with patch.object(paper.AlphaEvolve, "register_initial") as register:
                    if expected:
                        await self.invoke_cli(arguments, ScriptedProvider([]))
                        register.assert_called_once()
                        self.assertEqual(register.call_args.args[1].metrics["reward"], 7)
                    else:
                        with self.assertRaises(SystemExit):
                            await self.invoke_cli(arguments, ScriptedProvider([]))
                        register.assert_not_called()
            source.write_text(program(0).implementation + "\nclass Solution: pass\n")
            evaluation = FakeEvaluation()
            with self.assertRaises(SystemExit):
                await self.invoke_cli(
                    [
                        "--generations",
                        "0",
                        "--initial-policy",
                        str(source),
                        "--output",
                        str(root / "invalid"),
                    ],
                    ScriptedProvider([]),
                    evaluation=evaluation,
                )
            evaluation.evaluate.assert_not_awaited()

    async def test_overlapping_snapshot_stays_incomplete_when_sibling_is_cancelled(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            gym.make("CartPole-v1") as environment,
        ):
            evaluation = FakeEvaluation()
            provider = ScriptedProvider([program(i) for i in range(3)])
            generating, evaluated = (asyncio.Event(), asyncio.Event())
            acall = provider.acall

            async def generate(*args, **kwargs):
                index = len(provider.calls)
                response = await acall(*args, **kwargs)
                if index == 2:
                    generating.set()
                    await asyncio.Event().wait()
                return response

            async def evaluate(source, environment, seed):
                if "return 1" in source:
                    await generating.wait()
                return trajectory(7, {})

            def update(results):
                names = [row["policy"].name for key in results for row in generator._pending[key]]
                original_update(results)
                if "Policy 1" in names:
                    evaluated.set()

            provider.acall = generate
            evaluation.evaluate.side_effect = evaluate
            generator = paper.AlphaEvolve(
                "task", provider, config=paper.Config(islands=1, mode="rewrite", meta_interval=0)
            )
            self.addCleanup(generator.close)
            original_update = generator.update_results
            with (
                recorded_run(
                    name="overlap",
                    environment=environment,
                    path=Path(directory) / "run",
                    executor=fake_executor(evaluation=evaluation),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts),
                patch.object(generator, "update_results", side_effect=update),
            ):
                task = asyncio.create_task(
                    run_search(
                        generator,
                        run,
                        rollouts,
                        generations=3,
                        batch_size=1,
                        generation_concurrency=2,
                    )
                )
                try:
                    await asyncio.wait_for(evaluated.wait(), 2)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                with run.database() as db:
                    attempts = db.exec(select(Evaluation).order_by(Evaluation.attempt)).all()
                    self.assertEqual(
                        [row.status for row in attempts], ["evaluated", "evaluated", "cancelled"]
                    )
                    self.assertEqual(attempts[1].generation, attempts[2].generation)
                    self.assertFalse(db.get(Generation, attempts[2].generation).complete)
                    self.assertTrue(db.get(Generation, attempts[0].generation).complete)

    async def test_concurrent_generation_preserves_failed_revision_while_repair_waits(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            gym.make("CartPole-v1") as environment,
        ):
            evaluation = FakeEvaluation()
            provider = ScriptedProvider([program(i) for i in range(4)])
            sibling_started, repairing, sibling_saved = (asyncio.Event() for _ in range(3))
            acall = provider.acall

            async def generate(*args, **kwargs):
                index = len(provider.calls)
                response = await acall(*args, **kwargs)
                if index == 2:
                    sibling_started.set()
                    await repairing.wait()
                elif index == 3:
                    repairing.set()
                    await sibling_saved.wait()
                return response

            async def evaluate(source, environment, seed):
                if "return 1" in source:
                    await sibling_started.wait()
                    raise PolicyError("bad concurrent action")
                return trajectory(7, {})

            provider.acall = generate
            evaluation.evaluate.side_effect = evaluate
            generator = paper.AlphaEvolve(
                "task", provider, config=paper.Config(islands=1, mode="rewrite", meta_interval=0)
            )
            self.addCleanup(generator.close)
            with recorded_run(
                name="repair-overlap",
                environment=environment,
                path=Path(directory) / "run",
                executor=fake_executor(evaluation=evaluation),
                console=Console(file=io.StringIO()),
            ) as (run, rollouts):
                save = run.save

                def save_and_release(*records):
                    save(*records)
                    if any(
                        (
                            isinstance(row, Evaluation)
                            and row.attempt == 3
                            and (row.status == "generated")
                            for row in records
                        )
                    ):
                        sibling_saved.set()

                with patch.object(run, "save", side_effect=save_and_release):
                    await asyncio.wait_for(
                        run_search(
                            generator,
                            run,
                            rollouts,
                            generations=3,
                            batch_size=1,
                            generation_concurrency=2,
                        ),
                        3,
                    )
                with run.database() as db:
                    rows = db.exec(
                        select(Evaluation)
                        .where(Evaluation.attempt == 2)
                        .order_by(Evaluation.revision)
                    ).all()
                    self.assertEqual([row.status for row in rows], ["failed", "evaluated"])
                    self.assertIn("bad concurrent action", rows[0].error)
                    self.assertIsNone(rows[1].error)

    async def test_variants_share_mechanics_but_change_founding_and_feedback(self):
        rendered = {}
        for variant in (original, improved):
            terminal_output = io.StringIO()
            with (
                self.subTest(variant=variant.__name__),
                patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
                tempfile.TemporaryDirectory() as directory,
                gym.make("CartPole-v1", max_episode_steps=3) as env,
                recorded_run(
                    name="comparison",
                    path=Path(directory) / "run",
                    environment=env,
                    executor=fake_executor(evaluation=FakeEvaluation()),
                    console=Console(file=terminal_output, force_terminal=False),
                ) as (run, rollouts),
            ):
                provider = ScriptedProvider([program(i) for i in range(9)])
                agent = variant.AlphaEvolve(
                    "task", provider, config=variant.Config(mode="rewrite", reset_interval=0)
                )
                policies = await agent.generate(n=8)
                scores = {p.id: float(i) for i, p in enumerate(policies)}
                details = {p.id: {0: 100.0 + i, 1: -100.0 + i} for i, p in enumerate(policies)}
                agent.update_scores(scores, seed_scores=details)
                expected = [7] * 4 if variant is original else list(range(4, 8))
                self.assertEqual(
                    [p.policy.name for p in agent.islands], [f"Policy {i}" for i in expected]
                )
                self.assertEqual(agent.best, policies[7])
                parent = agent.islands[-1]
                texts = {}
                variant_name = variant.__name__.rsplit(".", 1)[-1]
                for operation, output in (("mutate", Mutation), ("rewrite", _PolicyResponse)):
                    texts[operation] = render(
                        f"{variant_name}/prompts/{operation}.j2",
                        instance=agent,
                        schema=output.model_json_schema(),
                        parent=parent,
                        inspirations=[parent],
                        guidance="",
                        failures=[],
                    )
                texts["guidance"] = render(
                    f"{variant_name}/prompts/evolve_prompt.j2",
                    instance=agent,
                    schema=Guidance.model_json_schema(),
                    parent=parent,
                    ideas=[],
                    failures=[],
                )
                for text in texts.values():
                    self.assertEqual("Per-seed rewards" in text, variant is improved)
                    self.assertEqual('"1": -93.0' in text, variant is improved)
                texts["initialize"] = provider.calls[0]
                texts["repair"] = render(
                    "original/prompts/repair.j2",
                    instance=agent.healer,
                    schema=_PolicyResponse.model_json_schema(),
                    reference="",
                    failed="broken",
                    diagnostic="syntax",
                )
                rendered[variant.__name__] = texts
                await run_search(agent, run, rollouts, generations=1, batch_size=1, seeds=(0, 1))
                self.assertEqual(agent.completed, 9)
                self.assertEqual(len(provider.calls), 9)
                self.assertEqual(len(run.scores(run.policies()[0])), 2)
                saved = (run.path / "run.log").read_text()
                self.assertIn(variant.AlphaEvolve.__module__, saved)
                self.assertIn("Generated Policy 8", saved)
                self.assertIn("score=", saved)
                self.assertIn("Generated Policy 8", terminal_output.getvalue())
        for operation in ("initialize", "repair"):
            self.assertEqual(
                rendered[original.__name__][operation], rendered[improved.__name__][operation]
            )


if __name__ == "__main__":
    unittest.main()
