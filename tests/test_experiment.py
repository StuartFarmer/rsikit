"""Shared runner integration with real evaluators and scripted model calls."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from research.cli import load_component, parse_config
from research.experiment import EvaluationConfig
from research.rewards import episode_scores
from rsikit import PolicyDefinition, Run
from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program


class ExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_paper_adapter_cleanup_preserves_primary_error_and_template_root(self):
        from slick import prompts

        from research.alphaevolve import cli, paper
        from tests.test_alphaevolve import program as alpha_program

        previous_root = prompts.TEMPLATE_ROOT
        primary = RuntimeError("original infrastructure failure")
        close = paper.AlphaEvolve.close

        def broken_close(agent):
            close(agent)
            raise OSError("checkpoint unavailable")

        async def evaluate(policies):
            raise primary

        with tempfile.TemporaryDirectory() as directory:
            with Run.create(name="cleanup", path=Path(directory) / "run") as run:
                with patch.object(paper.AlphaEvolve, "close", broken_close):
                    with self.assertRaises(RuntimeError) as raised:
                        await cli.optimize(
                            task="task",
                            provider=ScriptedProvider([alpha_program(0)]),
                            evaluate=evaluate,
                            run=run,
                            seed=0,
                            options=dict(
                                proposals=1,
                                generation=dict(concurrency=1, timeout=120),
                                videos=dict(top=0),
                            ),
                        )
                self.assertIs(raised.exception, primary)
                self.assertEqual(prompts.TEMPLATE_ROOT, previous_root)

    async def test_all_adapters_evaluate_admitted_candidates_when_budget_expires(self):
        from research.providers import BudgetProvider
        from tests.test_alphaevolve import program as alpha_program
        from tests.test_lineagesearch import experiments, families
        from tests.test_lineagesearch import program as lineage_program

        cases = [
            (
                "alphaevolve",
                dict(variant=v, proposals=2, proposal_batch_size=2, islands=1),
                [alpha_program(0)],
                1,
            )
            for v in ("paper", "original", "improved")
        ]
        cases += [
            (
                "shinka",
                dict(generations=1, proposal_batch_size=2, islands=1),
                [alpha_program(0)],
                1,
            ),
            ("elite", dict(generations=1, population=2, elites=1), [program(0)], 1),
            (
                "lineage",
                dict(families=1, decomposition_k=1, initial_per_family=2, max_attempts=2),
                [families(), experiments(0, 1), lineage_program(0)],
                3,
            ),
        ]
        for selector, options, responses, cap in cases:
            with (
                self.subTest(selector=selector, options=options),
                tempfile.TemporaryDirectory() as directory,
            ):
                component = load_component(selector, kind="optimizer", base_dir=Path.cwd())
                provider = BudgetProvider(
                    ScriptedProvider(responses),
                    max_calls=cap,
                    max_input_tokens=65536,
                    max_output_tokens=16384,
                )
                measured = []

                async def evaluate(policies):
                    measured.extend(policies)
                    return {p.id: episodes({0: 5}) for p in policies}

                with Run.create(name="budget", path=Path(directory) / "run") as run:
                    result = await component.optimize(
                        task="task",
                        provider=provider,
                        evaluate=evaluate,
                        run=run,
                        seed=0,
                        options={
                            **options,
                            "generation": dict(concurrency=1, timeout=120),
                            "videos": dict(top=0),
                        },
                    )
                    self.assertEqual(len(result), 1)
                    self.assertEqual([p.id for p in result], [p.id for p in measured])
                    self.assertEqual(provider.calls, cap)

    async def test_all_builtin_adapters_delegate_and_rank_finalists(self):
        from research.providers import BudgetProvider
        from rsikit import search
        from tests.test_alphaevolve import program as alpha_program
        from tests.test_lineagesearch import experiments, families
        from tests.test_lineagesearch import program as lineage_program

        cases = [
            (
                "alphaevolve",
                {
                    "variant": v,
                    "proposals": 2,
                    "proposal_batch_size": 2,
                    "islands": 1,
                    "meta_interval": 0,
                },
                [alpha_program(0), alpha_program(1)],
            )
            for v in ("paper", "original", "improved")
        ] + [
            (
                "shinka",
                {"generations": 1, "proposal_batch_size": 2, "islands": 1},
                [alpha_program(0), alpha_program(1)],
            ),
            ("elite", {"generations": 1, "population": 2, "elites": 1}, [program(0), program(1)]),
            (
                "lineage",
                {"families": 1, "decomposition_k": 1, "initial_per_family": 2, "max_attempts": 2},
                [families(), experiments(0, 1), lineage_program(0), lineage_program(1)],
            ),
        ]
        for selector, options, responses in cases:
            with (
                self.subTest(selector=selector, options=options),
                tempfile.TemporaryDirectory() as d,
            ):
                component = load_component(selector, kind="optimizer", base_dir=Path.cwd())
                config = parse_config(
                    ["run"],
                    config=dict(
                        env="CartPole-v1",
                        optimizer=selector,
                        model="scripted",
                        output=d,
                        optimizer_options=options,
                    ),
                )
                provider = BudgetProvider(
                    ScriptedProvider(responses), max_input_tokens=65536, max_output_tokens=16384
                )

                async def evaluate(policies):
                    return {p.id: episodes({0: i + 1}) for i, p in enumerate(policies)}

                with Run.create(name="adapter", path=Path(d) / "run") as run:
                    with patch.object(component, "search", wraps=search) as shared:
                        result = await component.optimize(
                            task="task",
                            provider=provider,
                            evaluate=evaluate,
                            run=run,
                            seed=0,
                            options={
                                **config["optimizer_options"],
                                "generation": config["generation"],
                                "videos": config["videos"],
                            },
                        )
                    self.assertEqual(shared.await_count, 1)
                    self.assertEqual(len(result), 2)
                    self.assertIn("1", result[0].name)
                    self.assertEqual(len({p.id for p in result}), 2)

    async def test_resume_keeps_completed_candidates_and_charges_previous_calls(self):
        from slick.providers import ProviderError
        from sqlmodel import select

        from research import experiment
        from research.elitesearch.records import Organism

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = parse_config(
                ["run"],
                config=dict(
                    env="CartPole-v1",
                    optimizer="elite",
                    model="scripted",
                    output=str(output),
                    optimizer_options=dict(
                        population=1, elites=1, generations=2, new_fraction=1.0, remix_fraction=0.0
                    ),
                    evaluation=dict(seeds=[7], max_steps=2, workers=1),
                    budget=dict(max_calls=3),
                ),
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(0), ProviderError("connection lost")]),
                ),
                self.assertRaisesRegex(ProviderError, "connection lost"),
            ):
                await experiment.run_experiment(config)
            original_config = (output / "config.yaml").read_bytes()
            original_manifest = (output / "manifest.json").read_bytes()
            self.assertEqual(json.loads(original_manifest)["optimization_schedule"], "round-v1")
            with Run.open(output) as run:
                with run.database() as db:
                    first = db.get(Organism, 1).model_dump()
            self.assertEqual(first["status"], "evaluated")
            moved = output.rename(Path(directory) / "moved")
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(1)]),
                ),
            ):
                from research.cli import main

                await asyncio.to_thread(main, ["resume", str(moved)])
                summary = json.loads((moved / "summary.json").read_text())
                # Resuming a finished search must not generate or evaluate it again.
                again = await experiment.resume_experiment(moved)
            self.assertEqual(summary, again)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["generation"]["calls"], 3)
            self.assertEqual(summary["phases"]["search"]["requested_candidates"], 2)
            self.assertEqual(summary["phases"]["search"]["seed_runs"], 2)
            self.assertEqual((moved / "config.yaml").read_bytes(), original_config)
            self.assertEqual((moved / "manifest.json").read_bytes(), original_manifest)
            with Run.open(moved) as run:
                with run.database() as db:
                    rows = db.exec(select(Organism).order_by(Organism.id)).all()
                    self.assertEqual(rows[0].model_dump(), first)
                    self.assertEqual([r.status for r in rows], ["evaluated", "evaluated"])

    async def test_legacy_resume_requires_completed_round_and_preserves_manifest(self):
        from research.elitesearch.records import Generation
        from research.experiment import resume_experiment, run_experiment

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = parse_config(
                ["run"],
                config=dict(
                    env="CartPole-v1",
                    optimizer="elite",
                    model="scripted",
                    output=str(output),
                    optimizer_options=dict(population=1, elites=1, generations=1),
                    evaluation=dict(seeds=[7], workers=1, max_steps=1),
                ),
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(0)]),
                ),
            ):
                await run_experiment(config)
            manifest = json.loads((output / "manifest.json").read_text())
            manifest.pop("optimization_schedule")
            original = json.dumps(manifest)
            (output / "manifest.json").write_text(original)
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch("research.providers.UsageOpenRouter", return_value=ScriptedProvider([])),
            ):
                await resume_experiment(output)
            self.assertEqual((output / "manifest.json").read_text(), original)
            event = json.loads((output / "schedule_changes.jsonl").read_text())
            self.assertEqual(event["to"], "round-v1")
            with Run.open(output) as run:
                with run.database() as db:
                    row = db.get(Generation, 1)
                    row.status = "running"
                    db.add(row)
                    db.commit()
            with self.assertRaisesRegex(ValueError, "incomplete legacy"):
                await resume_experiment(output)

    async def test_gym_sessions_match_direct_episode_and_cache(self):
        from rsikit.envs.tasks import make_environment
        from rsikit.evaluation import _run_episode
        from rsikit.policy import load_policy

        for name in ("CartPole-v1", "Blackjack"):
            with self.subTest(env=name), tempfile.TemporaryDirectory() as directory:
                definition = load_component(name, kind="environment", base_dir=Path.cwd())
                options = definition.Options(
                    **({"shoes_per_episode": 1} if name == "Blackjack" else {})
                )
                evaluation = EvaluationConfig(
                    seeds=[7], workers=1, max_steps=5 if name == "CartPole-v1" else None
                )
                policy = PolicyDefinition.from_text(json.loads(program(0))["implementation"])
                direct = await _run_episode(
                    lambda: make_environment(
                        name, max_steps=evaluation.max_steps, shoes_per_episode=1
                    ),
                    lambda obs, act, instructions: load_policy(
                        policy.source, obs, act, instructions
                    ),
                    env_seed=7,
                    policy_seed=7,
                )
                with Run.create(name="session", path=Path(directory) / "run") as run:
                    async with definition.open_evaluator(
                        options=options, evaluation=evaluation, run=run
                    ) as evaluate:
                        result = await evaluate([policy], [7])
                        self.assertIs(type(result[policy.id]), dict)
                        self.assertEqual(
                            episode_scores(result[policy.id]), {7: direct.total_reward}
                        )
                        cached = await evaluate([policy], [7])

                        self.assertEqual(
                            cached[policy.id][7].encode(),
                            result[policy.id][7].encode(),
                        )
                        self.assertIsNotNone(run.load_episode(policy, 7))

    async def test_ocean_session_matches_upstream_rollout(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import rollout

        for name in ("g2048", "breakout"):
            with self.subTest(env=name), tempfile.TemporaryDirectory() as directory:
                definition = load_component(
                    "ocean:" + name, kind="environment", base_dir=Path.cwd()
                )
                evaluation = EvaluationConfig(
                    seeds=[7], workers=1, batch_size=2, max_steps=4, score_key="return"
                )
                policy = policies(name)[0]
                direct = await rollout(
                    policy.source,
                    [7],
                    batch_size=2,
                    max_steps=4,
                    env_name=name,
                    score_key="return",
                )
                with Run.create(name="session", path=Path(directory) / "run") as run:
                    async with definition.open_evaluator(
                        options=definition.Options(), evaluation=evaluation, run=run
                    ) as evaluate:
                        result = await evaluate([policy], [7])
                    self.assertIs(type(result[policy.id]), dict)
                    self.assertEqual(
                        episode_scores(result[policy.id])[7], direct["results"][0]["score"]
                    )
                    events = [
                        json.loads(s)
                        for s in (run.path / "panels/evaluations.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(events[0]["steps"], 4)

    async def test_scripted_search_real_backends_and_evaluate_only(self):
        from research.experiment import evaluate_policies, run_experiment

        for name in ("CartPole-v1", "ocean:g2048", "ocean:breakout"):
            with self.subTest(env=name), tempfile.TemporaryDirectory() as directory:
                args = [
                    "run",
                    "--env",
                    name,
                    "--optimizer",
                    "elite",
                    "--model",
                    "scripted",
                    "--output",
                    str(Path(directory) / "search"),
                    "--population",
                    "2",
                    "--generations",
                    "1",
                    "--elites",
                    "1",
                    "--max-repairs",
                    "0",
                    "--seeds",
                    "7",
                    "--test-seeds",
                    "9",
                    "--max-steps",
                    "4",
                    "--workers",
                    "1",
                ]
                if name.startswith("ocean:"):
                    args += ["--batch-size", "2"]
                responses = [program(i) for i in range(2)]
                if name.startswith("ocean:"):
                    responses = [
                        json.dumps(
                            {
                                **json.loads(p),
                                "implementation": json.loads(p)["implementation"].replace(
                                    f"return {i}", f"return [{i}] * len(observation)"
                                ),
                            }
                        )
                        for i, p in enumerate(responses)
                    ]
                raw = ScriptedProvider(responses)
                config = parse_config(args)
                with (
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                    patch("research.providers.UsageOpenRouter", return_value=raw),
                ):
                    summary = await run_experiment(config)
                self.assertEqual(summary["status"], "completed", summary)
                self.assertEqual(summary["generation"]["calls"], 2)
                self.assertIsNone(summary["generation"]["reserved_cost"])
                self.assertEqual(summary["phases"]["search"]["requested_candidates"], 2)
                self.assertIsNone(summary["generation"]["output_tokens"])
                winner = Path(config["output"]) / "winner.py"
                self.assertTrue(winner.is_file())
                saved = yaml.safe_load((winner.parent / "config.yaml").read_text())
                self.assertEqual(saved, config)
                evaluate_args = [
                    "evaluate",
                    "--env",
                    name,
                    "--policy",
                    str(winner),
                    "--output",
                    str(Path(directory) / "evaluation"),
                    "--seeds",
                    "9",
                    "--max-steps",
                    "4",
                    "--workers",
                    "1",
                ]
                if name.startswith("ocean:"):
                    evaluate_args += ["--batch-size", "2"]
                with patch(
                    "research.providers.UsageOpenRouter",
                    side_effect=AssertionError("No model needed"),
                ):
                    result = await evaluate_policies(parse_config(evaluate_args))
                self.assertEqual(result["measurements"][summary["winner"]], summary["test"])

    async def test_custom_files_selection_and_budget_exhaustion(self):
        from research.experiment import run_experiment

        environment_source = """from contextlib import asynccontextmanager
+from research.experiment import EnvironmentDefinition, Options
+from tests.helpers import episodes
+@asynccontextmanager
+async def open_evaluator(*, options, evaluation, run):
+    async def evaluate(policies, seeds):
+        with (run.path / "panels.txt").open("a") as f:
+            f.write(str(list(seeds)) + "\\n")
+        if 200 in seeds:
+            assert (run.path / "winner.py").exists()
+        return {p.id: episodes({s: (10 if p.name == "Policy 1" and s == 100 else 1) for s in seeds}) for p in policies}
+    yield evaluate
+environment = EnvironmentDefinition(Options, lambda p: None, lambda o, e: "Test scoring", open_evaluator)
+""".replace("\n+", "\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "environment.py"
            env.write_text(environment_source)
            config = parse_config(
                [
                    "run",
                    "--env",
                    str(env),
                    "--optimizer",
                    "elite",
                    "--model",
                    "scripted",
                    "--output",
                    str(root / "run"),
                    "--population",
                    "3",
                    "--generations",
                    "1",
                    "--elites",
                    "2",
                    "--max-repairs",
                    "0",
                    "--max-calls",
                    "2",
                    "--spend-cap",
                    "100",
                    "--input-price",
                    "1",
                    "--output-price",
                    "1",
                    "--seeds",
                    "7",
                    "--validation-seeds",
                    "100",
                    "--test-seeds",
                    "200",
                    "--finalists",
                    "2",
                ]
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(0), program(1)]),
                ),
            ):
                summary = await run_experiment(config)
            self.assertEqual(summary["status"], "budget_exhausted", summary)
            self.assertEqual(PolicyDefinition.from_file(root / "run/winner.py").name, "Policy 1")
            self.assertEqual(
                (root / "run/panels.txt").read_text().splitlines(), ["[7]", "[100]", "[200]"]
            )

    async def test_selection_failure_ties_and_empty_panels(self):
        from research.experiment import select_winner

        policies = [
            PolicyDefinition.from_text(json.loads(program(i))["implementation"], name=f"Policy {i}")
            for i in range(2)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            Run.create(name="selection", path=Path(directory) / "run") as run,
        ):
            panels = []

            async def evaluate(ps, seeds, phase):
                panels.append((phase, [p.id for p in ps]))
                if phase == "test":
                    self.assertTrue((run.path / "winner.py").exists())
                    return {p.id: episodes(failure="test failure") for p in ps}
                return {p.id: episodes({s: 1 for s in seeds}) for p in ps}

            options = dict(finalists=2, validation_seeds=[100], test_seeds=[200])
            result = await select_winner(policies, evaluate, run, options)
            self.assertEqual(result["winner"], policies[0].id)
            self.assertEqual(result["status"], "evaluation_failed")
            self.assertEqual(panels[-1], ("test", [policies[0].id]))
            panels.clear()
            result = await select_winner([], evaluate, run, options)
            self.assertEqual(result["status"], "no_valid_candidate")
            self.assertEqual(panels, [])
            result = await select_winner(
                policies, evaluate, run, {**options, "validation_seeds": [], "test_seeds": []}
            )
            self.assertEqual(result["winner"], policies[0].id)
            self.assertEqual(result["test"], "not evaluated")

    async def test_cancellation_and_infrastructure_leave_summary(self):
        from research.experiment import run_experiment

        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as directory:
                config = parse_config(
                    ["run"],
                    config=dict(
                        env="CartPole-v1",
                        optimizer="elite",
                        model="scripted",
                        output=str(Path(directory) / "run"),
                        optimizer_options=dict(
                            population=1, elites=1, generations=1, max_repairs=0
                        ),
                        budget=dict(spend_cap=10, input_price=1, output_price=1),
                    ),
                )
                entered, closed = asyncio.Event(), asyncio.Event()

                class FailedProvider:
                    async def acall(self, *args, **kwargs):
                        entered.set()
                        try:
                            if cancel:
                                await asyncio.Event().wait()
                            raise RuntimeError("provider offline")
                        finally:
                            closed.set()

                with (
                    patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                    patch("research.providers.UsageOpenRouter", return_value=FailedProvider()),
                ):
                    task = asyncio.create_task(run_experiment(config))
                    await asyncio.wait_for(entered.wait(), 5)
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                        await task
                self.assertTrue(closed.is_set())
                summary = json.loads((Path(config["output"]) / "summary.json").read_text())
                self.assertEqual(summary["status"], "cancelled" if cancel else "failed")
                self.assertNotIn("test", summary)

    async def test_ocean_infrastructure_is_not_a_policy_failure(self):
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator
        from rsikit.evaluation import InfrastructureError

        with tempfile.TemporaryDirectory() as directory:
            async with PanelEvaluator(directory, max_steps=2) as evaluator:
                with patch.object(
                    evaluator, "_panel", side_effect=InfrastructureError("worker missing")
                ):
                    with self.assertRaises(InfrastructureError):
                        await evaluator.evaluate([policies()[0]], [0])

    async def test_custom_optimizer_file_and_cached_execution_counts(self):
        from research.experiment import run_experiment

        source = """from pydantic import BaseModel, ConfigDict
from rsikit import PolicyDefinition
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    repeats: int = 2
 def_placeholder
""".replace(
            " def_placeholder",
            """def add_arguments(parser):
    parser.add_argument("--repeats", type=int)
async def optimize(*, task, provider, evaluate, run, options, seed):
    p = PolicyDefinition.from_text("from rsikit import Policy\\nclass Solution(Policy):\\n    async def act(self, observation): return 0\\n")
    for _ in range(options["repeats"]):
        await evaluate([p])
    return [p]
""",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer = root / "optimizer.py"
            optimizer.write_text(source)
            config = parse_config(
                ["run"],
                config=dict(
                    env="CartPole-v1",
                    optimizer=str(optimizer),
                    model="scripted",
                    output=str(root / "run"),
                    evaluation=dict(seeds=[7], max_steps=2, workers=1),
                    budget=dict(spend_cap=10, input_price=1, output_price=1, max_calls=1),
                ),
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch("research.providers.UsageOpenRouter", return_value=ScriptedProvider([])),
            ):
                summary = await run_experiment(config)
            self.assertEqual(summary["status"], "completed")
            tally = summary["phases"]["search"]
            self.assertEqual(tally["requested_candidates"], 2)
            self.assertEqual(tally["evaluation_submissions"], 1)
            self.assertEqual(tally["seed_runs"], 1)
            self.assertEqual(summary["generation"]["calls"], 0)

    async def test_video_queue_keeps_blackjack_metadata(self):
        from research.experiment import run_experiment

        completed = []

        async def videos(queue, path, top, workers):
            while (number := await queue.get()) is not None:
                completed.append(number)
            metadata = json.loads((path / "experiment.json").read_text())
            self.assertEqual(metadata["shoes_per_seed"], 1)
            self.assertEqual((top, workers), (1, 1))

        with tempfile.TemporaryDirectory() as directory:
            config = parse_config(
                ["run"],
                config=dict(
                    env="Blackjack",
                    optimizer="elite",
                    model="scripted",
                    output=str(Path(directory) / "run"),
                    environment=dict(shoes_per_episode=1),
                    optimizer_options=dict(population=1, elites=1, generations=1, max_repairs=0),
                    evaluation=dict(seeds=[7], workers=1),
                    videos=dict(top=1),
                    budget=dict(spend_cap=10, input_price=1, output_price=1),
                ),
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(0)]),
                ),
                patch("research.elitesearch.videos.generation_videos", side_effect=videos),
            ):
                summary = await run_experiment(config)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(completed, [1])

    async def test_real_blackjack_generation_video_and_replay_accounting(self):
        from research.elitesearch.videos import export
        from research.experiment import run_experiment

        with tempfile.TemporaryDirectory() as directory:
            config = parse_config(
                ["run"],
                config=dict(
                    env="Blackjack",
                    optimizer="elite",
                    model="scripted",
                    output=str(Path(directory) / "run"),
                    environment=dict(shoes_per_episode=1),
                    optimizer_options=dict(population=1, elites=1, generations=1, max_repairs=0),
                    evaluation=dict(seeds=[7], workers=1),
                    videos=dict(top=1),
                    budget=dict(spend_cap=10, input_price=1, output_price=1),
                ),
            )
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
                patch(
                    "research.providers.UsageOpenRouter",
                    return_value=ScriptedProvider([program(0)]),
                ),
            ):
                summary = await run_experiment(config)
            root = Path(config["output"])
            self.assertTrue((root / "videos/index.html").is_file())
            self.assertEqual(summary["phases"]["rendering"]["seed_runs"], 1)
            before = (root / "videos/executions.jsonl").read_text()
            await export(root, root / "videos", top=1, workers=1)
            self.assertEqual((root / "videos/executions.jsonl").read_text(), before)

    async def test_ocean_cancelled_execution_is_counted(self):
        from research.experiment import evaluate_policies
        from research.ocean.baselines import policies
        from research.ocean.evaluator import PanelEvaluator

        entered = asyncio.Event()

        async def panel(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.py"
            policies()[0].to_file(policy)
            config = parse_config(
                [
                    "evaluate",
                    "--env",
                    "ocean:g2048",
                    "--policy",
                    str(policy),
                    "--output",
                    str(root / "run"),
                    "--seeds",
                    "7",
                    "--max-steps",
                    "2",
                ]
            )
            with patch.object(PanelEvaluator, "_panel", side_effect=panel):
                task = asyncio.create_task(evaluate_policies(config))
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["status"], "cancelled")
            self.assertEqual(summary["phases"]["evaluation"]["evaluation_submissions"], 1)
            self.assertIsNone(summary["phases"]["evaluation"]["transitions"])
