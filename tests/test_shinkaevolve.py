"""Exercise the main-branch Shinka mechanisms through the current Run boundary."""

import asyncio
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from jinja2 import Environment, nodes
from rich.console import Console
from slick import prompts
from slick.providers import ProviderError
from sqlmodel import select

import examples.shinkaevolve as example
from examples.shinkaevolve import run_search
from research import shinkaevolve
from research.shinkaevolve import Config, Evaluation, Generation, ShinkaEvolve
from research.shinkaevolve.generation import Edit, Mutation
from rsikit.evaluation import PolicyError
from tests.helpers import fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import SOURCE, program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation

ROOT = Path(shinkaevolve.__file__).parent / "prompts"


class ShinkaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_cli_selects_models_and_writes_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            console = Console(file=io.StringIO())
            with (
                patch(
                    "sys.argv",
                    [
                        "shinkaevolve",
                        "--generations",
                        "1",
                        "--batch-size",
                        "2",
                        "--seeds",
                        "0",
                        "1",
                        "--model",
                        "first",
                        "--model",
                        "second",
                        "--output",
                        str(output),
                    ],
                ),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    example,
                    "OpenRouterAPI",
                    return_value=ScriptedProvider([program(0), program(1)]),
                ) as create_provider,
                patch.object(
                    example, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                ),
                patch("rsikit.progress.controller.Console", return_value=console),
            ):
                await example.main()
            self.assertEqual(
                [call.kwargs["model"] for call in create_provider.call_args_list],
                ["first", "second"],
            )
            self.assertEqual(
                json.loads((output / "experiment.json").read_text())["model"], ["first", "second"]
            )
            self.assertIn("Generated Policy", (output / "run.log").read_text())
            with (
                gym.make("CartPole-v1") as env,
                recorded_run(output, environment=env) as (run, rollouts),
                run.database() as db,
            ):
                self.assertEqual(len(db.exec(select(Evaluation)).all()), 2)
                self.assertTrue(db.exec(select(Generation)).one().complete)

    async def test_three_patch_operations_preserve_boundaries_and_crossover_ancestry(self):
        for mode in ("diff", "full", "cross"):
            with self.subTest(mode=mode):
                child = (
                    Mutation(
                        name="Child",
                        description="Improvement",
                        edits=[Edit(search="return 1", replacement="return 2")],
                    )
                    if mode == "diff"
                    else program(2)
                )
                provider = ScriptedProvider([program(0), program(1), child])
                agent = ShinkaEvolve(
                    "task",
                    provider,
                    config=Config(
                        islands=1,
                        meta_interval=0,
                        parent_selection="best",
                        patch_types=((mode, 1),),
                        migration_interval=0,
                    ),
                )
                parents = await agent.generate(n=2, concurrency=1)
                agent.update_scores({p.id: float(i) for i, p in enumerate(parents)})
                (policy,) = await agent.generate()
                self.assertIn("return 2", policy.source)
                row = agent.evaluations[-1]
                self.assertEqual(row.parents[0], parents[1].id)
                self.assertEqual(len(row.parents), 2 if mode == "cross" else 1)
                self.assertIn("Policy 0", provider.calls[-1])
                self.assertIn("preserved", provider.calls[-1])
                with self.assertRaises(ValueError):
                    agent.update_scores({policy.id: math.nan})
                self.assertIsNone(row.score)
                agent.update_scores({policy.id: 2})
                self.assertEqual(agent.best, policy)
                self.assertEqual(row.model_gain, "1")
                self.assertEqual(sum(agent.offspring.values()), 1)
                with self.assertRaises(KeyError):
                    agent.update_scores({policy.id: 2})
        for path in ROOT.glob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )

    async def test_parent_samplers_model_credit_pruning_and_migration(self):
        first, second = ScriptedProvider([program(0)]), ScriptedProvider([program(1), program(2)])
        agent = ShinkaEvolve(
            "task",
            first,
            ensemble=(first, second),
            seed=1,
            config=Config(
                islands=1,
                meta_interval=0,
                patch_types=(("full", 1),),
                archive_size=2,
                elite_ratio=0.5,
                migration_interval=0,
            ),
        )
        for score in (-1e308, 1e308, 0):
            (policy,) = await agent.generate()
            agent.update_scores({policy.id: score})
        self.assertEqual([row.model for row in agent.evaluations[:2]], [0, 1])
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertTrue(all(math.isfinite(weight) for weight in agent.model_weights()))
        self.assertEqual(len(agent.islands[0]), 2)
        population = list(agent._archive.values())
        for sampler in ("uniform", "best", "power", "weighted"):
            selected = ShinkaEvolve("task", first, config=Config(parent_selection=sampler))
            weights = selected.selection_weights(population)
            self.assertTrue(all(math.isfinite(weight) and weight >= 0 for weight in weights))
            self.assertGreater(sum(weights), 0)
            if sampler == "best":
                self.assertEqual(weights, [0, 1, 0])
            if sampler == "power":
                self.assertEqual(weights, [1 / 3, 1, 1 / 2])
        agent.config = Config(archive_size=3, migration_rate=1)
        agent.islands = [[population[1], population[2]], [population[0]]]
        agent.migrate()
        self.assertNotIn(population[1], agent.islands[1])
        self.assertIn(population[2], agent.islands[1])
        self.assertEqual(agent.events[-1]["migration"], [population[2].policy.id])

    async def test_optional_embedding_novelty_resamples_before_evaluation(self):
        embedded = []

        async def embed(code):
            embedded.append(code)
            return [1.0, 0.0]

        provider = ScriptedProvider([program(0), program(1), program(2)])
        judge = ScriptedProvider(
            [
                '{"novel": false, "reason": "Only a rename"}',
                '{"novel": true, "reason": "New algorithm"}',
            ]
        )
        agent = ShinkaEvolve(
            "task",
            provider,
            embed=embed,
            novelty_provider=judge,
            config=Config(islands=1, meta_interval=0, patch_types=(("full", 1),)),
        )
        (initial,) = await agent.generate()
        agent.update_scores({initial.id: 1})
        (child,) = await agent.generate()
        self.assertEqual(child.name, "Policy 2")
        self.assertEqual(agent.evaluations[-1].proposals, 2)
        self.assertEqual(agent.novelty_calls, 2)
        self.assertEqual(len(embedded), 3)
        self.assertTrue(all("from rsikit import Policy" not in code for code in embedded))
        self.assertIn("Only a rename", provider.calls[-1])
        self.assertEqual(agent.completed, 1)
        agent.update_scores({child.id: 2})
        self.assertEqual(agent.completed, 2)

    async def test_meta_guidance_retains_last_valid_scratchpad(self):
        provider = ScriptedProvider([program(0), program(1), program(2)])
        meta = ScriptedProvider(['{"recommendations": ["Reuse successful ideas"]}', "not JSON"])
        agent = ShinkaEvolve(
            "task",
            provider,
            meta_provider=meta,
            config=Config(islands=1, meta_interval=1, patch_types=(("full", 1),)),
        )
        for score in (1, 2, 3):
            (policy,) = await agent.generate()
            agent.update_scores({policy.id: score})
        self.assertEqual(agent.scratchpad, ["Reuse successful ideas"])
        self.assertEqual(agent.meta_calls, 2)
        self.assertIn("Reuse successful ideas", provider.calls[-1])
        self.assertIn("Policy 0", meta.calls[0])
        self.assertIn("error", agent.events[-1])

    async def test_bounded_repairs_failed_versions_and_islands_are_saved_as_typed_records(self):
        broken = program(0).model_copy(update={"implementation": SOURCE + "}"})
        provider = ScriptedProvider(
            [
                broken,
                broken,
                broken,
                broken,
                program(0),
                program(9),
                program(8),
                program(1),
                broken,
                broken,
            ]
        )
        agent = ShinkaEvolve(
            "task",
            provider,
            config=Config(
                islands=1,
                max_proposals=1,
                max_repairs=1,
                meta_interval=0,
                patch_types=(("full", 1),),
            ),
        )
        evaluation = FakeEvaluation()

        async def evaluate(implementation, environment, seed):
            if "return 9" in implementation or "return 8" in implementation:
                if seed == 1:
                    raise PolicyError("Action outside action_space")
                return trajectory(100.0, {})
            return trajectory(8.0 if "return 1" in implementation else 7.0, {})

        evaluation.evaluate.side_effect = evaluate
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as env:
            path = Path(directory) / "run"
            with recorded_run(
                name="shinka",
                path=path,
                environment=env,
                executor=fake_executor(evaluation=evaluation),
                console=Console(file=io.StringIO()),
            ) as (run, rollouts):
                await run_search(
                    agent,
                    run,
                    rollouts,
                    generations=3,
                    batch_size=2,
                    generation_concurrency=1,
                    seeds=(0, 1),
                )
            with recorded_run(path, environment=env) as (run, rollouts), run.database() as db:
                rows = db.exec(
                    select(Evaluation).order_by(Evaluation.attempt, Evaluation.revision)
                ).all()
                snapshots = db.exec(select(Generation).order_by(Generation.number)).all()
                self.assertEqual(
                    [row.status for row in rows],
                    [
                        "discarded",
                        "discarded",
                        "evaluated",
                        "failed",
                        "discarded",
                        "evaluated",
                        "discarded",
                    ],
                )
                self.assertEqual((rows[3].revision, rows[4].revision), (0, 1))
                self.assertNotEqual(rows[3].policy_id, rows[4].policy_id)
                self.assertTrue(all((snapshot.complete for snapshot in snapshots)))
                self.assertEqual(snapshots[0].islands, [[]])
                self.assertEqual(max((c["score"] for c in snapshots[-1].islands[0])), 8)
                self.assertEqual(snapshots[-1].seeds, [0, 1])
                self.assertEqual(len(run.policies()), 4)
        self.assertEqual(agent.completed, 6)
        self.assertEqual(sum(map(len, agent.model_gains)), 6)
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertEqual(agent._pending, {})
        self.assertIs(agent.records()[-1], agent.generations[-1])

    async def test_concurrent_repair_uses_its_own_failed_response(self):
        provider = ScriptedProvider([])
        second_started, repairing = asyncio.Event(), asyncio.Event()
        calls = 0

        async def respond(context, **kwargs):
            nonlocal calls
            calls += 1
            number = calls
            if number == 1:
                await second_started.wait()
                return "BROKEN_FIRST_RESPONSE", []
            if number == 2:
                second_started.set()
                await repairing.wait()
                return program(1).model_dump_json(), []
            self.assertIn("BROKEN_FIRST_RESPONSE", context)
            repairing.set()
            return program(0).model_dump_json(), []

        provider.acall = respond
        agent = ShinkaEvolve("task", provider)
        policies = await asyncio.wait_for(agent.generate(n=2, concurrency=2), 2)
        self.assertEqual([p.name for p in policies], ["Policy 0", "Policy 1"])
        self.assertEqual(agent.repair_calls, 1)
        with self.assertRaises(ValueError):
            await agent.generate(concurrency=0)

    async def test_concurrent_provider_failure_cancels_siblings_without_fitness_credit(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                provider = ScriptedProvider([])
                started, release = asyncio.Event(), asyncio.Event()
                active = calls = 0

                async def blocked(*args, **kwargs):
                    nonlocal active, calls
                    active += 1
                    calls += 1
                    number = calls
                    if calls == 2:
                        started.set()
                    try:
                        if number == 1:
                            await release.wait()
                            raise ProviderError("offline")
                        await asyncio.Event().wait()
                    finally:
                        active -= 1

                provider.acall = blocked
                agent = ShinkaEvolve("task", provider)
                task = asyncio.create_task(agent.generate(n=5, concurrency=2))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    if cancelled:
                        task.cancel()
                    else:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError if cancelled else ProviderError):
                        await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(active, 0)
                self.assertEqual(agent.model_gains, [[]])
                self.assertEqual(agent._pending, {})


if __name__ == "__main__":
    unittest.main()
