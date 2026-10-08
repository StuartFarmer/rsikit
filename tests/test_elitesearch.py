"""Elite retention, reproduction, repair, and complete-round isolated evaluations."""

import asyncio
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts
from slick.providers import ProviderError
from sqlmodel import select

from research.elitesearch import Config, EliteSearch, Generation, Organism
from rsikit import PolicyDefinition
from tests.helpers import episodes, fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_run import FakeEvaluation


def program(value):
    return json.dumps(
        dict(
            name=f"Policy {value}",
            description=f"Idea {value}",
            implementation="from rsikit import Policy\nclass Solution(Policy):\n"
            f"    async def act(self, observation):\n        return {value}\n",
        )
    )


class EliteSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(
            prompts,
            "TEMPLATE_ROOT",
            Path(__file__).resolve().parents[1] / "research/elitesearch/prompts",
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_fixed_population_snapshot_three_operators_and_top_elites(self):
        provider = ScriptedProvider([])
        next_value = 0

        async def respond(context, **kwargs):
            nonlocal next_value
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

        async def evaluate(policies):
            return {p.id: episodes({0: int(p.name.split()[-1])}) for p in policies}

        agent = EliteSearch(
            "Score policies",
            provider,
            evaluate,
            config=Config(elite_size=2, population_size=5, generations=2),
        )
        with patch.object(provider, "acall", side_effect=respond):
            await agent.run()
        self.assertEqual([row.score for row in agent.elites], [9, 8])
        self.assertEqual(len(agent.organisms), 10)
        founders, children = agent.organisms[:5], agent.organisms[5:]
        self.assertTrue(all(row.kind == "new" for row in founders))
        self.assertEqual(Counter(row.kind for row in children), {"new": 1, "edit": 2, "remix": 2})
        old_elites = set(agent.generations[0].elite_ids)
        for row in children:
            self.assertTrue(set(row.parent_ids) <= old_elites)
            self.assertEqual(len(row.parent_ids), {"new": 0, "edit": 1, "remix": 2}[row.kind])
        self.assertEqual(agent.reason, "completed")

    async def test_ties_and_worse_candidates_keep_incumbents(self):
        provider = ScriptedProvider([program(i) for i in range(6)])
        scores = iter([10, 9, 8, 9, -10, 8])

        async def evaluate(policies):
            return {p.id: episodes({0: next(scores)}) for p in policies}

        agent = EliteSearch(
            "Score",
            provider,
            evaluate,
            config=Config(
                elite_size=2,
                population_size=3,
                generations=2,
                new_fraction=1,
                remix_fraction=0,
                generation_concurrency=1,
            ),
        )
        await agent.run()
        self.assertEqual([row.id for row in agent.elites], [1, 2])
        self.assertEqual(agent.generations[0].elite_ids, agent.generations[1].elite_ids)

    async def test_repair_budget_covers_invalid_output_and_execution(self):
        provider = ScriptedProvider(["bad JSON", program(0), program(1)])
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(p.name for p in policies)
            return {
                p.id: episodes({}, failure="invalid action")
                if p.name == "Policy 0"
                else episodes({0: 12})
                for p in policies
            }

        agent = EliteSearch(
            "Score",
            provider,
            evaluate,
            config=Config(
                population_size=1,
                generations=1,
                max_repairs=2,
            ),
        )
        await agent.run()
        self.assertEqual(evaluated, ["Policy 0", "Policy 1"])
        self.assertEqual(agent.elites[0].score, 12)
        self.assertEqual(agent.organisms[0].repairs, 2)
        self.assertEqual(len(agent.organisms[0].revisions), 2)
        self.assertIn("invalid action", provider.calls[-1])

    async def test_edits_can_change_source_outside_marker_comments(self):
        initial = json.loads(program(0))
        initial["implementation"] = (
            "# outside\n# EVOLVE-BLOCK-START\n" + initial["implementation"] + "# EVOLVE-BLOCK-END\n"
        )
        provider = ScriptedProvider(
            [
                json.dumps(initial),
                json.dumps(
                    dict(
                        name="Changed",
                        description="Edit outside comments.",
                        edits=[
                            dict(search="# outside", replacement="OFFSET = 1"),
                            dict(search="return 0", replacement="return OFFSET"),
                        ],
                    )
                ),
            ]
        )
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(policies)
            return {p.id: episodes({0: len(evaluated)}) for p in policies}

        agent = EliteSearch(
            "Score",
            provider,
            evaluate,
            config=Config(
                population_size=1,
                generations=2,
                new_fraction=0,
                remix_fraction=0,
                max_repairs=0,
            ),
        )
        await agent.run()
        self.assertEqual(len(evaluated), 2)
        self.assertIn("OFFSET = 1", evaluated[-1].source)
        self.assertEqual(agent.elites[0].name, "Changed")

    async def test_failed_edit_retains_organism_metadata(self):
        response = dict(
            name="Failed edit",
            description="A proposed change.",
            edits=[dict(search="missing source", replacement="replacement")],
        )
        provider = ScriptedProvider([program(0), json.dumps(response)])

        async def evaluate(policies):
            return {p.id: episodes({0: 1}) for p in policies}

        agent = EliteSearch(
            "Score",
            provider,
            evaluate,
            config=Config(
                population_size=1,
                generations=2,
                new_fraction=0,
                remix_fraction=0,
                max_repairs=0,
            ),
        )
        await agent.run()
        failed = agent.organisms[-1]
        self.assertEqual(
            (failed.name, failed.description), (response["name"], response["description"])
        )
        self.assertEqual(failed.status, "discarded")
        self.assertIsNone(failed.policy_id)
        self.assertEqual(json.loads(failed.calls[-1]["raw"]), response)

    async def test_evaluation_failure_retains_generated_round_without_promotion(self):
        async def evaluate(policies):
            self.assertEqual(len(policies), 2)
            raise RuntimeError("worker offline")

        agent = EliteSearch(
            "Score",
            ScriptedProvider([program(0), program(1)]),
            evaluate,
            config=Config(population_size=2, generations=1),
        )
        with self.assertRaisesRegex(RuntimeError, "worker offline"):
            await agent.run()
        self.assertEqual(agent.reason, "error")
        self.assertEqual(agent.elites, [])
        self.assertEqual(len(agent._round), 2)

    async def test_invalid_scores_never_enter_leaderboard(self):
        for scores in ({0: float("nan")}, {0: float("inf")}):
            with self.subTest(scores=scores):

                async def evaluate(policies):
                    return {p.id: episodes(scores) for p in policies}

                agent = EliteSearch(
                    "Score",
                    ScriptedProvider([program(0)]),
                    evaluate,
                    config=Config(population_size=1, generations=1),
                )
                with self.assertRaises(ValueError):
                    await agent.run()
                self.assertEqual(agent.elites, [])
                self.assertEqual(agent.reason, "error")

    async def test_screening_rejection_does_not_promote_or_repair(self):
        async def evaluate(policies):
            return {
                p.id: episodes(scores={99: 100}, accepted=False, feedback="screened")
                if p.name == "Policy 0"
                else episodes(scores={0: 3})
                for p in policies
            }

        provider = ScriptedProvider([program(0), program(1)])
        agent = EliteSearch(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        await agent.run()
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertEqual(agent.organisms[0].status, "discarded")
        self.assertEqual(agent.organisms[0].error, "Evaluation rejected")
        self.assertEqual(agent.organisms[0].repairs, 0)
        self.assertEqual(len(provider.calls), 2)

    async def test_whole_generation_is_proposed_before_evaluation(self):
        provider = ScriptedProvider([program(0), program(1)])

        async def evaluate(policies):
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual([p.name for p in policies], ["Policy 0", "Policy 1"])
            return {p.id: episodes({0: 7}) for p in policies}

        agent = EliteSearch(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        await agent.run()
        self.assertTrue(all(row.status == "evaluated" for row in agent.organisms))

    async def test_duplicate_and_failed_populations_preserve_existing_elites(self):
        async def evaluate(policies):
            return {
                p.id: episodes({0: 7})
                if p.name == "Policy 0"
                else episodes({}, failure="broken policy")
                for p in policies
            }

        agent = EliteSearch(
            "Score",
            ScriptedProvider([program(0), program(0), program(1)]),
            evaluate,
            config=Config(
                population_size=1, generations=3, new_fraction=1, remix_fraction=0, max_repairs=0
            ),
        )
        await agent.run()
        self.assertEqual([row.id for row in agent.elites], [1])
        self.assertEqual(
            [row.status for row in agent.organisms], ["evaluated", "discarded", "discarded"]
        )
        self.assertEqual([g.elite_ids for g in agent.generations], [[1], [1], [1]])
        self.assertEqual(agent.reason, "completed")

    async def test_provider_failure_cancels_sibling_generation_before_evaluation(self):
        provider = ScriptedProvider([])
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = 0

        async def respond(context, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                await started.wait()
                raise ProviderError("offline")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def evaluate(policies):
            self.fail("evaluation started before generation finished")

        agent = EliteSearch(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        with patch.object(provider, "acall", side_effect=respond):
            with self.assertRaisesRegex(ProviderError, "offline"):
                await asyncio.wait_for(agent.run(), 2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(agent.reason, "error")

    async def test_cli_persists_elites_ancestry_logs_and_heldout(self):
        from examples import elitesearch as example

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            terminal = io.StringIO()
            evaluation = FakeEvaluation()
            with (
                patch(
                    "sys.argv",
                    [
                        "elitesearch",
                        "--env",
                        "CartPole-v1",
                        "--elites",
                        "1",
                        "--population",
                        "2",
                        "--generations",
                        "2",
                        "--new-fraction",
                        "1",
                        "--remix-fraction",
                        "0",
                        "--seeds",
                        "0",
                        "1",
                        "--heldout-seeds",
                        "99",
                        "--output",
                        str(output),
                    ],
                ),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    example,
                    "OpenRouterAPI",
                    return_value=ScriptedProvider([program(i) for i in range(4)]),
                ),
                patch.object(
                    example, "Executor", return_value=fake_executor(evaluation=evaluation)
                ),
                patch(
                    "rsikit.progress.controller.Console",
                    return_value=Console(file=terminal, width=140),
                ),
            ):
                await example.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["generations"], 2)
            self.assertEqual(summary["organisms"], 4)
            self.assertEqual(summary["elite_ids"], [1])
            self.assertEqual(summary["heldout"]["scores"], {"99": 7})
            self.assertEqual(
                PolicyDefinition.from_file(output / "best.py").source,
                json.loads(program(0))["implementation"],
            )
            self.assertIn("Elite leaderboard", terminal.getvalue())
            self.assertIn("promotions", (output / "run.log").read_text())
            with (
                gym.make("CartPole-v1") as env,
                recorded_run(output, environment=env) as (run, rollouts),
            ):
                with run.database() as db:
                    generations = db.exec(select(Generation).order_by(Generation.number)).all()
                    rows = db.exec(select(Organism).order_by(Organism.id)).all()
                    self.assertEqual([g.elite_ids for g in generations], [[1], [1]])
                    self.assertEqual(len(rows), 4)
                    self.assertTrue(all(row.seed_scores == {"0": 7, "1": 7} for row in rows))

    async def test_cli_generation_timeout_controls_request_and_search_deadline(self):
        from examples import elitesearch as example

        provider = ScriptedProvider([])

        async def stalled_response(*args, **kwargs):
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(example, "OpenRouterAPI", return_value=provider) as constructor,
                patch.object(provider, "acall", side_effect=stalled_response),
                patch.object(
                    example, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                ),
                patch(
                    "rsikit.progress.controller.Console", return_value=Console(file=io.StringIO())
                ),
            ):
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(
                        example.main(
                            [
                                "--env",
                                "Bitcoin",
                                "--population",
                                "1",
                                "--generations",
                                "1",
                                "--generation-timeout",
                                "0.01",
                                "--output",
                                str(output),
                            ]
                        ),
                        2,
                    )
            self.assertEqual(constructor.call_args.kwargs["timeout"], 0.01)
            # If the search ignored the option, the outer guard would cancel it instead.
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["reason"], "error")
            self.assertEqual(summary["generations"], 0)
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["generation_timeout"], 0.01)


if __name__ == "__main__":
    unittest.main()
