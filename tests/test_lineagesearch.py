"""Measured lineage selection, bounded exploration, and durable experiment evidence."""

import asyncio
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts
from slick.providers import ProviderError
from sqlmodel import select

from research.lineagesearch import Config, Family, LineageSearch, Study, Trial
from rsikit.evaluation import PolicyError
from tests.helpers import episodes, fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation

ROOT = Path(__file__).resolve().parents[1] / "research/lineagesearch" / "prompts"


def families(n=1):
    return json.dumps(
        {"families": [{"name": f"Family {i}", "mechanism": f"Mechanism {i}"} for i in range(n)]}
    )


def experiments(*values):
    return json.dumps(
        {
            "approaches": [
                {
                    "hypothesis": f"Hypothesis {v}",
                    "mechanism": f"Mechanism {v}",
                    "change": f"Change {v}",
                    "test": f"Test {v}",
                }
                for v in values
            ]
        }
    )


def program(value, *, name=None):
    return json.dumps(
        {
            "name": name or f"Policy {value}",
            "description": "Experimental policy",
            "implementation": "from rsikit import Policy\nclass Solution(Policy):\n"
            f"    async def act(self, observation):\n        return {value}\n",
        }
    )


def sequence(*values):
    return [item for value in values for item in (experiments(value), program(value))]


class LineageTests(unittest.IsolatedAsyncioTestCase):
    async def test_cull_contracts_then_expands_only_surviving_parents(self):
        agent = self.agent(
            [
                families(2),
                experiments(*range(10)),
                experiments(*range(10, 20)),
                *(program(i) for i in range(20)),
                experiments(20, 21),
                program(20),
                program(21),
            ],
            {f"Policy {i}": (i + 10 if i < 10 else i - 10 if i < 20 else i) for i in range(22)},
            max_attempts=22,
        )
        agent.config = replace(
            agent.config,
            families=2,
            initial_per_family=10,
            batch_size=2,
            exploration=1,
        )
        survivors = []

        def checkpoint(current):
            if current.families and current.families[0].frontier:
                survivors.append(list(current.families[0].frontier))

        agent.on_checkpoint = checkpoint
        await agent.run()
        self.assertEqual(survivors[0], [10, 9])
        self.assertTrue(all(row.parent_id in {9, 10} for row in agent.trials[20:]))
        self.assertEqual(agent.families[1].status, "culled")
        self.assertEqual(agent.families[1].frontier, [])
        self.assertTrue(all(row.family_id == 1 for row in agent.trials[20:]))
        self.assertEqual(agent.families[0].frontier, [22])
        self.assertEqual(len(agent.trials), 22)
        self.assertTrue(all(row.score is not None for row in agent.trials))

    def test_cull_rounding_zero_and_no_resurrection(self):
        for percent, expected in ((90, [25, 24, 23]), (0, list(range(25, 0, -1))), (99.9, [25])):
            with self.subTest(percent=percent):
                agent = self.agent([], {}, cull_percent=percent)
                family = Family(id=1, name="Family", mechanism="Mechanism")
                agent.families = [family]
                agent.trials = [
                    Trial(
                        id=i,
                        family_id=1,
                        batch=1,
                        kind="founder",
                        status="evaluated",
                        score=i,
                        seed_scores={"0": i},
                    )
                    for i in range(1, 26)
                ]
                agent._update(family, agent.trials[:], True)
                agent._cull(agent.trials[:])
                self.assertEqual(family.frontier, expected)
                old_survivors = set(expected)
                child = Trial(
                    id=26,
                    family_id=1,
                    batch=2,
                    kind="refine",
                    status="evaluated",
                    score=-1,
                    seed_scores={"0": -1},
                )
                agent.trials.append(child)
                agent._update(family, [child], True)
                agent._cull([child])
                self.assertTrue(set(family.frontier) <= old_survivors | {26})
                agent.config = replace(agent.config, exploration=1)
                self.assertTrue(
                    all(agent._select_parent(family).id in family.frontier for _ in range(100))
                )

    def test_invalid_cull_percent_is_rejected(self):
        for value in (-1, 100, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(cull_percent=value)

    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    def agent(self, responses, scores, **config):
        async def evaluate(policies):
            results = {}
            for policy in policies:
                value = scores[policy.name]
                if isinstance(value, Exception):
                    raise value
                results[policy.id] = episodes(
                    value if isinstance(value, dict) else {0: value, 1: value, 2: value}
                )
            return results

        return LineageSearch(
            "Improve measured reward",
            ScriptedProvider(responses),
            evaluate,
            config=Config(
                families=1,
                initial_per_family=1,
                batch_size=1,
                patience=2,
                bonus_batches=0,
                exploration=0,
                decomposition_k=config.pop("decomposition_k", 1),
                generation_concurrency=config.pop("generation_concurrency", 1),
                **config,
            ),
        )

    async def test_weaker_family_breaks_through_then_all_lineages_stagnate(self):
        values = [0, 1, 2, 3, 4, 5, 6]
        scores = dict(zip((f"Policy {v}" for v in values), [10, 1, 10, 20, 10, 20, 20]))
        agent = self.agent(
            [
                families(2),
                experiments(0),
                experiments(1),
                program(0),
                program(1),
                *sequence(2, 3, 4, 5, 6),
            ],
            scores,
        )
        agent.config = Config(
            decomposition_k=1,
            families=2,
            generation_concurrency=1,
            cull_percent=0,
            initial_per_family=1,
            batch_size=1,
            patience=2,
            bonus_batches=0,
            exploration=0,
        )
        result = await agent.run()
        self.assertEqual(result.reason, "completed")
        self.assertEqual(agent.best.name, "Policy 3")
        self.assertEqual([f.status for f in agent.families], ["stagnated", "stagnated"])
        self.assertEqual(agent.trials[3].parent_id, 2)
        self.assertTrue(
            all(
                any(t.kind == "pivot" and t.family_id == f.id for t in agent.trials)
                for f in agent.families
            )
        )
        self.assertEqual(result.attempts, 7)

    async def test_partial_budget_is_not_stagnation(self):
        agent = self.agent(
            [families(), *sequence(0, 1)], {"Policy 0": 0, "Policy 1": 0}, max_attempts=2
        )
        agent.config = Config(
            decomposition_k=1,
            families=1,
            initial_per_family=1,
            batch_size=3,
            patience=1,
            bonus_batches=0,
            max_attempts=2,
        )
        result = await agent.run()
        self.assertEqual(result.reason, "budget_exhausted")
        self.assertEqual(agent.families[0].status, "active")
        self.assertEqual(agent.families[0].stale_batches, 0)

    async def test_all_founders_are_planned_before_execution_even_with_small_budget(self):
        agent = self.agent(
            [families(2), experiments(0, 1), experiments(2, 3), program(0)],
            {"Policy 0": 7},
        )
        agent.config = Config(families=2, initial_per_family=2, max_attempts=1, decomposition_k=1)
        phases = []

        def checkpoint(current):
            phases.append(current.study.phase)
            if current.study.phase == "search":
                self.assertEqual([len(f.initial_approaches) for f in current.families], [2, 2])

        agent.on_checkpoint = checkpoint
        await agent.run()
        self.assertEqual(agent.study.attempts, 1)
        self.assertEqual(len(agent.trials), 1)
        self.assertEqual(agent.best.name, "Policy 0")
        self.assertIn("planning_approaches", phases)
        self.assertEqual(
            [c["operation"] for c in agent.study.calls], ["discover", "found", "found", "implement"]
        )

    async def test_decomposition_elects_by_lead_not_total_votes(self):
        partitions = [families().replace("Family 0", f"Partition {i}") for i in range(3)]

        def vote(n):
            return json.dumps({"candidate": n, "reason": "Broad, distinct coverage"})

        agent = self.agent(
            [
                "bad partition",
                *partitions,
                vote(1),
                vote(2),
                vote(2),
                vote(2),
                experiments(0),
                experiments(1),
                experiments(2),
                json.dumps({"candidate": True, "reason": "Invalid boolean"}),
                vote(2),
                vote(2),
                program(1),
            ],
            {"Policy 1": 7},
            max_attempts=1,
            decomposition_k=2,
        )
        await agent.run()
        self.assertEqual(agent.families[0].name, "Partition 1")
        self.assertEqual(agent.best.name, "Policy 1")
        root, child = agent.study.decompositions
        self.assertEqual(root["votes"], [1, 3, 0])
        self.assertEqual(root["outcome"], "elected")
        self.assertEqual(child["winner"], 2)
        self.assertEqual(child["coordinates"], {"mechanism family": "Partition 1"})
        self.assertIn("LEVEL CONTRACT", agent.provider.calls[4])
        self.assertIn("scipy.linalg.solve_discrete_are", agent.provider.calls[0])
        self.assertIn("CPU-only PyTorch", agent.provider.calls[-1])

    async def test_vote_cap_falls_back_and_records_the_actual_tally(self):
        for ballots, expected in (([2, 1], 2), ([0, 4], 1)):
            agent = self.agent(
                [families().replace("Family 0", f"Partition {i}") for i in range(3)]
                + [json.dumps({"candidate": n, "reason": "Review"}) for n in ballots],
                {},
                decomposition_k=2,
                decomposition_max_votes=2,
            )
            await agent._discover()
            decision = agent.study.decompositions[0]
            self.assertEqual(decision["winner"], expected)
            self.assertEqual(decision["outcome"], "fallback")
            self.assertEqual(agent.families[0].name, f"Partition {expected - 1}")

    async def test_partition_repair_context_is_local_and_latest_only(self):
        vote = json.dumps({"candidate": 1, "reason": "Coverage"})
        agent = self.agent(
            ["BROKEN_A", "BROKEN_B", families(), families(), families(), vote, vote],
            {},
            decomposition_k=2,
        )
        await agent._discover()
        prompts_seen = agent.provider.calls
        self.assertIn("BROKEN_A", prompts_seen[1])
        self.assertIn("BROKEN_B", prompts_seen[2])
        self.assertNotIn("BROKEN_A", prompts_seen[2])
        for text in prompts_seen[3:]:
            self.assertNotIn("BROKEN_A", text)
            self.assertNotIn("BROKEN_B", text)
        self.assertEqual(agent.study.calls[0]["raw"], "BROKEN_A")

    async def test_decomposition_parallelism_has_one_global_limit(self):
        agent = self.agent([], {}, decomposition_k=2, generation_concurrency=2)
        agent.families = [
            Family(id=i, name=f"Family {i}", mechanism=f"Mechanism {i}") for i in (1, 2)
        ]
        active = peak = 0
        active_families = {1: 0, 2: 0}
        overlapped_families = False
        found_count = vote_count = 0

        async def respond(context, **kwargs):
            nonlocal active, peak, overlapped_families, found_count, vote_count
            family = (
                json.loads(context.split("<evidence>")[1].split("</evidence>")[0])["family"]["id"]
                if "<evidence>" in context
                else None
            )
            active += 1
            peak = max(peak, active)
            if family is not None:
                active_families[family] += 1
                overlapped_families |= all(active_families.values())
                found_count += 1
            else:
                vote_count += 1
            try:
                await asyncio.sleep(0.01)
                return (
                    experiments(0)
                    if family is not None
                    else json.dumps({"candidate": 1, "reason": "Coverage"})
                ), []
            finally:
                active -= 1
                if family is not None:
                    active_families[family] -= 1

        with patch.object(agent.provider, "acall", side_effect=respond):
            await agent._plan_founders()
        self.assertEqual(peak, 2)
        self.assertTrue(overlapped_families)
        self.assertEqual((found_count, vote_count), (6, 4))
        self.assertEqual([len(f.initial_approaches) for f in agent.families], [1, 1])
        self.assertEqual(agent.study.phase, "search")
        self.assertEqual(agent.study.attempts, 0)

    async def test_parallel_decomposition_cancels_siblings_on_provider_failure(self):
        agent = self.agent([], {}, decomposition_k=2, generation_concurrency=2)
        both_started = asyncio.Event()
        started = active = 0

        async def respond(context, **kwargs):
            nonlocal started, active
            started += 1
            index = started
            active += 1
            if started == 2:
                both_started.set()
            try:
                await asyncio.wait_for(both_started.wait(), 0.5)
                if index == 1:
                    raise ProviderError("offline")
                await asyncio.Event().wait()
            finally:
                active -= 1

        with patch.object(agent.provider, "acall", side_effect=respond):
            with self.assertRaisesRegex(ProviderError, "offline"):
                await agent.run()
        self.assertEqual(active, 0)
        self.assertEqual(agent.study.reason, "error")

    async def test_complete_sweep_precedes_evaluation_without_overbooking_trials(self):
        agent = self.agent(
            [families(3), experiments(0), experiments(1), experiments(2), program(0), program(1)],
            {},
            max_attempts=2,
            generation_concurrency=2,
        )
        from dataclasses import replace

        agent.config = replace(agent.config, families=3)
        generated = []
        evaluated = []
        original_call = agent.provider.acall

        async def generate(context, **kwargs):
            result = await original_call(context, **kwargs)
            if result[0] in (program(0), program(1)):
                generated.append(result[0])
            return result

        async def evaluate(policies):
            self.assertEqual(len(generated), 2)
            evaluated.append([p.name for p in policies])
            return {p.id: episodes({0: 7}) for p in policies}

        agent.evaluate = evaluate
        with patch.object(agent.provider, "acall", side_effect=generate):
            await agent.run()
        self.assertEqual(evaluated, [["Policy 0", "Policy 1"]])
        self.assertEqual(agent.study.attempts, 2)
        self.assertEqual([t.id for t in agent.trials], [1, 2])
        self.assertEqual([f.batches for f in agent.families], [1, 1, 0])
        self.assertTrue(all(t.status == "evaluated" for t in agent.trials))

    async def test_evaluator_failure_preserves_complete_pending_sweep(self):
        agent = self.agent(
            [families(2), experiments(0), experiments(1), program(0), program(1)],
            {},
            max_attempts=2,
            generation_concurrency=2,
        )
        from dataclasses import replace

        agent.config = replace(agent.config, families=2)

        async def evaluate(policies):
            self.assertEqual([p.name for p in policies], ["Policy 0", "Policy 1"])
            raise RuntimeError("worker offline")

        agent.evaluate = evaluate
        with self.assertRaisesRegex(RuntimeError, "worker offline"):
            await agent.run()
        self.assertEqual(agent.study.reason, "error")
        self.assertTrue(all(t.status == "evaluating" for t in agent.trials))
        self.assertEqual([f.batches for f in agent.families], [0, 0])

    async def test_noise_does_not_promote_and_small_gains_accumulate(self):
        agent = self.agent(
            [families(), *sequence(0, 1, 2, 3)],
            {
                "Policy 0": 0,
                "Policy 1": {0: -10, 1: 1, 2: 12},
                "Policy 2": 0.6,
                "Policy 3": 1.2,
            },
            min_delta=1,
            max_attempts=4,
        )
        agent.config = Config(
            decomposition_k=1,
            families=1,
            initial_per_family=1,
            batch_size=1,
            patience=3,
            bonus_batches=0,
            min_delta=1,
            max_attempts=4,
        )
        incumbents = []

        def checkpoint(current):
            if current.families:
                incumbents.append((current.study.attempts, current.families[0].best_id))

        agent.on_checkpoint = checkpoint
        await agent.run()
        family = agent.families[0]
        self.assertEqual(agent.best.name, "Policy 3")
        self.assertEqual(family.checkpoint_id, 4)
        self.assertEqual(family.stale_batches, 0)
        self.assertIn((2, 1), incumbents)
        self.assertNotIn((2, 2), incumbents)

    async def test_ast_duplicates_and_malformed_outputs_exhaust_generation(self):
        responses = [
            families(),
            *sequence(0),
            experiments(1),
            program(0, name="Renamed"),
            experiments(2),
            "not json",
        ]
        agent = self.agent(responses, {"Policy 0": 5}, max_repairs=0)
        result = await agent.run()
        self.assertEqual(result.reason, "completed")
        self.assertEqual(agent.families[0].status, "generation_exhausted")
        self.assertEqual(agent.families[0].stale_batches, 0)
        self.assertEqual([t.status for t in agent.trials], ["evaluated", "rejected", "rejected"])
        self.assertIn("not json", [call.get("raw") for call in result.calls])

    async def test_infrastructure_failure_preserves_unfinished_trial(self):
        for failure in (ProviderError("offline"), RuntimeError("worker unavailable")):
            responses = [
                families(),
                experiments(0),
                failure if isinstance(failure, ProviderError) else program(0),
            ]
            agent = self.agent(responses, {"Policy 0": failure})
            with self.assertRaises(type(failure)):
                await agent.run()
            self.assertEqual(agent.study.reason, "error")
            self.assertEqual(agent.families[0].status, "active")
            self.assertEqual(agent.families[0].stale_batches, 0)
            self.assertIsNone(agent.trials[0].score)

    async def test_nonfinite_and_changed_seed_panel_fail_before_selection(self):
        for bad in ({0: float("nan"), 1: 1, 2: 1}, {99: 10}):
            agent = self.agent([families(), *sequence(0, 1)], {"Policy 0": 1, "Policy 1": bad})
            with self.assertRaises(ValueError):
                await agent.run()
            self.assertEqual(agent.best.name, "Policy 0")
            self.assertEqual(agent.families[0].stale_batches, 0)

    async def test_screening_rejection_does_not_promote_or_repair(self):

        agent = self.agent([families(), *sequence(0)], {}, max_attempts=1)

        async def evaluate(policies):
            return {
                p.id: episodes(scores={0: 100}, accepted=False, feedback="screened")
                for p in policies
            }

        agent.evaluate = evaluate
        await agent.run()
        self.assertIsNone(agent.best)
        self.assertEqual(agent.trials[0].status, "rejected")
        self.assertEqual(agent.trials[0].feedback, "")
        self.assertEqual(agent.trials[0].repairs, 0)
        self.assertIsNone(agent.trials[0].score)

    async def test_persisted_lineage_has_ancestry_measurements_and_completion(self):
        agent = self.agent(
            [families(), *sequence(0, 1, 2)], {"Policy 0": 1, "Policy 1": 1, "Policy 2": 1}
        )
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as env:
            async with recorded_run(
                name="lineage-test", environment=env, path=Path(directory) / "run"
            ) as (run, rollouts):
                agent.on_checkpoint = lambda current: run.save(*current.records())
                await agent.run()
                with run.database() as db:
                    self.assertEqual(db.exec(select(Study)).one().reason, "completed")
                    self.assertEqual(db.exec(select(Family)).one().status, "stagnated")
                    rows = db.exec(select(Trial).order_by(Trial.id)).all()
                    self.assertEqual(rows[-1].parent_id, 1)
                    self.assertEqual(rows[-1].seed_scores, {"0": 1, "1": 1, "2": 1})

    async def test_discovery_and_planning_failures_are_bounded(self):
        agent = self.agent(["bad", "bad", "bad"], {})
        result = await agent.run()
        self.assertEqual(result.reason, "generation_exhausted")
        self.assertEqual(result.attempts, 0)
        agent = self.agent([families(), "bad", "bad"], {}, max_repairs=0)
        result = await agent.run()
        self.assertEqual(result.attempts, 0)
        self.assertEqual(agent.families[0].status, "generation_exhausted")

    async def test_repairs_share_a_budget_across_generation_and_execution(self):
        agent = self.agent(
            [families(), experiments(0), "broken JSON", program(0), program(1)],
            {},
            max_attempts=1,
            max_repairs=2,
        )
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(p.name for p in policies)
            return {
                p.id: (
                    episodes({}, failure="Action outside action_space")
                    if p.name == "Policy 0"
                    else episodes({0: 5})
                )
                for p in policies
            }

        agent.evaluate = evaluate
        await agent.run()
        row = agent.trials[0]
        self.assertEqual(evaluated, ["Policy 0", "Policy 1"])
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertEqual((agent.study.attempts, row.repairs, row.status), (1, 2, "evaluated"))
        self.assertEqual([r["status"] for r in row.revisions], ["rejected", "execution_failed"])
        self.assertIn("broken JSON", agent.provider.calls[-2])
        self.assertIn("Action outside action_space", agent.provider.calls[-1])

    async def test_incompatible_constructor_repairs_before_evaluation(self):
        broken = json.loads(program(0))
        broken["implementation"] = broken["implementation"].replace(
            "    async def act",
            "    def __init__(self, instructions=None):\n"
            "        super().__init__(instructions=instructions)\n    async def act",
        )
        agent = self.agent(
            [families(), experiments(0), json.dumps(broken), program(1)],
            {"Policy 1": 7},
            max_attempts=1,
            max_repairs=1,
        )
        await agent.run()
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertEqual(agent.trials[0].repairs, 1)
        self.assertEqual(agent.trials[0].revisions[0]["status"], "rejected")

    async def test_sandbox_dependency_error_drives_repair_and_reevaluation(self):
        from examples.lineagesearch import measure

        broken = json.loads(program(0))
        broken["implementation"] = (
            "from scipy.linalg import solve_discrete_are\n" + broken["implementation"]
        )
        agent = self.agent(
            [families(), experiments(0), json.dumps(broken), program(1)],
            {},
            max_attempts=1,
            max_repairs=1,
        )
        evaluation = FakeEvaluation()

        async def execute(implementation, environment, seed):
            if "scipy" in implementation:
                raise PolicyError("ModuleNotFoundError: No module named 'scipy'")
            return trajectory(7.0, {})

        evaluation.evaluate.side_effect = execute
        with tempfile.TemporaryDirectory() as directory, gym.make("CartPole-v1") as env:
            with recorded_run(
                name="dependency-repair",
                environment=env,
                executor=fake_executor(evaluation=evaluation),
                path=Path(directory) / "run",
            ) as (run, rollouts):

                async def evaluate(policies):
                    return await measure(rollouts, policies, [0, 1])

                agent.evaluate = evaluate
                await agent.run()
                self.assertEqual(len(run.policies()), 2)
                self.assertEqual(run.scores(agent.best), {0: 7.0, 1: 7.0})
        self.assertEqual(agent.study.attempts, 1)
        self.assertEqual(agent.trials[0].repairs, 1)
        self.assertEqual(agent.trials[0].revisions[0]["status"], "execution_failed")
        self.assertIn("ModuleNotFoundError", agent.provider.calls[-1])

    async def test_exhausted_repairs_do_not_reevaluate_successful_siblings(self):
        agent = self.agent(
            [families(), experiments(0, 1), program(0), program(1), program(2)],
            {},
            max_attempts=2,
            max_repairs=1,
        )
        agent.config = Config(
            decomposition_k=1,
            families=1,
            initial_per_family=2,
            max_attempts=2,
            max_repairs=1,
            generation_concurrency=1,
        )
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(p.name for p in policies)
            return {
                p.id: (
                    episodes({0: 9})
                    if p.name == "Policy 0"
                    else episodes({}, failure="Invalid action")
                )
                for p in policies
            }

        agent.evaluate = evaluate
        await agent.run()
        self.assertEqual(evaluated, ["Policy 0", "Policy 1", "Policy 2"])
        self.assertEqual(agent.best.name, "Policy 0")
        self.assertEqual(agent.trials[1].repairs, 1)
        self.assertEqual(agent.trials[1].status, "execution_failed")

    async def test_invalid_decomposition_is_repaired_before_consuming_family_patience(self):
        agent = self.agent(
            [families(), "bad partition", experiments(0), program(0)],
            {"Policy 0": 1},
            max_attempts=1,
            max_repairs=1,
        )
        await agent.run()
        self.assertEqual(agent.best.name, "Policy 0")
        self.assertEqual(agent.study.attempts, 1)
        self.assertEqual(agent.families[0].failed_batches, 0)
        self.assertIn("bad partition", agent.provider.calls[2])

    async def test_invalid_runtime_repair_does_not_keep_previous_policy_identity(self):
        broken = json.loads(program(1))
        broken["implementation"] = "def broken(:"
        agent = self.agent(
            [families(), *sequence(0), json.dumps(broken)], {}, max_attempts=1, max_repairs=1
        )

        async def evaluate(policies):
            return {p.id: episodes({}, failure="Invalid action") for p in policies}

        agent.evaluate = evaluate
        await agent.run()
        row = agent.trials[0]
        self.assertEqual(row.status, "rejected")
        self.assertIsNone(row.policy_id)
        self.assertIsNotNone(row.revisions[0]["policy_id"])
        self.assertEqual(row.implementation, "def broken(:")

    async def test_repair_preserves_original_parent_boundaries_and_failed_source(self):
        founder = json.loads(program(0))
        founder["implementation"] = (
            founder["implementation"].replace(
                "        return 0",
                "        # EVOLVE-BLOCK-START\n        return 0\n        # EVOLVE-BLOCK-END",
            )
            + "\nUNCHANGED = 1\n"
        )
        invalid = {
            **founder,
            "name": "Invalid",
            "implementation": founder["implementation"]
            .replace("return 0", "return 1")
            .replace("UNCHANGED = 1", "UNCHANGED = 2"),
        }
        repaired = {
            **founder,
            "name": "Repaired",
            "implementation": founder["implementation"].replace("return 0", "return 2"),
        }
        agent = self.agent(
            [
                families(),
                experiments(0),
                json.dumps(founder),
                experiments(1),
                json.dumps(invalid),
                json.dumps(repaired),
            ],
            {"Policy 0": 0, "Repaired": 2},
            max_attempts=2,
            max_repairs=1,
        )
        await agent.run()
        child = agent.trials[1]
        self.assertEqual(child.parent_id, 1)
        self.assertEqual(child.repairs, 1)
        self.assertIn("UNCHANGED = 1", agent.best.source)
        self.assertIn("UNCHANGED = 2", child.revisions[0]["implementation"])
        self.assertIn("immutable", child.revisions[0]["error"])

    async def test_cli_saves_search_and_separate_heldout_results(self):
        from examples import lineagesearch as example

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            terminal = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "lineagesearch",
                        "--decomposition-k",
                        "1",
                        "--families",
                        "1",
                        "--initial",
                        "1",
                        "--batch-size",
                        "1",
                        "--patience",
                        "1",
                        "--bonus-batches",
                        "0",
                        "--max-repairs",
                        "1",
                        "--cull-percent",
                        "50",
                        "--seeds",
                        "0",
                        "1",
                        "--heldout-seeds",
                        "99",
                        "100",
                        "--output",
                        str(output),
                    ],
                ),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    example,
                    "OpenRouterAPI",
                    return_value=ScriptedProvider(
                        [families(), experiments(0), "bad JSON", program(0), *sequence(1)]
                    ),
                ),
                patch.object(
                    example, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
                ),
                patch(
                    "rsikit.progress.controller.Console",
                    return_value=Console(file=terminal, width=140),
                ),
            ):
                await example.main()
            summary = json.loads((output / "summary.json").read_text())
            plan = json.loads((output / "decomposition.json").read_text())
            self.assertEqual(
                plan["levels"], ["mechanism family", "experimental approach", "policy"]
            )
            self.assertEqual(len(plan["families"][0]["initial_approaches"]), 1)
            self.assertEqual(
                [d["operation"] for d in plan["decompositions"]], ["discover", "found"]
            )
            self.assertEqual(summary["reason"], "completed")
            self.assertEqual(summary["heldout"]["scores"], {"99": 7, "100": 7})
            for message in (
                "Starting LineageSearch",
                "Family leaderboard",
                "selected partition",
                "Repairing",
                "Generated Policy",
                "evaluating",
                "stagnated",
            ):
                self.assertIn(message, terminal.getvalue())
            log = (output / "run.log").read_text()
            self.assertIn("Repairing", log)
            self.assertIn("score=7", log)
            self.assertIn("survivors for next decomposition", log)
            with (
                gym.make("CartPole-v1") as env,
                recorded_run(output, environment=env) as (run, rollouts),
            ):
                with run.database() as db:
                    self.assertEqual(db.exec(select(Study)).one().config["cull_percent"], 50)
                    trials = db.exec(select(Trial)).all()
                    self.assertEqual(len(trials), 2)
                    self.assertTrue(all(set(t.seed_scores) == {"0", "1"} for t in trials))
                    self.assertEqual(trials[0].repairs, 1)
                    self.assertEqual(trials[0].revisions[0]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
