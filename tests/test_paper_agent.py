"""Exercise archive-driven generation, evaluated feedback and optimizer reopening."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts, render

from research import alphaevolve
from research.alphaevolve import paper
from research.alphaevolve.generation import Mutation
from research.alphaevolve.original.agent import Guidance
from rsikit.policy import Policy
from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program


class PaperAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent)
        root.start()
        self.addCleanup(root.stop)

    async def test_reopen_retries_interrupted_original_attempt_without_losing_successful_sibling(
        self,
    ):
        from slick.providers import ProviderError

        from rsikit import search

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.sqlite"
            config = paper.Config(islands=1, proposals=2, batch_size=2, meta_interval=0)
            agent = paper.AlphaEvolve(
                "task",
                ScriptedProvider([program(0), ProviderError("offline")]),
                config=config,
                database_path=path,
            )

            async def first_evaluate(policies):
                return {p.id: episodes({0: 5}) for p in policies}

            with self.assertRaises(ProviderError):
                await search(agent, first_evaluate)
            self.assertIsNotNone(agent.best)
            agent.close()
            provider = ScriptedProvider([program(1)])
            agent = paper.AlphaEvolve("task", provider, config=config, database_path=path)
            self.addCleanup(agent.close)
            measured = []

            async def evaluate(policies):
                measured.extend(p.name for p in policies)
                return {p.id: episodes({0: 5}) for p in policies}

            await search(agent, evaluate)
            self.assertEqual(measured, ["Policy 1"])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(agent.attempts), 2)
            self.assertEqual(agent.completed, 2)
            self.assertTrue(agent.done)

    async def test_reopen_preserves_pending_round_then_repairs_without_new_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.sqlite"
            config = paper.Config(
                islands=1, proposals=2, batch_size=2, max_repairs=1, meta_interval=0
            )
            agent = paper.AlphaEvolve(
                "task",
                ScriptedProvider([program(0), program(1)]),
                config=config,
                database_path=path,
            )
            first, second = await agent.propose()
            agent.close()
            provider = ScriptedProvider([])
            agent = paper.AlphaEvolve("task", provider, config=config, database_path=path)
            self.assertEqual([p.id for p in await agent.propose()], [first.id, second.id])
            self.assertEqual(provider.calls, [])
            agent.update({first.id: episodes({0: 1, 1: 3}), second.id: episodes(failure="broken")})
            agent.close()
            agent = paper.AlphaEvolve(
                "task", ScriptedProvider([program(2)]), config=config, database_path=path
            )
            (repaired,) = await agent.propose()
            agent.update({repaired.id: episodes({0: 4, 1: 6})})
            self.assertTrue(agent.done)
            self.assertEqual(len(agent.attempts), 2)
            self.assertEqual(agent.completed, 2)
            self.assertEqual(agent.best.id, repaired.id)
            agent.close()
            agent = paper.AlphaEvolve(
                "task", ScriptedProvider([]), config=config, database_path=path
            )
            self.assertTrue(agent.done)
            self.assertEqual(agent.best.id, repaired.id)
            self.assertEqual(await agent.propose(), [])
            agent.close()

    async def test_legacy_completed_archive_opens_but_unresolved_stream_is_rejected(self):
        for complete in (True, False):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "population.sqlite"
                config = paper.Config(islands=1, proposals=1, meta_interval=0)
                agent = paper.AlphaEvolve(
                    "task", ScriptedProvider([program(0)]), config=config, database_path=path
                )
                (policy,) = await agent.propose()
                if complete:
                    agent.update({policy.id: episodes({0: 3})})
                agent.checkpoint()
                state = agent.database.load_state("optimizer")
                state.pop("round_state")
                agent.database.save_state("optimizer", state)
                agent.database.close()
                if complete:
                    restored = paper.AlphaEvolve(
                        "task", ScriptedProvider([]), config=config, database_path=path
                    )
                    self.assertTrue(restored.done)
                    self.assertEqual(restored.best.id, policy.id)
                    restored.close()
                else:
                    with self.assertRaisesRegex(ValueError, "Legacy streaming checkpoint"):
                        paper.AlphaEvolve(
                            "task", ScriptedProvider([]), config=config, database_path=path
                        )

    def test_invalid_counters_fail_before_starting_a_search(self):
        for field in (
            "migration_interval",
            "migration_count",
            "meta_interval",
            "max_repairs",
            "inspirations",
        ):
            for value in (-1, 1.5, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    paper.Config(**{field: value})

    def result(self, reward, feature):
        return paper.EvaluationResult(
            metrics={"reward": reward, "robustness": -abs(reward)},
            features={"speed": feature},
            feedback="Measured forward speed; reached the time limit.",
            seed_scores={0: reward},
        )

    async def test_round_measurements_build_internal_metrics_and_preserve_seed_identity(self):
        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(0), program(1)]),
            config=paper.Config(
                islands=1,
                proposals=2,
                batch_size=2,
                meta_interval=0,
                features={"mean_reward": (0, 10, 2), "reward_std": (0, 5, 2)},
            ),
        )
        self.addCleanup(agent.close)
        first, second = await agent.propose()
        restored = Policy.from_text(first.to_text())
        agent.update({restored.id: episodes({0: 3, 1: 7}), second.id: episodes({0: 8, 1: 8})})
        candidates = {c.policy.id: c for c in agent.database.all()}
        self.assertEqual(
            candidates[first.id].metrics, {"reward": 5, "worst_reward": 3, "stability": -2}
        )
        self.assertEqual(candidates[first.id].features, {"mean_reward": 5, "reward_std": 2})
        self.assertEqual(candidates[first.id].seed_scores, {0: 3, 1: 7})
        self.assertEqual(agent.best.id, second.id)
        self.assertTrue(agent.done)

    async def test_invalid_measurements_leave_entire_round_pending(self):
        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(0), program(1)]),
            config=paper.Config(
                islands=1, proposals=2, batch_size=2, features={"speed": (0, 10, 2)}
            ),
        )
        self.addCleanup(agent.close)
        first, second = await agent.propose()
        for results in (
            {first.id: episodes({0: 3})},
            {first.id: episodes({0: 3}, features={"speed": 2}), second.id: episodes({0: 7})},
            {p.id: episodes({0: 3}, features={"speed": True}) for p in (first, second)},
        ):
            with self.assertRaises(ValueError):
                agent.update(results)
            self.assertEqual(agent.completed, 0)
            self.assertEqual(agent.database.all(), [])
        agent.update(
            {
                p.id: episodes({0: 3}, metrics={"reward": 9}, features={"speed": 2})
                for p in (first, second)
            }
        )
        self.assertEqual(agent.database.best.score, 9)

    async def test_pending_round_and_repair_queue_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.sqlite"
            config = paper.Config(islands=1, proposals=2, batch_size=2, max_repairs=1)
            agent = paper.AlphaEvolve(
                "task",
                ScriptedProvider([program(0), program(1)]),
                config=config,
                database_path=path,
            )
            first, second = await agent.propose()
            agent.close()
            agent = paper.AlphaEvolve(
                "task", ScriptedProvider([program(2)]), config=config, database_path=path
            )
            self.assertEqual([p.id for p in await agent.propose()], [first.id, second.id])
            agent.update({first.id: episodes({0: 4}), second.id: episodes(failure="broken")})
            agent.close()
            agent = paper.AlphaEvolve(
                "task", ScriptedProvider([program(2)]), config=config, database_path=path
            )
            self.addCleanup(agent.close)
            (repaired,) = await agent.propose()
            agent.update({repaired.id: episodes({0: 8})})
            self.assertTrue(agent.done)
            self.assertEqual((agent.completed, agent.repair_calls), (2, 1))

    async def test_scalar_feedback_updates_the_paper_archive(self):
        agent = paper.AlphaEvolve(
            "task", ScriptedProvider([program(0)]), config=paper.Config(islands=1)
        )
        self.addCleanup(agent.close)
        (policy,) = await agent.generate(1)
        agent.update_scores({policy.id: 3}, seed_scores={policy.id: {42: 3}})
        (candidate,) = agent.database.all()
        self.assertEqual(candidate.policy.id, policy.id)
        self.assertEqual(candidate.metrics, {"reward": 3})
        self.assertEqual(candidate.seed_scores, {42: 3})
        self.assertEqual(agent.database.load_state("optimizer")["completed"], 1)

    async def test_runtime_policy_error_is_repaired_before_entering_archive(self):
        provider = ScriptedProvider([program(0), program(1)])
        agent = paper.AlphaEvolve("task", provider, config=paper.Config(islands=1, meta_interval=0))
        self.addCleanup(agent.close)
        diagnostic = "ValueError: cannot reshape array of size 6 into shape (2,6)"
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(policies)
            if len(evaluated) == 1:
                return {policies[0].id: episodes(failure=diagnostic)}
            return {p.id: episodes({0: 1}) for p in policies}

        await paper.search(agent, evaluate, proposals=1, evaluation_batch_size=1)

        self.assertEqual(len(evaluated), 2)
        self.assertEqual(agent.best._implementation, program(1).implementation)
        self.assertEqual([c.policy.id for c in agent.database.all()], [evaluated[1].id])
        self.assertEqual((agent.completed, agent.repair_calls), (1, 1))
        self.assertEqual(agent._pending, {})
        self.assertIn(diagnostic, provider.calls[1])
        self.assertIn(agent.libraries, provider.calls[1])

    async def test_lower_reward_niches_feed_generation_and_restore(self):
        config = paper.Config(
            islands=1,
            features={"speed": (0, 2, 2)},
            exploration=1,
            mode="rewrite",
            meta_interval=0,
            migration_interval=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.sqlite"
            provider = ScriptedProvider([program(i) for i in range(4)])
            agent = paper.AlphaEvolve("task", provider, config=config, database_path=path)
            initial = await agent.generate(n=2)
            agent.update_results(
                {
                    initial[0].id: self.result(10, 0.2),
                    initial[1].id: self.result(5, 1.2),
                }
            )
            self.assertEqual(len(agent.database.members(0)), 2)
            parents = {agent.sample()[1].policy.id for _ in range(30)}
            self.assertEqual(parents, {p.id for p in initial})
            await agent.generate()
            self.assertIn('"robustness"', provider.calls[-1])
            self.assertIn('"speed"', provider.calls[-1])
            self.assertIn("Measured forward speed", provider.calls[-1])
            rng_state = agent.rng.getstate()
            expected_parent = agent.sample()[1].policy.id
            agent.rng.setstate(rng_state)
            agent.close()
            restored = paper.AlphaEvolve("task", provider, config=config, database_path=path)
            self.assertEqual(restored.best.id, initial[0].id)
            self.assertEqual(restored.completed, 2)
            self.assertEqual(restored.generation_calls, 3)
            self.assertEqual(len(restored.database.members(0)), 2)
            self.assertEqual(restored.sample()[1].policy.id, expected_parent)
            await restored.generate()
            self.assertEqual(restored.attempts[-1]["id"], 4)
            restored.close()

    async def test_renamed_duplicate_does_not_reward_guidance_for_noise(self):
        duplicate = program(0).model_copy(
            update={
                "name": "Renamed",
                "implementation": program(0).implementation.replace(
                    "return 0", "return 0  # comment"
                ),
            }
        )
        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([duplicate]),
            config=paper.Config(islands=1, mode="rewrite", meta_interval=0),
        )
        self.addCleanup(agent.close)
        seed = Policy.from_text(program(0).implementation, name="Initial")
        agent.register_initial(seed, paper.EvaluationResult(metrics={"reward": 1}))
        child = (await agent.generate())[0]
        agent.update_results({child.id: paper.EvaluationResult(metrics={"reward": 100})})
        self.assertEqual(agent.prompt_ideas[0].reward, 0)
        self.assertEqual(agent.best.id, seed.id)
        self.assertEqual(agent.attempts[-1]["score"], 100)  # Preserve measured history.

    async def test_invalid_result_batch_does_not_partially_enter_archive(self):
        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(0), program(1)]),
            config=paper.Config(islands=1, features={"speed": (0, 2, 2)}, meta_interval=0),
        )
        self.addCleanup(agent.close)
        policies = await agent.generate(n=2)
        with self.assertRaises(ValueError):
            agent.update_results(
                {
                    policies[0].id: self.result(10, 0.2),
                    policies[1].id: paper.EvaluationResult(metrics={"reward": 20}),
                }
            )
        self.assertEqual(agent.database.all(), [])
        self.assertEqual(len(agent._pending), 2)
        self.assertEqual(agent.completed, 0)

    async def test_evaluated_seed_and_rejected_child(self):
        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(1)]),
            config=paper.Config(mode="rewrite", meta_interval=0),
        )
        self.addCleanup(agent.close)
        seed = Policy.from_text(program(0).implementation, name="Initial")
        agent.register_initial(seed, paper.EvaluationResult(metrics={"reward": 1}))
        self.assertTrue(all(island.policy.id == seed.id for island in agent.islands))
        child = (await agent.generate())[0]
        agent.update_results(
            {
                child.id: paper.EvaluationResult(
                    metrics={"reward": 100},
                    accepted=False,
                    feedback="Rejected by screening",
                )
            }
        )
        self.assertEqual(agent.best.id, seed.id)
        self.assertEqual(len(agent.database.all()), 1)
        self.assertEqual(agent.attempts[-1]["status"], "discarded")
        self.assertEqual(agent._pending, {})

    async def test_meta_guidance_uses_measured_feedback_and_retains_credit(self):
        provider = ScriptedProvider(
            [
                program(0),
                Guidance(instruction="Try alternating stance phases"),
                program(1),
            ]
        )
        agent = paper.AlphaEvolve(
            "task",
            provider,
            config=paper.Config(islands=1, mode="rewrite", meta_interval=2),
        )
        self.addCleanup(agent.close)
        initial = (await agent.generate())[0]
        agent.update_results(
            {
                initial.id: paper.EvaluationResult(
                    metrics={"reward": 1},
                    feedback="No forward displacement",
                )
            }
        )
        child = (await agent.generate())[0]
        self.assertIn("No forward displacement", provider.calls[1])
        self.assertIn("Try alternating stance phases", provider.calls[2])
        agent.update_results({child.id: paper.EvaluationResult(metrics={"reward": 2})})
        self.assertEqual(agent.prompt_ideas[-1].score, 1)
        self.assertEqual(agent.database.load_state("optimizer")["prompt_ideas"][-1]["reward"], 1)
        schema = Mutation.model_json_schema()
        text = render(
            "paper/prompts/mutate.j2",
            instance=agent,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parent=agent.islands[0],
            inspirations=[],
            guidance="",
            failures=[],
        )
        self.assertIn("exact edits", text)


if __name__ == "__main__":
    unittest.main()
