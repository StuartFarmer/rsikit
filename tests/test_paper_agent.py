"""Exercise archive-driven generation, evaluated feedback and optimizer reopening."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts, render

from research import alphaevolve
from research.alphaevolve import paper
from research.alphaevolve.original.agent import Guidance
from rsikit.evaluation import PolicyError
from rsikit.generation.edits import Mutation
from rsikit.policy import Policy
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program
from tests.test_episode_storage import trajectory


class PaperAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent)
        root.start()
        self.addCleanup(root.stop)

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

    async def test_propose_and_episode_pairs_derive_fitness_by_persistent_identity(self):
        from rsikit import Episode, Optimizer

        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(0), program(1)]),
            config=paper.Config(
                islands=1,
                meta_interval=0,
                features={"mean_reward": (0, 10, 2), "reward_std": (0, 5, 2)},
            ),
        )
        self.addCleanup(agent.close)
        optimizer: Optimizer = agent
        policies = await optimizer.propose(2)
        # Reloaded definitions have different Python identities but the same persistent ID.
        restored = Policy.from_text(policies[0].to_text())
        first_episode = Episode(
            observations=[0, 1, 2],
            actions=[0, 0],
            rewards=[1.0, 2.0],
            terminations=[False, True],
            truncations=[False, False],
            infos=[{}, {}, {}],
        )
        episodes = [first_episode, trajectory(7), trajectory(8)]
        optimizer.update(zip([restored, policies[0], policies[1]], episodes, strict=True))
        candidates = {row.policy.id: row for row in agent.database.all()}
        first = candidates[policies[0].id]
        self.assertEqual(first.metrics, {"reward": 5, "worst_reward": 3, "stability": -2})
        self.assertEqual(first.features, {"mean_reward": 5, "reward_std": 2})
        self.assertEqual(first.seed_scores, {})  # An Episode does not identify its seed.
        self.assertEqual(
            candidates[policies[1].id].metrics, {"reward": 8, "worst_reward": 8, "stability": 0}
        )
        self.assertEqual(agent.best.id, policies[1].id)
        self.assertEqual(agent.completed, 2)
        self.assertEqual(agent._pending, {})

    async def test_bad_episode_pairs_leave_the_whole_batch_pending(self):
        from rsikit import Episode

        agent = paper.AlphaEvolve(
            "task",
            ScriptedProvider([program(0), program(1)]),
            config=paper.Config(islands=1, meta_interval=0),
        )
        self.addCleanup(agent.close)
        policies = await agent.generate(2)
        for episodes in (
            [trajectory(3), Episode()],
            [trajectory(3), trajectory(float("nan"))],
            [trajectory(3)],
        ):
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                agent.update(zip(policies, episodes, strict=True))
            self.assertEqual(agent.completed, 0)
            self.assertEqual(agent.database.all(), [])
            self.assertEqual(len(agent._pending), 2)

    async def test_scalar_feedback_updates_the_paper_archive(self):
        agent = paper.AlphaEvolve(
            "task", ScriptedProvider([program(0)]), config=paper.Config(islands=1)
        )
        self.addCleanup(agent.close)
        (policy,) = await agent.propose(1)
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
                error = PolicyError(diagnostic)
                error.failures = {policies[0].id: diagnostic}
                raise error
            return {p.id: paper.EvaluationResult({"reward": 1}) for p in policies}

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
