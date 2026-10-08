import unittest

from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.search_helpers import measure, policy, proposal, templates


class ReEvoTests(unittest.IsolatedAsyncioTestCase):
    async def test_reflection_crossover_elite_mutation_and_budget(self):
        from research import reevo

        provider = ScriptedProvider(
            [proposal(1), proposal(2), "compare", proposal(5), "word " * 80, proposal(4)]
        )
        events = []
        with templates(reevo):
            agent = reevo.ReEvo(
                "task",
                provider,
                measure,
                config=reevo.Config(
                    initial_size=2,
                    population_size=1,
                    max_evaluations=4,
                    mutation_rate=1,
                ),
                on_event=lambda kind, data: events.append((kind, data)),
            )
            result = await agent.run()
        self.assertEqual(agent.best.id, policy(5).id)
        self.assertEqual(len(result.individuals), 4)
        self.assertEqual(result.individuals[-1].parents, [2])
        self.assertEqual(len(agent.long_term.split()), 49)
        self.assertIn("Policy 5", provider.calls[-1])
        self.assertEqual(result.stop_reason, "budget")
        self.assertEqual(result.reflections[0]["pairs"], [[0, 1]])
        agent.population = [result.individuals[0]]
        # The global incumbent remains eligible after generational replacement.
        agent.config = reevo.Config(population_size=1, max_evaluations=5)
        self.assertEqual([row.id for row in agent.select_parents()[0]], [0, 2])
        self.assertTrue(any(kind == "raw_response" for kind, _ in events))

    async def test_bad_generation_and_policy_failure_consume_budget(self):
        from research import reevo

        async def failed(policies):
            return {p.id: episodes({}, failure="bad action") for p in policies}

        with templates(reevo):
            agent = reevo.ReEvo(
                "task",
                ScriptedProvider(["bad json", proposal(2)]),
                failed,
                config=reevo.Config(initial_size=2, max_evaluations=2),
            )
            result = await agent.run()
        self.assertIsNone(agent.best)
        self.assertEqual(len(result.individuals), 2)
        self.assertTrue(all(row.error for row in result.individuals))

    async def test_ties_stop_and_zero_budget_does_not_evaluate_seed(self):
        from research import reevo

        with templates(reevo):
            agent = reevo.ReEvo(
                "task",
                ScriptedProvider([proposal(1), proposal(1)]),
                measure,
                config=reevo.Config(initial_size=2, max_evaluations=5),
            )
            self.assertEqual((await agent.run()).stop_reason, "no_distinct_parents")
            agent = reevo.ReEvo(
                "task",
                ScriptedProvider([]),
                measure,
                seed_policy=policy(3),
                config=reevo.Config(max_evaluations=0),
            )
            self.assertEqual((await agent.run()).individuals, [])

    async def test_real_process_measurements_are_persisted(self):
        import tempfile
        from pathlib import Path

        from research import reevo
        from rsikit import Executor, Run
        from tests.test_execution import ProcessEnv

        with tempfile.TemporaryDirectory() as directory, templates(reevo):
            with ProcessEnv() as env:
                async with (
                    Executor(concurrency=2) as executor,
                    Run.create(name="reevo", path=Path(directory) / "run") as run,
                ):

                    async def evaluate(policies):
                        return await run.evaluate(
                            policies, environment=env, executor=executor, seeds=(0, 1)
                        )

                    agent = reevo.ReEvo(
                        "task",
                        ScriptedProvider([proposal(1), proposal(2)]),
                        evaluate,
                        config=reevo.Config(initial_size=2, max_evaluations=2),
                    )
                    await agent.run()
                    self.assertIsNotNone(agent.best)
                    self.assertEqual(len(run.policies()), 2)
                    self.assertIsNotNone(run.load_episode(agent.best, 0))

    async def test_disabled_operators_stop_without_spinning(self):
        from research import reevo

        with templates(reevo):
            result = await reevo.ReEvo(
                "task",
                ScriptedProvider([proposal(1)]),
                measure,
                config=reevo.Config(initial_size=1, crossover_rate=0, mutation_rate=0),
            ).run()
        self.assertEqual(result.stop_reason, "no_offspring")
