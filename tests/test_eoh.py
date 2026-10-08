import unittest
from unittest.mock import Mock

from tests.providers import ScriptedProvider
from tests.search_helpers import measure, policy, proposal, templates


class EoHTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_operators_rank_sampling_and_elite_survival(self):
        from research import eoh

        provider = ScriptedProvider([proposal(v) for v in range(6)])
        with templates(eoh):
            agent = eoh.EoH("task", provider, measure)
            result = await agent.run(population_size=1, generations=1, parents=3)
        self.assertEqual(agent.best.id, policy(5).id)
        self.assertEqual(
            [a["operation"] for a in agent.attempts], ["INIT", "E1", "E2", "M1", "M2", "M3"]
        )
        self.assertEqual([len(a["parents"]) for a in agent.attempts], [0, 3, 3, 1, 1, 1])
        self.assertEqual(agent.evaluations, 6)
        self.assertEqual(len(result), 1)
        agent.rng = Mock()
        agent._select_parents([result[0], result[0]], 5)
        self.assertEqual(agent.rng.choices.call_args.kwargs, {"weights": [1 / 3, 1 / 4], "k": 5})

    async def test_rejections_are_bounded_and_preserve_incumbent(self):
        from research import eoh

        with templates(eoh):
            agent = eoh.EoH(
                "task", ScriptedProvider([proposal(5), "bad JSON", proposal(1)]), measure
            )
            await agent.run(population_size=1, generations=1, operators=("M1", "M3"))
            self.assertEqual(agent.best.id, policy(5).id)
            self.assertEqual(agent.attempts[1]["status"], "rejected")
            agent = eoh.EoH("task", ScriptedProvider(["bad", "bad"]), measure)
            with self.assertRaisesRegex(RuntimeError, "initialize"):
                await agent.run(population_size=1, init_attempts=2)

    async def test_evaluator_errors_propagate(self):
        from research import eoh

        async def broken(policies):
            raise ValueError("scorer bug")

        agent = eoh.EoH("task", ScriptedProvider([proposal(1)]), broken)
        with templates(eoh), self.assertRaisesRegex(ValueError, "scorer bug"):
            await agent.run(population_size=1)
        self.assertEqual(agent.attempts[0]["status"], "error")

    async def test_explicit_evaluator_rejection_consumes_attempt(self):
        from research import eoh
        from research.eoh.agent import CandidateRejected

        calls = 0

        async def evaluate(policies):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise CandidateRejected("infeasible candidate")
            return await measure(policies)

        with templates(eoh):
            agent = eoh.EoH("task", ScriptedProvider([proposal(1), proposal(2)]), evaluate)
            await agent.run(population_size=1, generations=0)
        self.assertEqual(agent.best.id, policy(2).id)
        self.assertEqual(agent.attempts[0]["status"], "rejected")
        self.assertEqual(agent.evaluations, 2)
