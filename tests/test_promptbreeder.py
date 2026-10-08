import unittest

from tests.providers import ScriptedProvider
from tests.search_helpers import measure, policy, proposal, templates


class PromptBreederTests(unittest.IsolatedAsyncioTestCase):
    async def test_hypermutation_is_inherited_and_valid_worse_child_replaces_loser(self):
        from research import promptbreeder

        provider = ScriptedProvider(
            [proposal(5), proposal(2), "Focus on control stability", proposal(-1)]
        )
        with templates(promptbreeder):
            agent = promptbreeder.PromptBreeder("task", provider, measure)
            result = await agent.run(population_size=2, tournaments=1, seed=0)
        self.assertEqual(agent.best.id, policy(5).id)
        child = next(row for row in result.population if row.policy.id == policy(-1).id)
        self.assertEqual(child.mutation, "Focus on control stability")
        self.assertIn("Focus on control stability", provider.calls[-1])
        self.assertEqual(agent.model_calls, 4)

    async def test_odd_population_pairs_do_not_overlap_and_invalid_child_retains_loser(self):
        from research import promptbreeder

        provider = ScriptedProvider([proposal(1), proposal(2), proposal(3), "improve", "bad json"])
        with templates(promptbreeder):
            agent = promptbreeder.PromptBreeder("task", provider, measure)
            result = await agent.run(population_size=3, tournaments=1, seed=0)
        self.assertEqual(
            {r.policy.id for r in result.population}, {policy(v).id for v in (1, 2, 3)}
        )
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(len(set(result.pairs[0])), 2)
        self.assertTrue(result.attempts[-1]["error"])

    async def test_initialization_budget_is_bounded(self):
        from research import promptbreeder

        with templates(promptbreeder), self.assertRaisesRegex(RuntimeError, "initialize"):
            await promptbreeder.PromptBreeder("task", ScriptedProvider(["bad"] * 6), measure).run(
                population_size=2
            )
