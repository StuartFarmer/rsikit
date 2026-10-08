import unittest

from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.search_helpers import policy, proposal, templates
from tests.test_episode_storage import trajectory


class GEPATests(unittest.IsolatedAsyncioTestCase):
    def test_frontier_keeps_specialists_and_prunes_redundant_ties(self):
        from research.gepa.agent import pareto_weights

        self.assertEqual(pareto_weights([(10, 0), (0, 10), (5, 5)]), {0: 1, 1: 1})
        self.assertEqual(len(pareto_weights([(1, 1), (1, 1)])), 1)

    async def test_matched_training_panel_and_complete_selection_budget(self):
        from research import gepa

        calls, traces = [], []

        async def evaluate(policies, *, seeds):
            calls.append((policies[0].id, tuple(seeds)))
            return {
                p.id: episodes({s: float(p.description.split()[-1]) for s in seeds})
                for p in policies
            }

        def trace(p, seed):
            traces.append(seed)
            return trajectory(1)

        provider = ScriptedProvider([proposal(2)])
        with templates(gepa):
            agent = gepa.GEPA("task", provider, evaluate, trace)
            result = await agent.run(
                policy(1),
                train_seeds=[0, 1],
                selection_seeds=[100, 101],
                budget=6,
                minibatch_size=1,
            )
        self.assertEqual(result.rollouts, 6)
        self.assertEqual(calls[1][1], calls[2][1])
        self.assertEqual(calls[-1][1], (100, 101))
        self.assertEqual(len(traces), 1)
        self.assertNotIn(100, traces)
        self.assertEqual(agent.best.id, policy(2).id)
        self.assertNotIn('"seed": 100', provider.calls[0])

    async def test_rejection_and_short_budget_keep_incumbent(self):
        from research import gepa
        from tests.search_helpers import measure

        for response in (proposal(0), "bad json"):
            with templates(gepa):
                agent = gepa.GEPA(
                    "task", ScriptedProvider([response]), measure, lambda p, s: trajectory()
                )
                result = await agent.run(
                    policy(1), train_seeds=[0], selection_seeds=[10], budget=4, minibatch_size=1
                )
            self.assertEqual(agent.best.id, policy(1).id)
            self.assertEqual(len(result.population), 1)
        with templates(gepa):
            provider = ScriptedProvider([])
            agent = gepa.GEPA("task", provider, measure, lambda p, s: trajectory())
            await agent.run(
                policy(1), train_seeds=[0], selection_seeds=[10], budget=3, minibatch_size=1
            )
            self.assertEqual(provider.calls, [])
            with self.assertRaises(ValueError):
                await agent.run(policy(1), train_seeds=[0], selection_seeds=[0])
