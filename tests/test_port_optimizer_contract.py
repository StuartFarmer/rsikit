"""The six ports use the same complete-round policy/episode boundary."""

import asyncio
import unittest

from rsikit import Episode, PolicyDefinition, search
from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.search_helpers import measure, policy, proposal, templates
from tests.test_episode_storage import trajectory


class PortOptimizerContractTests(unittest.IsolatedAsyncioTestCase):
    def optimizer(self, kind, responses=None):
        from research import eoh, evox, promptbreeder, reevo
        from research.evox.runtime import run_python_strategy

        module, cls, kwargs = {
            "reevo": (
                reevo,
                reevo.ReEvo,
                {"config": reevo.Config(initial_size=2, max_evaluations=2)},
            ),
            "evox": (
                evox,
                evox.EvoX,
                {
                    "config": evox.Config(iterations=2, window=2),
                    "run_strategy": run_python_strategy,
                },
            ),
            "eoh": (eoh, eoh.EoH, {"population_size": 2, "generations": 0}),
            "promptbreeder": (
                promptbreeder,
                promptbreeder.PromptBreeder,
                {"population_size": 2, "tournaments": 0},
            ),
        }[kind]
        patcher = templates(module)
        patcher.start()
        self.addCleanup(patcher.stop)
        responses = [proposal(1), proposal(2)] if responses is None else responses
        if kind == "evox":
            responses = ['{"refine":"tune","diverge":"explore"}', *responses]
        provider = ScriptedProvider(responses)
        agent = cls("task", provider, **kwargs)
        self.addAsyncCleanup(agent.aclose)
        return agent, provider

    async def test_complete_round_validation_best_seed_panel_and_raw_evidence(self):
        for kind in ("reevo", "evox", "eoh", "promptbreeder"):
            with self.subTest(kind=kind):
                agent, _ = self.optimizer(kind)
                self.assertFalse(agent.done)
                (first,) = await agent.propose()
                self.assertIsInstance(first, PolicyDefinition)
                with self.assertRaises(RuntimeError):
                    await agent.propose()
                invalid_episode = trajectory()
                invalid_episode.rewards[0] = float("nan")
                for invalid in (
                    {},
                    {"unknown": episodes({0: 3})},
                    {first.id: 3},
                    {first.id: {0: Episode()}},
                    {first.id: {0: invalid_episode}},
                ):
                    with self.assertRaises(ValueError):
                        agent.update(invalid)
                    self.assertIsNone(agent.best)
                raw = episodes({7: 3, 9: 7})
                raw[7].artifacts["blob"] = b"preserve"
                raw[7].infos[-1]["fitness"] = 5
                agent.update({first.id: raw})
                self.assertEqual(agent.best.id, first.id)
                self.assertIs(agent.evidence[-1][first.id][7], raw[7])
                with self.assertRaises(ValueError):
                    agent.update({first.id: raw})
                (second,) = await agent.propose()
                with self.assertRaises(ValueError):
                    agent.update({second.id: episodes({0: 9, 1: 9})})
                agent.update({second.id: episodes({9: 8, 7: 8})})
                self.assertEqual(agent.best.id, second.id)
                self.assertEqual(await agent.propose(), [])
                self.assertTrue(agent.done)
                self.assertEqual(await search(agent, measure), agent.best)

    async def test_failed_panels_and_screening_never_select_partial_rewards(self):
        for kind in ("reevo", "evox", "eoh", "promptbreeder"):
            for rejected in ({}, {0: trajectory(1000), 1: Episode(error="crashed")}):
                with self.subTest(kind=kind, rejected=bool(rejected)):
                    agent, _ = self.optimizer(kind, [proposal(1), proposal(2), proposal(3)])
                    (first,) = await agent.propose()
                    agent.update({first.id: rejected})
                    self.assertIsNone(agent.best)
                    self.assertEqual(agent.evidence[0][first.id], rejected)
                    best = await search(agent, measure)
                    self.assertIsNotNone(best)
                    self.assertNotEqual(best.id, first.id)
                    self.assertTrue(agent.done)

    async def test_interrupted_generation_stays_interrupted_and_keeps_incumbent(self):
        for kind in ("reevo", "evox", "eoh", "promptbreeder"):
            for failure in (RuntimeError("offline"), asyncio.CancelledError()):
                with self.subTest(kind=kind, failure=type(failure).__name__):
                    agent, provider = self.optimizer(kind, [proposal(1)])
                    (first,) = await agent.propose()
                    agent.update({first.id: episodes({0: 3})})

                    async def interrupted(*args, **kwargs):
                        raise failure

                    provider.acall = interrupted
                    for _ in range(2):
                        with self.assertRaises(type(failure)):
                            await agent.propose()
                        self.assertFalse(agent.done)
                        self.assertEqual(agent.best.id, first.id)
                        self.assertEqual(len(agent.evidence), 1)

    async def test_interrupted_evaluation_keeps_outstanding_round(self):
        agent, _ = self.optimizer("reevo")

        async def interrupted(policies):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await search(agent, interrupted)
        self.assertFalse(agent.done)
        with self.assertRaises(RuntimeError):
            await agent.propose()
        pending = agent._pending_policy
        agent.update({pending.id: episodes({0: 5})})
        self.assertEqual(agent.best.id, pending.id)
        await search(agent, measure)
        self.assertTrue(agent.done)

    async def test_gepa_selection_panel_and_training_evidence_are_separate(self):
        from research import gepa

        with templates(gepa):
            agent = gepa.GEPA(
                "task",
                ScriptedProvider([proposal(2)]),
                measure,
                lambda *_: trajectory(1),
                initial_policy=policy(1),
                train_seeds=[0],
                selection_seeds=[10, 11],
                budget=6,
                minibatch_size=1,
            )
            (initial,) = await agent.propose()
            with self.assertRaises(ValueError):
                agent.update({initial.id: episodes({10: 1})})
            self.assertIsNone(agent.best)
            agent.update({initial.id: episodes({11: 2, 10: 1})})
            (child,) = await agent.propose()
            agent.update({child.id: episodes({11: 5, 10: 4})})
            self.assertEqual(agent.best.id, child.id)
            self.assertEqual(agent.result.population[-1].scores, (4, 5))
            self.assertEqual(agent.result.rollouts, 6)
            self.assertEqual(len(agent.training_evidence), 2)
            self.assertEqual(await agent.propose(), [])
            self.assertTrue(agent.done)

    async def test_gepa_failed_initial_panel_completes_without_selecting_partial_rewards(self):
        from research import gepa

        agent = gepa.GEPA(
            "task", ScriptedProvider([]), measure, lambda *_: trajectory(), initial_policy=policy(1)
        )
        (initial,) = await agent.propose()
        agent.update({initial.id: {10: trajectory(1000), 11: Episode(error="failed")}})
        self.assertEqual(await agent.propose(), [])
        self.assertTrue(agent.done)
        self.assertIsNone(agent.best)

    async def test_stop_deployments_are_bounded_and_use_named_policy_evidence(self):
        from research import stop_optimizer

        inputs = []

        async def execute(source, initial, caps):
            inputs.append(initial)
            return policy(len(inputs)).to_text()

        async def utility(source):
            return 0.5

        agent = stop_optimizer.STOP(
            "task",
            ScriptedProvider([]),
            execute,
            [stop_optimizer.Problem(policy(0).to_text(), "reward", utility)],
            rounds=0,
            deployments=2,
        )
        (first,) = await agent.propose()
        agent.update({first.id: episodes({0: 5})})
        self.assertEqual(agent.best.id, first.id)
        (second,) = await agent.propose()
        self.assertEqual(inputs[-1], first.to_text())
        agent.update({second.id: episodes({0: 3})})
        self.assertEqual(agent.best.id, first.id)
        self.assertEqual(agent.best.name, "Policy 1")
        self.assertEqual(await agent.propose(), [])
        self.assertTrue(agent.done)
