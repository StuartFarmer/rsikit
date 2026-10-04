"""Real optimizers obey the same complete-round contract."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from research import alphaevolve
from research.alphaevolve import improved, original, paper
from rsikit import Measurement, search
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program


class OptimizerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_alpha_variants_use_the_same_runner_and_seed_evidence(self):
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent):
            for variant in (original, improved, paper):
                with self.subTest(variant=variant.__name__):
                    agent = variant.AlphaEvolve(
                        "task",
                        ScriptedProvider([program(0)]),
                        config=variant.Config(islands=1, proposals=1, batch_size=1),
                    )
                    if hasattr(agent, "close"):
                        self.addCleanup(agent.close)

                    async def evaluate(policies):
                        return {p.id: Measurement({0: 3, 1: 7}) for p in policies}

                    best = await search(agent, evaluate)
                    self.assertEqual(agent.islands[0].score, 5)
                    self.assertEqual(best.id, agent.best.id)
                    self.assertTrue(agent.done)
                    self.assertEqual(await agent.propose(), [])
                    self.assertEqual(agent.completed, 1)
                    with self.assertRaises(ValueError):
                        agent.update({best.id: Measurement({0: 3, 1: 7})})

    async def test_alpha_round_validation_and_duplicate_attempts(self):
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent):
            for variant in (original, improved, paper):
                agent = variant.AlphaEvolve(
                    "task",
                    ScriptedProvider([program(0), program(0)]),
                    config=variant.Config(islands=1, proposals=2, batch_size=2),
                )
                if hasattr(agent, "close"):
                    self.addCleanup(agent.close)
                (policy,) = await agent.propose()
                with self.assertRaises(RuntimeError):
                    await agent.propose()
                with self.assertRaises(ValueError):
                    agent.update({})
                self.assertEqual(agent.completed, 0)
                agent.update({policy.id: Measurement({0: 4})})
                self.assertEqual(agent.completed, 2)
                self.assertTrue(agent.done)

    async def test_alpha_repairs_only_failures_without_new_attempts(self):
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent):
            for variant in (original, improved, paper):
                agent = variant.AlphaEvolve(
                    "task",
                    ScriptedProvider([program(0), program(1), program(2)]),
                    config=variant.Config(islands=1, proposals=2, batch_size=2, max_repairs=1),
                )
                if hasattr(agent, "close"):
                    self.addCleanup(agent.close)
                first, second = await agent.propose()
                agent.update(
                    {first.id: Measurement({0: 3}), second.id: Measurement(failure="broken")}
                )
                self.assertFalse(agent.done)
                (replacement,) = await agent.propose()
                self.assertNotIn(replacement.id, (first.id, second.id))
                agent.update({replacement.id: Measurement({0: 7})})
                self.assertTrue(agent.done)
                self.assertEqual(len(agent.attempts), 2)
                self.assertEqual(agent.completed, 2)
                self.assertEqual(agent.repair_calls, 1)
