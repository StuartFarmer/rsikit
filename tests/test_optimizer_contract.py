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

    async def test_shinka_rounds_repair_without_an_extra_generation(self):
        from research import shinkaevolve
        from research.shinkaevolve import Config, ShinkaEvolve

        with patch.object(prompts, "TEMPLATE_ROOT", Path(shinkaevolve.__file__).parent / "prompts"):
            agent = ShinkaEvolve(
                "task",
                ScriptedProvider([program(0), program(1), program(2)]),
                config=Config(
                    islands=1, generations=1, batch_size=2, max_repairs=1, meta_interval=0
                ),
            )
            first, second = await agent.propose()
            with self.assertRaises(RuntimeError):
                await agent.propose()
            with self.assertRaises(ValueError):
                agent.update({first.id: Measurement({0: 1})})
            agent.update(
                {first.id: Measurement({0: 3, 1: 7}), second.id: Measurement(failure="broken")}
            )
            self.assertFalse(agent.done)
            (repaired,) = await agent.propose()
            agent.update({repaired.id: Measurement({0: 8, 1: 8})})
            self.assertTrue(agent.done)
            self.assertEqual(len(agent.generations), 1)
            self.assertEqual(agent.completed, 2)
            self.assertEqual(sum(map(len, agent.model_gains)), 2)
            self.assertTrue(agent.generations[0].complete)
            self.assertEqual(agent.best.id, repaired.id)
            with self.assertRaises(ValueError):
                agent.update({repaired.id: Measurement({0: 8, 1: 8})})

    async def test_shinka_reflection_runs_before_next_proposal(self):
        from research import shinkaevolve
        from research.shinkaevolve import Config, ShinkaEvolve

        with patch.object(prompts, "TEMPLATE_ROOT", Path(shinkaevolve.__file__).parent / "prompts"):
            provider = ScriptedProvider(
                [program(0), '{"recommendations": ["Try steady control"]}', program(1)]
            )
            agent = ShinkaEvolve(
                "task",
                provider,
                config=Config(
                    islands=1,
                    batch_size=1,
                    generations=2,
                    meta_interval=1,
                    patch_types=(("full", 1.0),),
                ),
            )
            (first,) = await agent.propose()
            agent.update({first.id: Measurement({0: 1})})
            self.assertEqual(agent.meta_calls, 0)
            (second,) = await agent.propose()
            self.assertEqual(agent.meta_calls, 1)
            self.assertIn("Try steady control", provider.calls[-1])
            agent.update({second.id: Measurement({0: 2})})
            self.assertTrue(agent.done)
