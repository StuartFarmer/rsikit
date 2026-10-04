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
    def optimizer(self, kind):
        from research import elitesearch, lineagesearch, shinkaevolve
        from tests.test_elitesearch import program as elite_program
        from tests.test_lineagesearch import experiments, families
        from tests.test_lineagesearch import program as lineage_program

        if kind in ("original", "improved", "paper"):
            module = {"original": original, "improved": improved, "paper": paper}[kind]
            root = Path(alphaevolve.__file__).parent
            cls = module.AlphaEvolve
            config = module.Config(
                islands=1, proposals=2, batch_size=2, max_repairs=1, meta_interval=0
            )
            responses = [program(i) for i in range(3)]
        elif kind == "shinka":
            root = Path(shinkaevolve.__file__).parent / "prompts"
            cls = shinkaevolve.ShinkaEvolve
            config = shinkaevolve.Config(
                islands=1, generations=1, batch_size=2, max_repairs=1, meta_interval=0
            )
            responses = [program(i) for i in range(3)]
        elif kind == "elite":
            root = Path(elitesearch.__file__).parent / "prompts"
            cls = elitesearch.EliteSearch
            config = elitesearch.Config(
                population_size=2, elite_size=1, generations=1, max_repairs=1
            )
            responses = [elite_program(i) for i in range(3)]
        else:
            root = Path(lineagesearch.__file__).parent / "prompts"
            cls = lineagesearch.LineageSearch
            config = lineagesearch.Config(
                families=1,
                decomposition_k=1,
                initial_per_family=2,
                max_attempts=2,
                max_repairs=1,
                generation_concurrency=1,
            )
            responses = [families(), experiments(0, 1), *[lineage_program(i) for i in range(3)]]
        templates = patch.object(prompts, "TEMPLATE_ROOT", root)
        templates.start()
        self.addCleanup(templates.stop)
        provider = ScriptedProvider(responses)
        agent = cls("task", provider, config=config)
        if kind == "paper":
            self.addCleanup(agent.close)
        return agent, provider

    async def test_interrupted_generation_is_not_completion(self):
        from slick.providers import ProviderError

        for kind in ("original", "improved", "paper", "shinka"):
            with self.subTest(optimizer=kind):
                agent, provider = self.optimizer(kind)
                provider.responses = iter([program(0), ProviderError("offline")])

                async def evaluate(policies):
                    return {p.id: Measurement({0: 5}) for p in policies}

                with self.assertRaisesRegex(ProviderError, "offline"):
                    await search(agent, evaluate)
                self.assertFalse(agent.done)
                if kind == "shinka":
                    self.assertFalse(agent.generations[-1].complete)

    async def test_repairs_colliding_with_pending_ids_keep_attempts_separate(self):
        from dataclasses import replace

        for kind in ("original", "improved", "paper", "shinka"):
            for concurrency in (1, 2):
                with self.subTest(optimizer=kind, concurrency=concurrency):
                    agent, provider = self.optimizer(kind)
                    agent.config = replace(agent.config, generation_concurrency=concurrency)
                    provider.responses = iter([program(0), program(1), program(1), program(2)])
                    first, second = await agent.propose()
                    agent.update({p.id: Measurement(failure="broken") for p in (first, second)})
                    replacements = await agent.propose()
                    self.assertEqual(len(replacements), 2)
                    self.assertIn(second.id, {p.id for p in replacements})
                    agent.update({p.id: Measurement({0: 7}) for p in replacements})
                    self.assertEqual(agent.completed, 2)
                    self.assertTrue(agent.done)

    async def test_repair_collision_clears_superseded_failure_when_sibling_call_fails(self):
        from dataclasses import replace

        from slick.providers import ProviderError

        for kind in ("original", "improved", "paper", "shinka"):
            with self.subTest(optimizer=kind):
                agent, provider = self.optimizer(kind)
                agent.config = replace(agent.config, generation_concurrency=1)
                provider.responses = iter(
                    [program(0), program(1), program(1), ProviderError("offline")]
                )
                policies = await agent.propose()
                agent.update({p.id: Measurement(failure="broken") for p in policies})

                async def evaluate(policies):
                    return {p.id: Measurement({0: 5}) for p in policies}

                with self.assertRaises(ProviderError):
                    await search(agent, evaluate)
                self.assertEqual(agent.completed, 2)
                self.assertEqual(agent._pending, {})
                self.assertEqual(agent._repairs, {})

    async def test_legacy_wrapper_checkpoint_preserves_primary_error(self):
        for kind in ("elite", "lineage"):
            with self.subTest(optimizer=kind):
                agent, _ = self.optimizer(kind)
                primary = RuntimeError("original infrastructure failure")

                def broken_checkpoint(agent):
                    raise OSError("checkpoint unavailable")

                async def evaluate(policies):
                    agent.on_checkpoint = broken_checkpoint
                    raise primary

                agent.evaluate = evaluate
                with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError) as raised:
                    await agent.run()
                self.assertIs(raised.exception, primary)

    async def test_six_optimizers_conform_to_one_runner_and_feedback_contract(self):
        for kind in ("original", "improved", "paper", "shinka", "elite", "lineage"):
            for outcome in ("success", "screened", "repaired", "exhausted"):
                with self.subTest(optimizer=kind, outcome=outcome):
                    agent, provider = self.optimizer(kind)
                    panels, measurements = [], []

                    async def evaluate(policies):
                        self.assertFalse(agent.done)
                        with self.assertRaises(RuntimeError):
                            await agent.propose()
                        calls = len(provider.calls)
                        good = {p.id: Measurement({0: 3, 1: 7}) for p in policies}
                        for invalid in (
                            {},
                            {**good, "unknown": Measurement({0: 1, 1: 1})},
                            {p.id: 3.0 for p in policies},
                        ):
                            with self.assertRaises(ValueError):
                                agent.update(invalid)
                        self.assertEqual(len(provider.calls), calls)
                        if not panels:
                            first, second = policies
                            if outcome == "screened":
                                good[second.id] = Measurement(
                                    accepted=False, feedback="Below screening threshold"
                                )
                            elif outcome in ("repaired", "exhausted"):
                                good[second.id] = Measurement(failure="broken")
                        else:
                            self.assertEqual(len(policies), 1)
                            self.assertNotIn(policies[0].id, panels[0])
                            good[policies[0].id] = (
                                Measurement({0: 8, 1: 10})
                                if outcome == "repaired"
                                else Measurement(failure="still broken")
                            )
                        panels.append([p.id for p in policies])
                        measurements.append(good)
                        return good

                    best = await search(agent, evaluate)
                    self.assertTrue(agent.done)
                    self.assertEqual(best.id, agent.best.id)
                    self.assertIn(best.id, panels[-1] if outcome == "repaired" else panels[0])
                    self.assertEqual(len(panels), 2 if outcome in ("repaired", "exhausted") else 1)
                    self.assertEqual(await agent.propose(), [])
                    with self.assertRaises(ValueError):
                        agent.update(measurements[-1])
                    self.assertEqual(await search(agent, evaluate), best)

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

    async def test_elite_rounds_settle_repairs_before_promotion(self):
        from research import elitesearch
        from research.elitesearch import Config, EliteSearch
        from tests.test_elitesearch import program as elite_program

        with patch.object(prompts, "TEMPLATE_ROOT", Path(elitesearch.__file__).parent / "prompts"):
            agent = EliteSearch(
                "task",
                ScriptedProvider([elite_program(i) for i in range(3)]),
                config=Config(population_size=2, elite_size=1, generations=1, max_repairs=1),
            )
            first, second = await agent.propose()
            with self.assertRaises(RuntimeError):
                await agent.propose()
            with self.assertRaises(ValueError):
                agent.update({first.id: Measurement({0: 3}), second.id: Measurement({1: 4})})
            self.assertIsNone(agent.organisms[0].score)
            agent.update({first.id: Measurement({0: 3}), second.id: Measurement(failure="broken")})
            self.assertEqual(agent.elites, [])
            (replacement,) = await agent.propose()
            agent.update({replacement.id: Measurement({0: 7})})
            self.assertTrue(agent.done)
            self.assertEqual(agent.best.id, replacement.id)
            self.assertEqual(len(agent.generations), 1)
            self.assertEqual(len(agent.organisms), 2)
            self.assertEqual(agent.organisms[1].repairs, 1)
            with self.assertRaises(ValueError):
                agent.update({replacement.id: Measurement({0: 7})})

    async def test_lineage_rounds_preserve_family_update_and_repair_boundaries(self):
        from research import lineagesearch
        from research.lineagesearch import Config, LineageSearch
        from tests.test_lineagesearch import experiments, families
        from tests.test_lineagesearch import program as lineage_program

        with patch.object(
            prompts, "TEMPLATE_ROOT", Path(lineagesearch.__file__).parent / "prompts"
        ):
            agent = LineageSearch(
                "task",
                ScriptedProvider(
                    [
                        families(),
                        experiments(0, 1),
                        lineage_program(0),
                        lineage_program(1),
                        lineage_program(2),
                    ]
                ),
                config=Config(
                    families=1,
                    decomposition_k=1,
                    initial_per_family=2,
                    max_attempts=2,
                    max_repairs=1,
                    generation_concurrency=1,
                ),
            )
            first, second = await agent.propose()
            with self.assertRaises(RuntimeError):
                await agent.propose()
            with self.assertRaises(ValueError):
                agent.update({first.id: Measurement({0: 3}), second.id: Measurement({1: 4})})
            self.assertIsNone(agent.trials[0].score)
            agent.update({first.id: Measurement({0: 3}), second.id: Measurement(failure="broken")})
            self.assertEqual(agent.families[0].batches, 0)
            (repaired,) = await agent.propose()
            agent.update({repaired.id: Measurement({0: 7})})
            self.assertTrue(agent.done)
            self.assertEqual(agent.best.id, repaired.id)
            self.assertEqual(agent.study.attempts, 2)
            self.assertEqual(agent.families[0].batches, 1)
            with self.assertRaises(ValueError):
                agent.update({repaired.id: Measurement({0: 7})})
