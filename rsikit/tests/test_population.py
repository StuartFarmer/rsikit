"""Verify strategy policy differences with deterministic proposals and measured scores."""

import unittest

from rsikit import AlphaEvolve, DGMArchive, EoH, Evaluation, HillClimb, SequentialStrategy


class PopulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_diversity_callback_does_not_commit_half_an_update(self):
        fail = True

        def cell(candidate):
            nonlocal fail
            if candidate.id and fail:
                fail = False
                raise RuntimeError("descriptor failed")
            return (candidate.id,)

        strategy = await self.make_strategy(AlphaEvolve, cell=cell)
        candidates = await strategy.generate()
        evaluations = [Evaluation(valid=True, metrics={"score": 2})]
        with self.assertRaisesRegex(RuntimeError, "descriptor failed"):
            await strategy.update(candidates, evaluations)
        self.assertEqual([c.id for c in strategy.history], [0])
        self.assertEqual(strategy.best.id, 0)
        self.assertEqual(strategy.selections, [])
        self.assertEqual(strategy.pending, candidates[0])
        await strategy.update(candidates, evaluations)
        self.assertEqual([c.id for c in strategy.history], [0, 1])
        self.assertEqual(strategy.best.id, 1)
        self.assertEqual(len(strategy.selections), 1)

    async def make_strategy(self, kind, **options):
        async def propose(parent, history):
            return f"source {len(history)}"

        return kind("seed", Evaluation(valid=True, metrics={"score": 1}), propose, **options)

    async def step(self, strategy, score, *, valid=True):
        candidates = await strategy.generate()
        await strategy.update(candidates, [Evaluation(valid=valid, metrics={"score": score})])
        return strategy.history[-1]

    async def test_shared_contract_preserves_pending_and_minimizes(self):
        with self.assertRaises(TypeError):
            SequentialStrategy("seed", Evaluation(valid=True), None)
        for kind in (HillClimb, AlphaEvolve, EoH, DGMArchive):
            with self.subTest(kind=kind):
                strategy = await self.make_strategy(kind, maximize=False)
                pending = await strategy.generate()
                with self.assertRaises(ValueError):
                    await strategy.update([], [])
                with self.assertRaises(RuntimeError):
                    await strategy.generate()
                self.assertEqual(strategy.pending, pending[0])
                await strategy.update(pending, [Evaluation(valid=True, metrics={"score": 0.5})])
                self.assertEqual(strategy.best.id, 1)
                await self.step(strategy, 0.9)
                self.assertEqual(strategy.best.id, 1)

    async def test_alphaevolve_retains_cell_diversity_and_reseeds_weak_islands(self):
        strategy = await self.make_strategy(
            AlphaEvolve, islands=2, cell=lambda c: (c.id % 2,), reset_interval=0
        )
        weaker = await self.step(strategy, 0.5)
        self.assertEqual(strategy.best.id, 0)
        self.assertTrue(any(weaker in island.values() for island in strategy.islands))
        winner = await self.step(strategy, 2)
        # Set explicit island champions to verify the reset policy independently of RNG draws.
        strategy.islands = [{(0,): strategy.history[0]}, {(0,): winner}]
        strategy.reset_islands()
        self.assertTrue(all(list(island.values()) == [winner] for island in strategy.islands))
        self.assertEqual(strategy.events[-1]["founder"], winner.id)
        self.assertIn("island", strategy.selections[-1])

    async def test_eoh_uses_snapshot_for_all_five_operators_before_selecting_elites(self):
        strategy = await self.make_strategy(EoH, population_size=2, seed=3)
        await self.step(strategy, 1.1)
        self.assertEqual(strategy.selections[-1]["operation"], "INIT")
        snapshot = tuple(strategy.population)
        operations = []
        for index in range(10):
            await self.step(strategy, 2 + index)
            operations.append(strategy.selections[-1]["operation"])
            if index < 9:
                self.assertEqual(tuple(strategy.population), snapshot)
            self.assertTrue(set(strategy.selections[-1]["parents"]) <= {0, 1})
        self.assertEqual(operations, [op for op in EoH.OPERATORS for _ in range(2)])
        self.assertEqual([c.evaluation.metrics["score"] for c in strategy.population], [11, 10])
        self.assertEqual(strategy.cycles, 1)
        self.assertEqual(strategy.best.evaluation.metrics["score"], 11)

    async def test_eoh_failed_attempt_advances_operator_cycle_without_entering_population(self):
        strategy = await self.make_strategy(EoH, population_size=1)
        for _ in range(5):
            await self.step(strategy, 100, valid=False)
        self.assertEqual(strategy.cycles, 1)
        self.assertEqual([c.id for c in strategy.population], [0])
        self.assertEqual([s["operation"] for s in strategy.selections], list(EoH.OPERATORS))

    async def test_dgm_archives_regressions_and_penalizes_parents_with_children(self):
        strategy = await self.make_strategy(DGMArchive, score_bounds=(0, 2))
        original_weight = strategy.selection_weights()[0]
        worse = await self.step(strategy, 0.2)
        self.assertEqual(strategy.archive, [strategy.history[0], worse])
        self.assertEqual(strategy.best.id, 0)
        self.assertAlmostEqual(strategy.selection_weights()[0], original_weight / 2)
        self.assertGreater(strategy.selection_weights()[1], 0)
        await self.step(strategy, 100, valid=False)
        self.assertEqual(len(strategy.archive), 2)

    async def test_dgm_can_branch_from_regression_instead_of_global_best(self):
        strategy = await self.make_strategy(DGMArchive)
        child = await self.step(strategy, 0.2)
        strategy.rng.choices = lambda *args, **kwargs: [child]
        pending = await strategy.generate()
        self.assertEqual(pending[0].parent_id, child.id)
        self.assertEqual(strategy.best.id, 0)
