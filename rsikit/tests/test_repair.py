"""Check bounded repair and its boundary with search state."""

import asyncio
import unittest

import rsikit


class RepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_checks_initial_source_and_every_repair_with_exact_budget(self):
        for budget, valid in [(0, False), (1, False), (2, True), (3, True)]:
            with self.subTest(budget=budget):
                checked, repaired = [], []

                async def check(source):
                    checked.append(source)
                    return rsikit.Evaluation(valid=source == "2", feedback=f"checked {source}")

                async def repair(source, evaluation):
                    repaired.append((source, evaluation.feedback))
                    return str(int(source) + 1)

                result = await rsikit.repair_until_valid(
                    "0", check=check, repair=repair, max_repairs=budget
                )
                self.assertEqual(result.valid, valid)
                self.assertEqual(checked, [str(i) for i in range(min(budget, 2) + 1)])
                self.assertEqual(repaired, [(s, f"checked {s}") for s in checked[:-1]])
                self.assertEqual([a.source for a in result.attempts], checked)
                if valid:
                    self.assertEqual(result.require_valid_source(), "2")
                else:
                    with self.assertRaises(rsikit.RepairExhausted) as caught:
                        result.require_valid_source()
                    self.assertIs(caught.exception.result, result)

    async def test_already_valid_source_never_calls_repair(self):
        async def check(source):
            return rsikit.Evaluation(valid=True)

        async def repair(source, evaluation):
            self.fail("valid source must not be repaired")

        result = await rsikit.repair_until_valid("valid", check=check, repair=repair)
        self.assertEqual(result.require_valid_source(), "valid")
        self.assertEqual(len(result.attempts), 1)

    async def test_scope_is_checked_before_testing_every_revision(self):
        seed = "fixed\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
        broken = seed.replace("x = 1", "x =")
        escaped = seed.replace("fixed", "changed").replace("x = 1", "x = 2")
        fixed = seed.replace("x = 1", "x = 2")
        repairs = iter([escaped, fixed])
        checked = []

        async def propose(parent, history):
            return broken

        async def check(source):
            checked.append(source)
            return rsikit.Evaluation(valid=source == fixed, feedback="syntax error")

        async def repair(source, evaluation):
            return next(repairs)

        proposer = rsikit.RepairingProposer(propose, check, repair, max_repairs=2)
        strategy = rsikit.HillClimb(
            seed, rsikit.Evaluation(valid=True, metrics={"score": 0}), proposer
        )
        candidates = await strategy.generate()
        self.assertEqual(candidates[0].source, fixed)
        self.assertEqual(checked, [broken, fixed])
        self.assertEqual([a.source for a in proposer.history[0].attempts], [broken, escaped, fixed])
        self.assertIn("immutable", proposer.history[0].attempts[1].evaluation.feedback)
        await strategy.update(candidates, [rsikit.Evaluation(valid=True, metrics={"score": 1})])
        self.assertEqual(strategy.best.source, fixed)
        self.assertEqual(len(strategy.history), 2)

    async def test_exhaustion_records_failure_without_pending_candidate(self):
        async def propose(parent, history):
            return "broken"

        async def check(source):
            return rsikit.Evaluation(valid=False, feedback="missing entry point")

        async def repair(source, evaluation):
            return "still broken"

        proposer = rsikit.RepairingProposer(propose, check, repair, max_repairs=1)
        strategy = rsikit.HillClimb(
            "seed", rsikit.Evaluation(valid=True, metrics={"score": 1}), proposer
        )
        self.assertEqual(await strategy.generate(), [])
        await strategy.update([], [])
        self.assertIsNone(strategy.pending)
        self.assertEqual(strategy.best.source, "seed")
        self.assertEqual(strategy.history[-1].source, "still broken")
        self.assertFalse(strategy.history[-1].evaluation.valid)
        self.assertIn("missing entry point", strategy.history[-1].evaluation.feedback)
        self.assertEqual(len(proposer.history[0].attempts), 2)
        self.assertEqual(await strategy.generate(), [])
        self.assertEqual([c.id for c in strategy.history], [0, 1, 2])

    async def test_callback_errors_and_cancellation_leave_search_state_unchanged(self):
        for operation in ("check", "repair"):
            for error in (RuntimeError, asyncio.CancelledError):
                with self.subTest(operation=operation, error=error):

                    async def propose(parent, history):
                        return "broken"

                    async def check(source):
                        if operation == "check":
                            raise error("worker failed")
                        return rsikit.Evaluation(valid=False)

                    async def repair(source, evaluation):
                        raise error("worker failed")

                    proposer = rsikit.RepairingProposer(propose, check, repair)
                    strategy = rsikit.HillClimb(
                        "seed", rsikit.Evaluation(valid=True, metrics={"score": 1}), proposer
                    )
                    with self.assertRaises(error):
                        await strategy.generate()
                    self.assertIsNone(strategy.pending)
                    self.assertEqual(len(strategy.history), 1)
                    self.assertEqual(proposer.history, [])
