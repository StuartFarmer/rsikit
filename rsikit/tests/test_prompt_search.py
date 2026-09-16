"""Instruction promotion depends on complete downstream measurements."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from rsikit import EvaluationError
from rsikit.prompt_search import PromptSearch, PromptTrial
from tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[2]


class PromptSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_paired_promotion_selection_privacy_and_no_weight_update(self):
        measurements = []

        async def evaluate(instruction, cases):
            measurements.append((instruction, cases))
            scores = {"seed": (1, 1), "overfit": (3, 0), "winner": (2, 2)}
            value = scores[instruction][cases == ("select",)]
            return PromptTrial(
                cases, (value,), ("SELECTION_SECRET" if cases == ("select",) else "dev feedback",)
            )

        provider = ScriptedProvider(["overfit", "winner", "winner", "  "])
        search = PromptSearch("Improve an instruction", provider, evaluate)
        result = await search.run("seed", development=("dev",), selection=("select",), revisions=4)
        self.assertEqual(result, "winner")
        self.assertEqual(
            [r["decision"] for r in search.history], ["retain", "promote", "unchanged", "blank"]
        )
        self.assertEqual(
            measurements,
            [
                ("seed", ("dev",)),
                ("seed", ("dev",)),
                ("overfit", ("dev",)),
                ("seed", ("select",)),
                ("overfit", ("select",)),
                ("winner", ("dev",)),
                ("seed", ("dev",)),
                ("winner", ("select",)),
                ("seed", ("select",)),
            ],
        )
        self.assertTrue(all("SELECTION_SECRET" not in call for call in provider.calls))

    async def test_ties_failures_and_invalid_evidence_cannot_promote(self):
        async def evaluate(instruction, cases):
            # Failed second output remains in the mean, not filtered away.
            scores = (2.0, -2.0) if instruction == "bad" else (0.0, 0.0)
            return PromptTrial(cases, scores, ("valid", "failed"))

        search = PromptSearch("Improve", ScriptedProvider(["bad"]), evaluate)
        self.assertEqual(
            await search.run("seed", development=("d0", "d1"), selection=("s0", "s1"), revisions=1),
            "seed",
        )
        self.assertNotIn("selection", search.history[-1])
        for trial in (
            PromptTrial(("wrong",), (1,), ("ok",)),
            PromptTrial(("dev",), (float("nan"),), ("ok",)),
            PromptTrial(("dev",), (1, 2), ("ok",)),
            PromptTrial(("dev",), (1,), ()),
        ):

            async def broken(instruction, cases):
                return trial

            with self.subTest(trial=trial):
                search = PromptSearch("Improve", ScriptedProvider([]), broken)
                with self.assertRaises(EvaluationError):
                    await search.run("seed", development=("dev",), selection=("select",))
                self.assertEqual(search.best, "seed")

        provider = ScriptedProvider([])
        with patch.object(search, "evaluate") as evaluate_callback:
            search.provider = provider
            with self.assertRaisesRegex(EvaluationError, "disjoint"):
                await search.run("seed", development=("same",), selection=("same",))
            evaluate_callback.assert_not_called()
        self.assertEqual(provider.calls, [])

    async def test_partial_comparison_and_capacity_failure_preserve_incumbent(self):
        async def evaluate(instruction, cases):
            if instruction == "child":
                raise RuntimeError("evaluator offline")
            return PromptTrial(cases, (-3.0,), ("negative error utility",))

        search = PromptSearch("Improve", ScriptedProvider(["child"]), evaluate)
        with self.assertRaisesRegex(RuntimeError, "evaluator offline"):
            await search.run("seed", development=("dev",), selection=("select",), revisions=1)
        self.assertEqual(search.best, "seed")
        self.assertIn("incumbent", search.history[0]["development"])
        self.assertEqual(search.history[0]["decision"], "interrupted")

        def no_capacity():
            raise RuntimeError("no capacity")

        search = PromptSearch(
            "Improve", ScriptedProvider(["child"]), evaluate, before_comparison=no_capacity
        )
        with self.assertRaisesRegex(RuntimeError, "no capacity"):
            await search.run("seed", development=("dev",), selection=("select",), revisions=1)
        self.assertEqual(len(search.measurements), 1)
        self.assertEqual(search.best, "seed")

        async def cancelled(instruction, cases):
            raise asyncio.CancelledError()

        search = PromptSearch("Improve", ScriptedProvider([]), cancelled)
        with self.assertRaises(asyncio.CancelledError):
            await search.run("seed", development=("dev",), selection=("select",))
        self.assertIn("CancelledError", search.measurements[0]["error"])

    async def test_offline_example_and_shared_budget(self):
        from rsikit.examples.prompt_search import (
            BudgetedProvider,
            BudgetExhausted,
            PackingTrials,
            demo_responses,
            run,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            raw = ScriptedProvider(demo_responses())
            result = await run(raw, output)
            self.assertEqual([r["decision"] for r in result["comparisons"]], ["retain", "promote"])
            self.assertEqual(result["calls"], len(raw.calls))
            self.assertEqual(result["calls"], 20)
            self.assertEqual(json.loads(output.read_text())["best"], result["best"])
            self.assertEqual(len(result["trials"]), 9)
            self.assertTrue(all(len(t["history"]) == 2 for t in result["trials"]))

            # A completed candidate survives failure of its following reflection.
            raw = ScriptedProvider(demo_responses())
            result = await run(raw, output, max_calls=1)
            self.assertEqual(result["status"], "budget")
            self.assertEqual(result["best"], result["initial_instruction"])
            self.assertEqual(len(raw.calls), 1)
            self.assertEqual(len(result["trials"][0]["history"]), 2)
            self.assertIn("BudgetExhausted", result["trials"][0]["reflections"][0]["error"])

        # Repair uses the same per-trial allowance, checked before dispatch.
        raw = ScriptedProvider(['{"description":"invalid","source":"bad source"}'])
        budget = BudgetedProvider(raw, max_calls=10)
        trials = PackingTrials(budget, max_trial_calls=1)
        with self.assertRaises(BudgetExhausted):
            await trials.evaluate_instruction("seed", ("dev/repeat-0",))
        self.assertEqual(len(raw.calls), 1)
        self.assertIsNone(budget.trial_start)
        self.assertEqual(trials.records[0]["proposals"][-1]["operation"], "repair")

        raw = ScriptedProvider([RuntimeError("transport")])
        budget = BudgetedProvider(raw, max_calls=1)
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await budget.acall("prompt")
        with self.assertRaises(BudgetExhausted):
            await budget.acall("prompt")
        self.assertEqual(budget.calls, 1)
