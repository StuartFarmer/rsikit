"""The same variation and selection components compose population and island recipes."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from rsikit.tests.providers import ScriptedProvider


class ConceptRecipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_population_and_adaptive_qd_islands(self):
        from rsikit.examples.concepts import demo, responses

        with patch.object(prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[2]):
            for qd in (False, True):
                provider = ScriptedProvider(responses())
                result = await demo(provider, qd=qd)
                self.assertEqual(result["calls"], 6)
                self.assertEqual(len(result["history"]), 7)
                self.assertEqual(len(result["operator_feedback"]), 6)
                self.assertEqual(len(result["island_feedback"]), 6)
                self.assertTrue(all(r["role"] == "operator" for r in result["operator_feedback"]))
                self.assertTrue(all(r["role"] == "island" for r in result["island_feedback"]))
                self.assertFalse(result["history"][5]["evaluation"]["valid"])
                self.assertEqual(result["best"]["evaluation"]["metrics"]["sum_radii"], 1.25)
                crossovers = [r for r in result["allocations"] if r["operation"] == "crossover"]
                self.assertTrue(crossovers)
                self.assertTrue(all(len(set(r["parents"])) == 2 for r in crossovers))
                if qd:
                    self.assertEqual(result["coverage"], [1.0, 1.0])
                    self.assertTrue(result["migration"])
                else:
                    self.assertEqual(result["migration"], [])

    async def test_failed_provider_has_no_invented_reward(self):
        from rsikit import Candidate, PromptProposer
        from rsikit.archives import EliteArchive
        from rsikit.examples.concepts import ConceptSearch, measure
        from rsikit.examples.prompt_search import PACKING
        from rsikit.selection import UCB1, ThompsonSampling

        source = (PACKING / "initial.py").read_text()
        initial = Candidate(id=0, source=source, evaluation=await measure(source))
        islands, operators = UCB1(role="island"), ThompsonSampling(role="operator")
        operations = PromptProposer("task", ScriptedProvider([RuntimeError("offline")]))
        search = ConceptSearch(
            initial,
            operations,
            measure,
            [EliteArchive(4, objective="sum_radii")],
            island_sampler=islands,
            operator_sampler=operators,
            select_parent=lambda population: population[0],
        )
        with patch.object(prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[2]):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await search.run(1)
        self.assertEqual(search.best.source, source)
        self.assertEqual(len(search.history), 1)
        self.assertEqual(islands.counts, {})
        self.assertEqual(operators.observations[-1]["outcome"], "error")
        self.assertIsNone(operators.observations[-1]["reward"])

        with (
            patch.object(prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[2]),
            patch.object(operations.provider, "acall", side_effect=asyncio.CancelledError()),
            self.assertRaises(asyncio.CancelledError),
        ):
            await search.run(1)
        self.assertEqual(operators.observations[-1]["outcome"], "cancelled")
        self.assertEqual(islands.counts, {})

    async def test_qd_grid_covers_valid_geometry_outside_scripted_descriptor_range(self):
        from rsikit import Draft
        from rsikit.examples.concepts import demo, responses
        from rsikit.examples.prompt_search import PACKING

        source = (PACKING / "initial.py").read_text().replace("0.10", "0.01")
        for before, after in (
            ("0.125", "0.025"),
            ("0.375", "0.275"),
            ("0.625", "0.525"),
            ("0.875", "0.775"),
        ):
            source = source.replace(f"({before},", f"({after},")
        script = responses()
        script[0] = Draft(description="small circles on left", source=source)
        with patch.object(prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[2]):
            result = await demo(ScriptedProvider(script), qd=True)
        self.assertTrue(result["history"][1]["evaluation"]["valid"])
        self.assertAlmostEqual(result["history"][1]["evaluation"]["metrics"]["mean_x"], 0.35)
