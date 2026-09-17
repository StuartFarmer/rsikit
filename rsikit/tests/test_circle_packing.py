"""End-to-end packing optimization with scripted model output and fixed geometry."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from slick import prompts

from rsikit.examples.circle_packing import experiment, visualize
from rsikit.examples.circle_packing.evaluate import evaluate, read_circles
from rsikit.sandbox import PythonSandbox
from rsikit.tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1] / "examples/circle_packing"


@unittest.skipUnless(
    os.environ.get("RSIKIT_DOCKER_TESTS") == "1", "Set RSIKIT_DOCKER_TESTS=1 for Docker integration"
)
class CirclePackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_alternative_strategies_use_the_same_packing_experiment(self):
        initial = (ROOT / "initial.py").read_text()
        improved = (
            initial.replace("0.10", "0.125")
            .replace("    return [", "    import math\n    return [")
            .removesuffix("\n")
        )
        for name, responses, expected_calls in [
            ("alphaevolve", [improved], 1),
            ("eoh", [{"description": "Increase grid radii to contact.", "source": improved}], 1),
            ("dgm-archive", ["Unused grid clearance permits radii 0.125.", improved], 2),
        ]:
            with (
                self.subTest(strategy=name),
                tempfile.TemporaryDirectory() as temporary,
                patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
            ):
                responses = [json.dumps(r) if isinstance(r, dict) else r for r in responses]
                provider = ScriptedProvider(responses)
                output = Path(temporary) / name
                strategy = await experiment.run(output, provider, iterations=1, strategy_name=name)
                self.assertAlmostEqual(strategy.best.evaluation.metrics["sum_radii"], 1.25)
                self.assertEqual(len(provider.calls), expected_calls)
                if name == "alphaevolve":
                    measured = (
                        provider.calls[0]
                        .split("<measured_packings>")[1]
                        .split("</measured_packings>")[0]
                    )
                    self.assertEqual(
                        json.loads(measured),
                        [{"id": 0, "circles": json.loads(json.dumps(read_circles(initial)))}],
                    )
                self.assertEqual(
                    json.loads((output / "summary.json").read_text())["strategy"], name
                )
                self.assertEqual(len(json.loads((output / "selections.json").read_text())), 1)
                self.assertTrue((output / "progress.gif").exists())

    async def test_eoh_operator_prompts_and_invalid_structured_response(self):
        initial = (ROOT / "initial.py").read_text()
        responses = [
            json.dumps(
                {
                    "description": operation,
                    "source": initial.replace("0.10", str(0.101 + i * 0.001)),
                }
            )
            for i, operation in enumerate(("E1", "E2", "M1", "M2", "M3"))
        ]
        provider = ScriptedProvider(responses + ["not JSON"])
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
        ):
            strategy = await experiment.run(
                Path(temporary) / "run",
                provider,
                iterations=6,
                strategy_name="eoh",
                population_size=1,
            )
            self.assertEqual(
                [r["operation"] for r in strategy.selections], ["E1", "E2", "M1", "M2", "M3", "E1"]
            )
            for op, call in zip(("E1", "E2", "M1", "M2", "M3"), provider.calls):
                self.assertIn(op + ":", call)
            self.assertFalse(strategy.history[-1].evaluation.valid)

    async def test_optimize_repair_evaluate_and_save_best(self):
        initial = (ROOT / "initial.py").read_text()
        # Real model responses commonly omit the final newline, including repairs.
        invalid = initial.replace("0.10", "0.20").removesuffix("\n")
        improved = """def pack_circles():
    # EVOLVE-BLOCK-START
    import numpy as np
    from scipy.optimize import minimize
    def objective(r):
        return -r[0]
    radius = float(minimize(objective, [0.1], bounds=[(0.1, 0.125)]).x[0])
    return [(float(x), float(y), radius) for y in np.arange(0.125, 0.75, 0.25)
            for x in np.arange(0.125, 1, 0.25)][:10]
    # EVOLVE-BLOCK-END"""
        worse = initial.replace("0.10", "0.11").removesuffix("\n")
        # Repeating a failed version must not buy another execution of that version.
        provider = ScriptedProvider([invalid, invalid, improved, worse])
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
        ):
            output = Path(temporary) / "run"
            make_frame = visualize.make_frame

            def final_frame(*args, **kwargs):
                saved = json.loads((output / "summary.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(saved["completed_attempts"], 2)
                return make_frame(*args, **kwargs)

            execute = PythonSandbox.__call__
            executions = []

            async def counted(sandbox, source, **kwargs):
                executions.append(source)
                return await execute(sandbox, source, **kwargs)

            with (
                patch.object(visualize, "make_frame", side_effect=final_frame) as frames,
                patch.object(PythonSandbox, "__call__", counted),
                patch.object(
                    visualize, "read_circles", side_effect=AssertionError("Use saved output")
                ),
            ):
                strategy = await experiment.run(output, provider, iterations=2)
            self.assertEqual(executions, [initial, invalid, improved, worse])
            self.assertEqual(frames.call_count, 3)
            self.assertAlmostEqual(strategy.best.evaluation.metrics["sum_radii"], 1.25)
            self.assertEqual((output / "best.py").read_text(), improved)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["improved"])
            self.assertAlmostEqual(summary["gain"], 0.25)
            traces = json.loads((output / "repairs.json").read_text())
            self.assertEqual([len(t) for t in traces], [3, 1])
            self.assertIn("circles", provider.calls[1].lower())
            self.assertTrue((output / "best.svg").exists())
            self.assertEqual(
                [p.name for p in sorted((output / "generations").glob("*.svg"))],
                ["0000.svg", "0001.svg", "0002.svg"],
            )
            with Image.open(output / "progress.gif") as animation:
                self.assertEqual(animation.n_frames, 3)
            self.assertTrue((output / "checks/0000/evaluation.json").exists())
            packings = json.loads((output / "packings.json").read_text())
            self.assertAlmostEqual(evaluate(packings["1"])["metrics"]["sum_radii"], 1.25)

    async def test_model_failure_preserves_baseline_and_failed_status(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
        ):
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await experiment.run(
                    output, ScriptedProvider([RuntimeError("provider unavailable")])
                )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")
            self.assertFalse(summary["improved"])
            self.assertTrue(
                evaluate(json.loads((output / "packings.json").read_text())["0"])["valid"]
            )
            self.assertTrue((output / "progress.gif").exists())


class GeometryTests(unittest.TestCase):
    def test_evaluator_checks_returned_data(self):
        circles = read_circles((ROOT / "initial.py").read_text())
        self.assertAlmostEqual(evaluate(circles)["metrics"]["sum_radii"], 1.0)
        for invalid in (
            [],
            None,
            {},
            [[0, 0, 0.1]] * 10,
            [[0.5, 0.5, True]] * 10,
            [[0.5, 0.5, float("nan")]] * 10,
            [[0.5, 0.5, -0.1]] * 10,
            [[0.5, 0.5, 1e309]] * 10,
            [[0.5, 0.5, 0.1]] * 10,
            [[0.5, 0.5]] * 10,
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(evaluate(invalid)["valid"])


class ModularPackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_shinka_filters_before_execution_and_saves_model_history(self):
        executions = []

        class LiteralSandbox:
            timeout, image = 10, "literal-parser"

            async def __call__(self, source):
                executions.append(source)
                return {"value": read_circles(source)}

        async def embed(source):
            return [1.0]

        initial = (ROOT / "initial.py").read_text()
        renamed = initial.replace("0.10", "0.100")
        improved = initial.replace("0.10", "0.125")
        provider = ScriptedProvider(
            [
                renamed,
                '{"novel": false, "reason": "Same packing"}',
                improved,
                '{"novel": true, "reason": "Improved radii"}',
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT.parents[2]),
        ):
            output = Path(temp) / "run"
            strategy = await experiment._run(
                output,
                provider,
                sandbox=LiteralSandbox(),
                iterations=1,
                strategy_name="shinkaevolve",
                shinka_options={"embed": embed, "patch_types": (("full", 1),)},
            )
            self.assertEqual(executions, [initial, improved])
            self.assertAlmostEqual(strategy.best.evaluation.metrics["sum_radii"], 1.25)
            saved = json.loads((output / "shinka.json").read_text())
            self.assertTrue(saved["novelty_enabled"])
            self.assertEqual(len(saved["proposals"]), 2)
            self.assertEqual(len(saved["model_gains"][0]), 1)
            self.assertEqual(json.loads((output / "selections.json").read_text())[0]["model"], 0)

    async def test_modular_generation_reflection_and_saved_evidence_without_docker(self):
        class LiteralSandbox:
            timeout, image = 10, "literal-parser"

            async def __call__(self, source):
                return {"value": read_circles(source)}

        initial = (ROOT / "initial.py").read_text()
        improved = initial.replace("0.10", "0.125")
        for kind in ("hillclimb", "eoh", "alphaevolve", "dgm-archive"):
            responses = (["Increase radii"] if kind == "dgm-archive" else []) + [
                json.dumps({"description": "Use clearance", "source": improved}),
                "Keep larger radii",
            ]
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as temp,
                patch.object(prompts, "TEMPLATE_ROOT", ROOT.parents[2]),
            ):
                output = Path(temp) / "run"
                result = await experiment._run(
                    output,
                    ScriptedProvider(responses),
                    sandbox=LiteralSandbox(),
                    prompt_mode="modular",
                    model_settings={
                        "model": "fixed-api-model",
                        "max_output_tokens": 1024,
                        "temperature": 0.3,
                    },
                    reflect=True,
                    iterations=1,
                    strategy_name=kind,
                )
                self.assertAlmostEqual(result.best.evaluation.metrics["sum_radii"], 1.25)
                self.assertEqual(
                    json.loads((output / "reflections.json").read_text())[0]["text"],
                    "Keep larger radii",
                )
                records = json.loads((output / "proposal_records.json").read_text())
                self.assertEqual(records[0]["parent_ids"], [0])
                self.assertEqual(records[0]["draft"]["source"], improved)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(
                    summary["model_settings"],
                    {"model": "fixed-api-model", "max_output_tokens": 1024, "temperature": 0.3},
                )
