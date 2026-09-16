"""Exercise the live example with mocked transport and real local evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from slick import prompts
from slick.providers import OpenRouterAPI

from rsikit.examples.sine import openrouter

ROOT = Path(__file__).resolve().parents[1]


class OpenRouterExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_proposal_is_repaired_then_measured(self):
        initial = (ROOT / "examples/sine/initial.py").read_text()
        broken = initial.replace("return x", "return x - x**3 /")
        fixed = initial.replace("return x", "return x - x**3 / 6")
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=source))])
            for source in [broken, fixed]
        ]
        provider = OpenRouterAPI(openrouter.MODEL, max_output_tokens=8192)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
            patch.object(provider, "_asend", new=AsyncMock(side_effect=responses)) as send,
        ):
            output = Path(temporary) / "experiment"
            strategy = await openrouter.run(output, provider, iterations=1)
            self.assertEqual(strategy.best.source, fixed)
            self.assertLess(strategy.best.evaluation.metrics["mse"], 0.00001)
            trace = json.loads((output / "repairs.json").read_text())[0]
            self.assertEqual([entry["evaluation"]["valid"] for entry in trace], [False, True])
            self.assertEqual(send.await_count, 2)
            requests = [call.args[0] for call in send.await_args_list]
            self.assertTrue(all(r["model"] == "openai/gpt-oss-120b:nitro" for r in requests))
            self.assertEqual(requests[0]["max_tokens"], 8192)
            self.assertIn("invalid syntax", requests[1]["messages"][0]["content"])
            self.assertIn(initial, requests[1]["messages"][0]["content"])

    async def test_repair_demo_uses_model_to_fix_supplied_broken_source(self):
        from tests.providers import ScriptedProvider

        initial = (ROOT / "examples/sine/initial.py").read_text()
        fixed = initial.replace("return x", "return x - x**3 / 6")
        provider = ScriptedProvider([fixed])
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(prompts, "TEMPLATE_ROOT", ROOT / "prompts"),
        ):
            strategy = await openrouter.run(
                Path(temporary) / "experiment", provider, iterations=1, repair_demo=True
            )
        self.assertEqual(strategy.best.source, fixed)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("Repair", provider.calls[0])
