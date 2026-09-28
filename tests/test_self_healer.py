"""A repair component works independently of optimizer state."""

import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from slick import prompts

from research import alphaevolve
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program


class SelfHealerTests(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_repair_keeps_raw_failures_and_task_context(self):
        from research.alphaevolve.original.healing import SelfHealer

        provider = ScriptedProvider(["malformed JSON", program(1)])
        healer = SelfHealer("Balance the pole", provider, context="Discrete left/right actions")
        first, second = {}, {}
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent):
            with self.assertRaises(ValidationError):
                await healer.repair("", "broken code", "invalid Python", record=first)
            repaired = await healer.repair("", first["raw"], "invalid JSON", record=second)
        self.assertEqual(first["raw"], "malformed JSON")
        self.assertEqual(repaired.implementation, program(1).implementation)
        self.assertIn("Balance the pole", provider.calls[0])
        self.assertIn("Discrete left/right actions", provider.calls[0])
        self.assertIn("malformed JSON", provider.calls[1])
        self.assertIn("implementation", second["raw"])
