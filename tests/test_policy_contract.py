"""Reject incompatible generated constructors without executing candidate code."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rsikit
from rsikit.policy import InvalidPolicy


class PolicyContractTests(unittest.TestCase):
    def test_definition_validates_fields_and_preserves_identity(self):
        self.assertTrue(hasattr(rsikit, "PolicyDefinition"))
        definition = rsikit.PolicyDefinition(source="pass", name="Example")
        for values in ({"source": 1}, {"name": " \n"}, {"description": None}, {"extra": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                rsikit.PolicyDefinition(**({"source": "pass"} | values))
        for name, value in (("source", "changed"), ("name", "changed")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setattr(definition, name, value)
        self.assertEqual(
            definition.id,
            rsikit.PolicyDefinition(source="pass", name="Example", description="new").id,
        )

    def test_loading_preserves_source_and_validation_is_explicit(self):
        self.assertTrue(hasattr(rsikit, "PolicyDefinition"))
        source = (
            "raise AssertionError('must not execute')\n"
            "from rsikit import Policy\nclass Solution(Policy):\n"
            "    async def act(self, observation): return 0\n"
        )
        policy = rsikit.PolicyDefinition.from_text(
            source, name="Example", description="Description"
        )
        self.assertIsInstance(policy, rsikit.PolicyDefinition)
        self.assertEqual((policy.name, policy.description), ("Example", "Description"))
        self.assertEqual(policy.source, source)
        self.assertEqual(policy.id, rsikit.PolicyDefinition.from_text(source, name="Example").id)
        policy.validate()
        for invalid in (
            "pass",
            "def broken(:",
            source + "class Solution: pass\n",
            source.replace("async def act(self, observation)", "def __init__(self)"),
        ):
            with self.subTest(source=invalid):
                candidate = rsikit.PolicyDefinition.from_text(invalid, name="Invalid")
                self.assertEqual(candidate.source, invalid)
                restored = rsikit.PolicyDefinition.from_text(candidate.to_text())
                self.assertEqual(restored.id, candidate.id)
                with self.assertRaises(InvalidPolicy):
                    candidate.validate()

    def test_policy_validation_does_not_interpret_evolution_markers(self):
        policy = rsikit.PolicyDefinition.from_text(
            "# EVOLVE-BLOCK-START\nfrom rsikit import Policy\n"
            "class Solution(Policy):\n    async def act(self, observation): return 0\n"
        )
        policy.validate()

    def test_text_and_file_round_trips_preserve_identity_and_source(self):
        self.assertTrue(hasattr(rsikit, "PolicyDefinition"))
        source = (
            "raise AssertionError('must not execute')\r\n"
            "from rsikit import Policy\r\nclass Solution(Policy):\r\n"
            "    async def act(self, observation): return 0\r\n"
        )
        policy = rsikit.PolicyDefinition.from_text(
            source, name="Café\nsolution", description='A "baseline".'
        )
        text = policy.to_text()
        compile(text, "solution.py", "exec")  # Serialized files remain ordinary Python.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.py"
            policy.to_file(path)
            for restored in (
                rsikit.PolicyDefinition.from_text(text),
                rsikit.PolicyDefinition.from_file(path),
            ):
                self.assertEqual(restored.id, policy.id)
                self.assertEqual(restored.name, policy.name)
                self.assertEqual(restored.description, policy.description)
                self.assertEqual(restored.source, source)
                self.assertEqual(restored.to_text(), text)
            # Hand-written Python files use the same loader and explicit metadata.
            path.write_text("class Solution: pass\n", encoding="utf-8")
            initial = rsikit.PolicyDefinition.from_file(path, name="Initial")
            self.assertEqual(initial.name, "Initial")
            self.assertEqual(initial.source, "class Solution: pass\n")

    def test_load_rejects_invalid_metadata_and_missing_files(self):
        self.assertTrue(hasattr(rsikit, "PolicyDefinition"))
        for metadata in ("null", '{"name": 1, "description": ""}', '{"name": "x"}', "not json"):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                rsikit.PolicyDefinition.from_text(
                    f"# rsikit-policy: {metadata}\nclass Solution: pass\n"
                )
        header = '# rsikit-policy: {"name": "Example", "description": ""}\n'
        with self.assertRaises(ValueError):
            rsikit.PolicyDefinition.from_text(header + header + "class Solution: pass\n")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                rsikit.PolicyDefinition.from_file(Path(directory) / "missing.py")

    def test_legacy_serialization_and_metadata_overrides(self):
        saved = (
            '# rsikit-policy: {"name": "Example", "description": "Saved"}\nclass Solution: pass\r\n'
        )
        definition = rsikit.PolicyDefinition.from_text(saved)
        self.assertEqual(
            definition.id, "60f7aa2e6b4b2577abce8e0dd11970eb04866920e5c659e791d41bfc809b9232"
        )
        self.assertEqual(definition.to_text(), saved)
        self.assertEqual(definition.source, "class Solution: pass\r\n")
        described = rsikit.PolicyDefinition.from_text(saved, description="Revised")
        self.assertEqual(described.description, "Revised")
        self.assertEqual(described.id, definition.id)
        renamed = rsikit.PolicyDefinition.from_text(saved, name="Renamed")
        self.assertEqual(renamed.name, "Renamed")
        self.assertNotEqual(renamed.id, definition.id)

    def test_failed_save_keeps_the_previous_file_and_cleans_up(self):
        policy = rsikit.PolicyDefinition.from_text("class Solution: pass\n", name="New")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.py"
            path.write_text("previous solution", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("cannot replace")):
                with self.assertRaisesRegex(OSError, "cannot replace"):
                    policy.to_file(path)
            self.assertEqual(path.read_text(), "previous solution")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_constructor_accepts_the_actual_worker_call(self):
        valid = (
            "self, observation_space, action_space, *, instructions=''",
            "self, observation_space, action_space, instructions=''",
            "self, *args, **kwargs",
            "self, observation_space, action_space, /, **kwargs",
            "self, observation_space, action_space, *, instructions, optional=1",
        )
        invalid = (
            "self",
            "self, instructions=None",
            "self, observation_space, action_space, rng",
            "self, *, observation_space, action_space, instructions=''",
            "self, observation_space, action_space",
            "self, observation_space, action_space, *, instructions, required",
            "self, observation_space, observation_space, **kwargs",
        )
        for signature in (*valid, *invalid):
            source = (
                "raise AssertionError('must not execute')\n"
                "from rsikit import Policy\nclass Solution(Policy):\n"
                f"    def __init__({signature}):\n        pass\n"
                "    async def act(self, observation):\n        return 0\n"
            )
            with self.subTest(signature=signature):
                if signature in valid:
                    rsikit.PolicyDefinition.from_text(source).validate()
                else:
                    with self.assertRaisesRegex(InvalidPolicy, "instructions"):
                        rsikit.PolicyDefinition.from_text(source).validate()
        rsikit.PolicyDefinition.from_text(
            "class Solution:\n    def __init__(self): pass\n"
            "    def __init__(self, *args, **kwargs): pass\n"
        ).validate()


if __name__ == "__main__":
    unittest.main()
