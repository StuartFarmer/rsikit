"""Reject incompatible generated constructors without executing candidate code."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rsikit
from rsikit.policy import InvalidPolicy, validate_policy


class PolicyContractTests(unittest.TestCase):
    def test_loading_preserves_source_and_validation_is_explicit(self):
        self.assertTrue(hasattr(rsikit.Policy, "from_text"), "Policy owns source loading")
        source = (
            "raise AssertionError('must not execute')\n"
            "from rsikit import Policy\nclass Solution(Policy):\n"
            "    async def act(self, observation): return 0\n"
        )
        policy = rsikit.Policy.from_text(source, name="Example", description="Description")
        self.assertTrue(issubclass(policy, rsikit.Policy))
        self.assertEqual((policy.name, policy.description), ("Example", "Description"))
        self.assertEqual(policy._implementation, source)
        self.assertEqual(policy.id, rsikit.Policy.from_text(source, name="Example").id)
        validate_policy(policy)
        for invalid in (
            "pass",
            "def broken(:",
            source + "class Solution: pass\n",
            source.replace("async def act(self, observation)", "def __init__(self)"),
        ):
            with self.subTest(source=invalid):
                candidate = rsikit.Policy.from_text(invalid, name="Invalid")
                self.assertEqual(candidate._implementation, invalid)
                restored = rsikit.Policy.from_text(candidate.to_text())
                self.assertEqual(restored.id, candidate.id)
                with self.assertRaises(InvalidPolicy):
                    validate_policy(candidate)

    def test_policy_validation_does_not_interpret_evolution_markers(self):
        policy = rsikit.Policy.from_text(
            "# EVOLVE-BLOCK-START\nfrom rsikit import Policy\n"
            "class Solution(Policy):\n    async def act(self, observation): return 0\n"
        )
        validate_policy(policy)

    def test_text_and_file_round_trips_preserve_identity_and_source(self):
        self.assertTrue(hasattr(rsikit.Policy, "from_text"), "Policy owns source loading")
        source = (
            "raise AssertionError('must not execute')\r\n"
            "from rsikit import Policy\r\nclass Solution(Policy):\r\n"
            "    async def act(self, observation): return 0\r\n"
        )
        policy = rsikit.Policy.from_text(source, name="Café\nsolution", description='A "baseline".')
        text = policy.to_text()
        compile(text, "solution.py", "exec")  # Serialized files remain ordinary Python.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.py"
            policy.to_file(path)
            for restored in (rsikit.Policy.from_text(text), rsikit.Policy.from_file(path)):
                self.assertEqual(restored.id, policy.id)
                self.assertEqual(restored.name, policy.name)
                self.assertEqual(restored.description, policy.description)
                self.assertEqual(restored._implementation, source)
                self.assertEqual(restored.to_text(), text)
            # Hand-written Python files use the same loader and explicit metadata.
            path.write_text("class Solution: pass\n", encoding="utf-8")
            initial = rsikit.Policy.from_file(path, name="Initial")
            self.assertEqual(initial.name, "Initial")
            self.assertEqual(initial._implementation, "class Solution: pass\n")

    def test_load_rejects_invalid_metadata_and_missing_files(self):
        self.assertTrue(hasattr(rsikit.Policy, "from_text"), "Policy owns source loading")
        for metadata in ("null", '{"name": 1, "description": ""}', '{"name": "x"}', "not json"):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                rsikit.Policy.from_text(f"# rsikit-policy: {metadata}\nclass Solution: pass\n")
        header = '# rsikit-policy: {"name": "Example", "description": ""}\n'
        with self.assertRaises(ValueError):
            rsikit.Policy.from_text(header + header + "class Solution: pass\n")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                rsikit.Policy.from_file(Path(directory) / "missing.py")

    def test_failed_save_keeps_the_previous_file_and_cleans_up(self):
        policy = rsikit.Policy.from_text("class Solution: pass\n", name="New")
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
                    validate_policy(rsikit.Policy.from_text(source))
                else:
                    with self.assertRaisesRegex(InvalidPolicy, "instructions"):
                        validate_policy(rsikit.Policy.from_text(source))
        validate_policy(
            rsikit.Policy.from_text(
                "class Solution:\n    def __init__(self): pass\n"
                "    def __init__(self, *args, **kwargs): pass\n"
            )
        )


if __name__ == "__main__":
    unittest.main()
