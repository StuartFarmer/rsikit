"""Reject incompatible generated constructors without executing candidate code."""

import unittest

from rsikit.generation.edits import InvalidCandidate, check_program


class PolicyContractTests(unittest.TestCase):
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
                    check_program(source)
                else:
                    with self.assertRaisesRegex(InvalidCandidate, "instructions"):
                        check_program(source)
        check_program(
            "class Solution:\n    def __init__(self): pass\n"
            "    def __init__(self, *args, **kwargs): pass\n"
        )


if __name__ == "__main__":
    unittest.main()
