"""Ignore only the optional final line ending when enforcing protected edits."""

import unittest

from rsikit import InvalidCandidate, validate_source


class SourceValidationTests(unittest.TestCase):
    def test_optional_final_line_ending_does_not_change_protected_source(self):
        seed = "fixed\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END"
        revised = seed.replace("x = 1", "x = 2")
        for parent_ending in ("", "\n", "\r\n"):
            for ending in ("", "\n", "\r\n"):
                with self.subTest(parent_ending=parent_ending, ending=ending):
                    validate_source(revised + ending, seed + parent_ending)
                    with self.assertRaisesRegex(InvalidCandidate, "unchanged"):
                        validate_source(seed + ending, seed + parent_ending)

    def test_other_protected_changes_still_fail(self):
        seed = "fixed\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
        revised = seed.replace("x = 1", "x = 2")
        for source in (
            revised.replace("fixed", "changed"),
            revised.replace("# EVOLVE-BLOCK-END", " # EVOLVE-BLOCK-END"),
            revised.replace("# EVOLVE-BLOCK-END", "# EVOLVE-BLOCK-END changed"),
            revised.replace("fixed\n", "fixed \n"),
            revised + "\n",
            revised + "extra = 1\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(InvalidCandidate):
                    validate_source(source, seed)
