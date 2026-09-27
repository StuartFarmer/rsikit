"""Population behavior: diversity, metric specialists, durable history, and validation."""

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from research.alphaevolve.paper.database import Candidate, Database
from rsikit.policy import _policy_class


def candidate(value, score, niche=0, stability=0):
    policy = _policy_class(str(value), f"class Solution:\n    value = {value}\n")
    return Candidate(policy, score, {"reward": score, "stability": stability}, {"x": niche})


class PopulationTests(unittest.TestCase):
    def database(self, **kwargs):
        db = Database(features={"x": (0, 10, 5)}, **kwargs)
        self.addCleanup(db.close)
        return db

    def test_niches_and_metric_specialists_are_sampled(self):
        db = self.database(islands=1, exploration=1)
        for item in [candidate(1, 100, 0, 2), candidate(2, 2, 0, 10), candidate(3, 1, 8)]:
            db.register(item, 0)
        self.assertEqual({c.score for c in db.members(0)}, {100, 2, 1})
        rng = random.Random(4)
        self.assertEqual({db.sample(rng)[1].score for _ in range(100)}, {100, 2, 1})
        self.assertEqual(db.best.score, 100)
        self.assertEqual(db.champions[0].score, 100)

    def test_exploitation_selects_other_metric_winner(self):
        db = self.database(islands=1, exploration=0, elite_fraction=0.2)
        db.register(candidate(1, 100, 0, 1), 0)
        db.register(candidate(2, 2, 0, 100), 0)
        rng = random.Random(1)
        self.assertEqual({db.sample(rng)[1].score for _ in range(100)}, {100, 2})

    def test_history_keeps_displaced_programs_and_cross_island_inspirations(self):
        db = self.database(islands=2)
        db.register(candidate(1, 1, 0, 1), 0)
        db.register(candidate(2, 2, 0, 2), 0)
        db.register(candidate(3, 3, 8, 3), 1)
        self.assertEqual(len(db.all()), 3)
        self.assertEqual(len(db.members(0)), 1)
        _, parent, inspirations = db.sample(random.Random(8), inspirations=3)
        self.assertEqual(len(inspirations), 1)
        self.assertNotEqual(parent.policy.id, inspirations[0].policy.id)

    def test_migration_preserves_unrelated_cells_and_history(self):
        db = self.database(islands=2)
        db.register(candidate(1, 100, 0), 0)
        db.register(candidate(2, 10, 8), 1)
        events = db.migrate(random.Random(7), count=10)
        self.assertTrue(events)
        self.assertEqual({c.score for c in db.members(0)}, {100, 10})
        self.assertEqual({c.score for c in db.members(1)}, {100, 10})
        self.assertEqual(len(db.all()), 2)

    def test_dedup_ignores_comments_names_and_does_not_replace_evaluation(self):
        db = self.database(islands=2)
        original = db.register(candidate(1, 10), 0)
        renamed = _policy_class("new name", "# comment\nclass Solution:\n    value=1\n")
        duplicate = db.register(replace(candidate(1, 999), policy=renamed), 1)
        self.assertEqual(duplicate, original)
        self.assertEqual(duplicate.policy.id, original.policy.id)
        self.assertEqual(duplicate.score, 10)
        self.assertEqual(len(db.all()), 1)
        self.assertEqual(db.members(1)[0].score, 10)

    def test_validation_and_storage_do_not_execute_source(self):
        db = self.database(islands=1)
        policy = _policy_class(
            "untrusted", "raise RuntimeError('never execute')\nclass Solution: pass"
        )
        item = replace(candidate(1, 2), policy=policy)
        db.validate(item, 0)
        self.assertEqual(db.all(), [])
        db.register(item, 0)
        self.assertEqual(db.best.policy._implementation, policy._implementation)

    def test_reopening_restores_history_cells_feedback_seeds_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.sqlite"
            db = Database(path, islands=2, features={"x": (0, 10, 5)})
            original = db.register(replace(candidate(1, 10), feedback="ok", seed_scores={7: 10}), 0)
            db.register(candidate(2, 12, 0, 1), 0, parent_id=original.policy.id)
            db.save_state("prompt", {"text": "try this", "iteration": 2})
            db.close()
            reopened = Database(path, islands=2, features={"x": (0, 10, 5)})
            self.addCleanup(reopened.close)
            self.assertEqual(len(reopened.all()), 2)
            self.assertEqual(reopened.all()[0].seed_scores, {7: 10})
            self.assertEqual(reopened.all()[0].feedback, "ok")
            self.assertEqual(reopened.best.score, 12)
            self.assertEqual(reopened.champions[1], None)
            self.assertEqual(reopened.load_state("prompt"), {"text": "try this", "iteration": 2})
            self.assertEqual(reopened.load_state("absent", 3), 3)
            with self.assertRaises(ValueError):
                Database(path, islands=3, features={"x": (0, 10, 5)})
            with self.assertRaises(ValueError):
                reopened.register(replace(candidate(3, 2), metrics={"reward": 2}), 1)

    def test_descriptor_overflow_clamps_to_edge_cells(self):
        db = self.database(islands=1)
        for item in [candidate(1, 10, -100), candidate(2, 9, 0), candidate(3, 5, 100)]:
            db.register(item, 0)
        self.assertEqual({c.score for c in db.members(0)}, {10, 5})

    def test_rejects_invalid_configuration_and_results(self):
        for kwargs in [
            {"islands": 0},
            {"islands": True},
            {"elite_fraction": 0},
            {"exploration": float("nan")},
            {"features": {"x": (1, 0, 2)}},
            {"features": {"x": (0, 1, 0)}},
            {"features": {"x": (0, 1, 1.5)}},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Database(**kwargs)
        db = self.database(islands=1)
        good = candidate(1, 2)
        for bad in [
            replace(good, score=3),
            replace(good, metrics={"other": 2}),
            replace(good, features={}),
            replace(good, features={"x": 0, "extra": 1}),
            replace(good, features={"x": float("inf")}),
            replace(good, metrics={"reward": 2, "stability": float("nan")}),
            replace(good, seed_scores={1: float("nan")}),
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                db.register(bad, 0)
        self.assertEqual(db.all(), [])
        with self.assertRaises(ValueError):
            db.sample(random.Random(1))
        with self.assertRaises(ValueError):
            db.register(good, 1)
        with self.assertRaises(ValueError):
            db.register(good, 0, parent_id="missing")


if __name__ == "__main__":
    unittest.main()
