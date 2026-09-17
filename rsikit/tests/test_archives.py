"""Retention, measured niches and migration are independent of parent selection."""

import unittest

from rsikit import Candidate, Evaluation, EvaluationError


def measured(identifier, score, niche=0):
    return Candidate(
        id=identifier,
        source=str(identifier),
        evaluation=Evaluation(
            valid=True,
            metrics={"score": score, "niche": niche},
        ),
    )


class ArchiveTests(unittest.TestCase):
    def test_retention_policies_and_direction(self):
        from rsikit.archives import EliteArchive, QDArchive, SteppingStoneArchive

        seed, weaker, worse, better = (
            measured(0, 10),
            measured(1, 2, 1),
            measured(2, 1, 1),
            measured(3, 3, 1),
        )
        elite = EliteArchive(1)
        stepping = SteppingStoneArchive()
        qd = QDArchive(lambda c: c.evaluation.metrics["niche"], cell_count=2)
        for archive in (elite, stepping, qd):
            self.assertTrue(archive.add(seed))
            self.assertFalse(archive.add(seed))
        self.assertFalse(elite.add(weaker))
        self.assertTrue(stepping.add(weaker))
        self.assertTrue(qd.add(weaker))
        self.assertFalse(qd.add(worse))
        self.assertTrue(qd.add(better))
        self.assertEqual([c.id for c in elite.candidates], [0])
        self.assertEqual([c.id for c in stepping.candidates], [0, 1])
        self.assertEqual([c.id for c in qd.candidates], [0, 3])
        self.assertEqual(qd.coverage, 1.0)
        minimum = EliteArchive(1, maximize=False)
        minimum.add(seed)
        minimum.add(weaker)
        self.assertEqual(minimum.candidates, (weaker,))
        self.assertFalse(minimum.add(measured(4, 2)))  # stable tie
        invalid = Candidate(id=5, source="bad", evaluation=Evaluation(valid=False))
        for archive in (elite, stepping, qd):
            self.assertFalse(archive.add(invalid))
            with self.assertRaises(EvaluationError):
                archive.add(Candidate(id=6, source="unmeasured"))
            with self.assertRaises(EvaluationError):
                archive.add(Candidate(id=7, source="missing", evaluation=Evaluation(valid=True)))

    def test_grid_uses_measured_features_and_failed_descriptor_does_not_mutate(self):
        from rsikit.archives import FeatureGrid, QDArchive

        grid = FeatureGrid(bounds=((0, 1), (-1, 1)), bins=(2, 2))
        self.assertEqual(grid.size, 4)
        self.assertEqual(FeatureGrid(bounds=((-1e308, 1e308),), bins=(2,)).locate((0,)), (1,))
        for features, cell in (((0, -1), (0, 0)), ((1, 1), (1, 1)), ((0.5, 0), (1, 1))):
            self.assertEqual(grid.locate(features), cell)
        for features in ((-0.1, 0), (0, 1.1), (0,), (float("nan"), 0)):
            with self.assertRaises(EvaluationError):
                grid.locate(features)
        qd = QDArchive(
            lambda c: grid.locate((c.evaluation.metrics["niche"], 0)), cell_count=grid.size
        )
        seed = measured(0, 1, 0)
        qd.add(seed)
        with self.assertRaises(EvaluationError):
            qd.add(measured(1, 100, 2))
        self.assertEqual(qd.candidates, (seed,))
        self.assertEqual(qd.coverage, 0.25)

    def test_migration_uses_snapshot_and_destination_admission(self):
        from rsikit.archives import EliteArchive, SteppingStoneArchive
        from rsikit.islands import migrate

        islands = [SteppingStoneArchive() for _ in range(3)]
        for index, island in enumerate(islands):
            island.add(measured(index, index))
        events = migrate(islands, ((0, 1), (1, 2), (2, 0)), select=lambda candidates: candidates)
        self.assertEqual([[c.id for c in a.candidates] for a in islands], [[0, 2], [1, 0], [2, 1]])
        self.assertEqual([e["candidate_id"] for e in events], [0, 1, 2])
        self.assertTrue(all(e["admitted"] for e in events))
        elite = EliteArchive(1)
        elite.add(measured(10, 10))
        events = migrate([islands[0], elite], ((0, 1),), select=lambda candidates: candidates)
        self.assertEqual(elite.candidates[0].id, 10)
        self.assertTrue(all(not e["admitted"] for e in events))
