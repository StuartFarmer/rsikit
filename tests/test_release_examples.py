"""Featured journeys run real workers and save usable winners without model calls."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import fmean

from examples.custom_system import GridSearch
from rsikit import Episode, PolicyDefinition, Run
from tests.helpers import episodes


class ReleaseExamplesTests(unittest.TestCase):
    def test_offline_journeys_save_measured_winners(self):
        for module, count in (("existing_system", 2), ("custom_system", 3)):
            with self.subTest(example=module), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                result = subprocess.run(
                    [sys.executable, "-m", f"examples.{module}", "--output", str(output)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                winner = PolicyDefinition.from_file(output / "best.py")
                winner.validate()
                with Run.open(output) as run:
                    policies = run.policies()
                    self.assertEqual(len(policies), count)
                    self.assertIn(winner, policies)
                    scores = run.scores(winner)
                    self.assertEqual(set(scores), {0, 1, 100, 101})
                    best_mean = max(
                        fmean(run.scores(policy)[seed] for seed in (0, 1)) for policy in policies
                    )
                    self.assertEqual(fmean(scores[s] for s in (0, 1)), best_mean)
                    for seed, score in scores.items():
                        episode = run.load_episode(winner, seed)
                        self.assertIsNone(episode.error)
                        self.assertEqual(episode.total_reward, score)
                        self.assertGreater(score, 0)
                        self.assertLessEqual(score, 100)
                    self.assertIn(f"training_mean={best_mean:.1f}", result.stdout)
                    self.assertIn(
                        f"heldout_mean={fmean(scores[s] for s in (100, 101)):.1f}",
                        result.stdout,
                    )
                    exports = list((output / "exports").glob("*.py"))
                    self.assertEqual(
                        {PolicyDefinition.from_file(p).id for p in exports},
                        {p.id for p in policies},
                    )


class CustomFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_feedback_rejects_failures_and_ranks_fitness(self):
        optimizer = GridSearch()
        candidates = await optimizer.propose()
        with self.assertRaises(RuntimeError):
            await optimizer.propose()
        feedback = {p.id: episodes({0: 2, 1: 2}) for p in candidates}
        with self.assertRaises(ValueError):
            optimizer.update({})
        wrong_panel = dict(feedback)
        wrong_panel[candidates[0].id] = episodes({0: 2})
        with self.assertRaises(ValueError):
            optimizer.update(wrong_panel)
        self.assertFalse(optimizer.done)
        self.assertIsNone(optimizer.best)

        # A huge partial reward cannot beat a complete successful panel.
        feedback[candidates[0].id] = episodes({0: 1000})
        feedback[candidates[0].id][1] = Episode(error="invalid action")
        for episode in feedback[candidates[2].id].values():
            episode.infos[-1]["fitness"] = 3.0
        optimizer.update(feedback)
        self.assertTrue(optimizer.done)
        self.assertEqual(optimizer.best, candidates[2])
        self.assertEqual(optimizer.best_score, 3.0)
        self.assertEqual(await optimizer.propose(), [])
        with self.assertRaises(ValueError):
            optimizer.update(feedback)


if __name__ == "__main__":
    unittest.main()
