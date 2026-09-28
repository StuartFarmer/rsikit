"""Shared evaluation outcomes and Gym adaptation through real Run persistence."""

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym

import rsikit
from research.alphaevolve.paper.evaluation import assess
from research.rollouts import Rollouts
from rsikit.evaluation import InfrastructureError, PolicyError
from tests.test_episode_storage import trajectory
from tests.test_run import FakeSandbox

SOURCE = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation): return 0\n"


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(hasattr(rsikit.Policy, "from_text"), "Expose public policy construction")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.environment = gym.make("CartPole-v1")
        self.addCleanup(self.environment.close)
        self.sandbox = FakeSandbox()
        self.run = rsikit.Run.create(
            name="shared-evaluation",
            path=Path(directory.name) / "run",
        )
        self.rollouts = Rollouts(
            self.environment, rsikit.Executor(sandbox=self.sandbox, concurrency=2), self.run
        )
        self.addCleanup(self.run.close)
        self.good = rsikit.Policy.from_text(SOURCE, name="Good")
        self.bad = rsikit.Policy.from_text(SOURCE + "# fails\n", name="Bad")

    def test_outcomes_validate_evidence_and_preserve_legacy_constructors(self):
        from research.alphaevolve.paper import EvaluationResult
        from research.elitesearch import Measurement as EliteMeasurement
        from research.lineagesearch import Measurement as LineageMeasurement

        good = EliteMeasurement({0: 3})
        failed = EliteMeasurement({}, failure="bad action")
        feedback = LineageMeasurement({0: 3}, "useful feedback")
        rejected = EvaluationResult({"reward": 1}, accepted=False)
        self.assertNotIsInstance(good, EvaluationResult)
        self.assertEqual(good.scores, {0: 3})
        self.assertFalse(failed.accepted)
        self.assertEqual(failed.failure, "bad action")
        self.assertEqual(feedback.feedback, "useful feedback")
        self.assertIsNone(rejected.failure)
        for kwargs in (
            {"metrics": {"reward": float("nan")}},
            {"seed_scores": {True: 3}},
            {"seed_scores": {0: float("inf")}},
            {"failure": ""},
            {"failure": 42},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationResult(**kwargs)

    async def test_policy_failure_preserves_successful_siblings_and_cached_episodes(self):
        async def evaluate(source, environment, seed, timeout):
            if source.endswith("# fails\n") and seed == 1:
                raise PolicyError("invalid action")
            return trajectory(float(seed + 2), {})

        self.sandbox.evaluate.side_effect = evaluate
        results = await assess(self.rollouts, [self.good, self.bad], seeds=iter([0, 1, 0]))
        self.assertEqual(results[self.good.id].seed_scores, {0: 2, 1: 3})
        self.assertEqual(
            results[self.good.id].metrics, {"reward": 2.5, "worst_reward": 2, "stability": -0.5}
        )
        self.assertIn("invalid action", results[self.bad.id].failure)
        self.assertFalse(results[self.bad.id].accepted)
        self.assertEqual(results[self.bad.id].metrics, {})
        self.assertEqual(self.run.scores(self.bad), {0: 2, 1: None})
        self.sandbox.evaluate.side_effect = None
        recovered = await assess(self.rollouts, [self.good, self.bad], seeds=[0, 1])
        self.assertEqual(recovered[self.bad.id].seed_scores, {0: 2, 1: 7})
        self.assertEqual(self.sandbox.evaluate.await_count, 5)

    async def test_screening_handles_failure_rejection_and_success_separately(self):
        low = rsikit.Policy.from_text(SOURCE + "# low\n", name="Low")

        async def evaluate(source, environment, seed, timeout):
            if source.endswith("# fails\n"):
                raise PolicyError("broken")
            return trajectory(0.0 if source.endswith("# low\n") else float(10 + seed), {})

        self.sandbox.evaluate.side_effect = evaluate
        results = await assess(
            self.rollouts,
            [self.good, self.bad, low],
            seeds=[0, 2],
            features=("mean_reward", "reward_std"),
            screening_seeds=[0],
            screening_min_reward=5,
        )
        self.assertIsNotNone(results[self.bad.id].failure)
        self.assertIsNone(results[low.id].failure)
        self.assertFalse(results[low.id].accepted)
        self.assertEqual(results[self.good.id].features, {"mean_reward": 11, "reward_std": 1})
        self.assertEqual(self.run.scores(low), {0: 0})
        self.assertEqual(self.sandbox.evaluate.await_count, 4)

    async def test_invalid_configuration_and_infrastructure_do_not_become_bad_fitness(self):
        for kwargs in (
            {"seeds": []},
            {"seeds": [True]},
            {"features": ["unknown"]},
            {"screening_seeds": [0]},
            {"screening_min_reward": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await assess(self.rollouts, [self.good], **kwargs)
        self.sandbox.start.assert_not_awaited()
        self.sandbox.evaluate.side_effect = InfrastructureError("offline")
        with self.assertRaisesRegex(InfrastructureError, "offline"):
            await assess(self.rollouts, [self.good])


if __name__ == "__main__":
    unittest.main()
