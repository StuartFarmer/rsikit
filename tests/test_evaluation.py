"""Shared evaluation outcomes and Gym adaptation through real Run persistence."""

import tempfile
import unittest
from functools import partial
from pathlib import Path

import gymnasium as gym

import rsikit
from research.alphaevolve.paper.evaluation import assess
from rsikit.evaluation import InfrastructureError, PolicyError, episode_error, episode_scores
from tests.helpers import fake_executor
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation

SOURCE = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation): return 0\n"


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(
            hasattr(rsikit.PolicyDefinition, "from_text"), "Expose public policy construction"
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.environment = gym.make("CartPole-v1")
        self.addCleanup(self.environment.close)
        self.evaluation = FakeEvaluation()
        self.run = rsikit.Run.create(
            name="shared-evaluation",
            path=Path(directory.name) / "run",
        )
        self.evaluator = partial(
            self.run.evaluate,
            environment=self.environment,
            executor=fake_executor(evaluation=self.evaluation, concurrency=2),
        )
        self.addCleanup(self.run.close)
        self.good = rsikit.PolicyDefinition.from_text(SOURCE, name="Good")
        self.bad = rsikit.PolicyDefinition.from_text(SOURCE + "# fails\n", name="Bad")

    async def test_policy_failure_preserves_successful_siblings_and_cached_episodes(self):
        async def evaluate(source, environment, seed):
            if source.endswith("# fails\n") and seed == 1:
                raise PolicyError("invalid action")
            return trajectory(float(seed + 2), {})

        self.evaluation.evaluate.side_effect = evaluate
        results = await assess(self.evaluator, [self.good, self.bad], seeds=iter([0, 1, 0]))
        self.assertEqual(episode_scores(results[self.good.id]), {0: 2, 1: 3})
        self.assertIn("invalid action", episode_error(results[self.bad.id]))
        self.assertIsNotNone(episode_error(results[self.bad.id]))
        self.assertEqual(self.run.scores(self.bad), {0: 2, 1: None})
        self.assertIn("invalid action", self.run.load_episode(self.bad, 1).error)
        self.evaluation.evaluate.side_effect = None
        recovered = await assess(self.evaluator, [self.good, self.bad], seeds=[0, 1])
        self.assertEqual(episode_scores(recovered[self.bad.id]), {0: 2, 1: 7})
        self.assertEqual(self.evaluation.evaluate.await_count, 5)

    async def test_screening_handles_failure_rejection_and_success_separately(self):
        low = rsikit.PolicyDefinition.from_text(SOURCE + "# low\n", name="Low")

        async def evaluate(source, environment, seed):
            if source.endswith("# fails\n"):
                raise PolicyError("broken")
            return trajectory(0.0 if source.endswith("# low\n") else float(10 + seed), {})

        self.evaluation.evaluate.side_effect = evaluate
        results = await assess(
            self.evaluator,
            [self.good, self.bad, low],
            seeds=[0, 2],
            features=("mean_reward", "reward_std"),
            screening_seeds=[0],
            screening_min_reward=5,
        )
        self.assertIsNotNone(episode_error(results[self.bad.id]))
        self.assertIsNone(episode_error(results[low.id]))
        self.assertEqual(results[low.id], {})
        self.assertEqual(set(results[self.good.id]), {0, 2})
        self.assertEqual(self.run.scores(low), {0: 0})
        self.assertEqual(self.evaluation.evaluate.await_count, 4)

    async def test_invalid_configuration_and_infrastructure_do_not_become_bad_fitness(self):
        for kwargs in (
            {"seeds": []},
            {"seeds": [True]},
            {"features": ["unknown"]},
            {"screening_seeds": [0]},
            {"screening_min_reward": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await assess(self.evaluator, [self.good], **kwargs)
        self.evaluation.evaluate.side_effect = InfrastructureError("offline")
        with self.assertRaisesRegex(InfrastructureError, "offline"):
            await assess(self.evaluator, [self.good])


if __name__ == "__main__":
    unittest.main()
