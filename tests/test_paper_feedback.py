"""Optional LLM grading uses real rendered prompts and scripted responses."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from slick import prompts
from slick.providers import ProviderError

from research import alphaevolve
from research.alphaevolve.paper.evaluation import (
    EvaluationResult,
    EvaluationStage,
    evaluate_cascade,
)
from research.alphaevolve.paper.feedback import LLMFeedback
from rsikit.policy import Policy
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program


class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent)
        root.start()
        self.addCleanup(root.stop)
        self.policy = Policy.from_text(program(0).implementation, name="Candidate")
        self.measured = EvaluationResult(
            {"reward": 9}, {"variability": 2}, "Completed every episode", {3: 9}
        )

    async def test_rubric_grades_add_objectives_to_measured_cascade(self):
        provider = ScriptedProvider(
            [
                json.dumps(
                    {
                        "metrics": {"clarity": 0.8},
                        "feedback": "The branches are understandable.",
                        "accepted": True,
                    }
                )
            ]
        )
        grader = LLMFeedback(provider, {"clarity": "Rate clarity from 0 to 1."})

        async def measured(policy):
            return self.measured

        result = await evaluate_cascade(
            self.policy, [EvaluationStage(measured)], feedback_evaluator=grader
        )
        self.assertEqual(result.metrics, {"reward": 9, "clarity": 0.8})
        self.assertEqual(result.features, self.measured.features)
        self.assertEqual(result.seed_scores, {3: 9})
        self.assertEqual(
            result.feedback, "Completed every episode\nThe branches are understandable."
        )
        self.assertTrue(result.accepted)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn('"clarity": 0.8', grader.attempts[0]["raw"])
        rendered = provider.calls[0]
        for text in (
            self.policy._implementation,
            '"reward": 9',
            '"variability": 2',
            "Completed every episode",
            '"3": 9',
            "Rate clarity from 0 to 1.",
            "maximize",
            '"properties"',
        ):
            self.assertIn(text, rendered)

    async def test_rubric_can_reject_a_measured_candidate(self):
        provider = ScriptedProvider(
            [
                json.dumps(
                    {
                        "metrics": {"clarity": 0},
                        "feedback": "Fails the required clarity rubric.",
                        "accepted": False,
                    }
                )
            ]
        )
        result = await LLMFeedback(provider, {"clarity": "Reject opaque implementations."})(
            self.policy, self.measured
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics, {"clarity": 0})
        self.assertIn("Fails", result.feedback)

    async def test_grader_cannot_replace_measured_metrics(self):
        provider = ScriptedProvider([])
        with self.assertRaisesRegex(ValueError, "measured"):
            await LLMFeedback(provider, {"reward": "Give a reward estimate."})(
                self.policy, self.measured
            )
        self.assertEqual(provider.calls, [])

    async def test_unknown_missing_nonfinite_and_malformed_output_propagate_without_retry(self):
        for metrics in ({"other": 1}, {}, {"clarity": float("inf")}):
            with self.subTest(metrics=metrics):
                provider = ScriptedProvider(
                    [
                        json.dumps(
                            {
                                "metrics": metrics,
                                "feedback": "assessment",
                                "accepted": True,
                            }
                        )
                    ]
                )
                with self.assertRaises(ValueError):
                    await LLMFeedback(provider, {"clarity": "Rate clarity."})(
                        self.policy, self.measured
                    )
                self.assertEqual(len(provider.calls), 1)
        provider = ScriptedProvider(["not JSON"])
        with self.assertRaises(ValidationError):
            await LLMFeedback(provider, {"clarity": "Rate clarity."})(self.policy, self.measured)
        self.assertEqual(len(provider.calls), 1)

    async def test_provider_failure_propagates(self):
        provider = ScriptedProvider([ProviderError("offline")])
        with self.assertRaisesRegex(ProviderError, "offline"):
            await LLMFeedback(provider, {"clarity": "Rate clarity."})(self.policy, self.measured)
        self.assertEqual(len(provider.calls), 1)


if __name__ == "__main__":
    unittest.main()
