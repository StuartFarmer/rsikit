"""Optional rubric-based LLM objectives after trusted measurements (paper §2.4)."""

from pydantic import BaseModel, FiniteFloat
from slick import prompt
from slick.providers import Provider

from rsikit.generation import RecordingProvider
from rsikit.policy import Policy

from .evaluation import EvaluationResult


class _Assessment(BaseModel, extra="forbid"):
    metrics: dict[str, FiniteFloat]
    feedback: str
    accepted: bool


class LLMFeedback:
    """Add rubric objectives without replacing measurements; failures do not retry.

    Pass this instance as evaluate_cascade's feedback_evaluator. Configure Slick's
    template root as for the optimizer. Raw model responses stay in attempts,
    including responses that fail parsing or rubric validation.
    """

    def __init__(self, provider: Provider, criteria: dict[str, str]):
        self.provider = provider
        self.criteria = dict(criteria)
        self.attempts: list[dict] = []

    async def __call__(self, policy: type[Policy], measured: EvaluationResult) -> EvaluationResult:
        if self.criteria.keys() & measured.metrics.keys():
            raise ValueError("Grader criteria must not replace measured metrics")
        record = {"policy_id": policy.id}
        self.attempts.append(record)
        return await self.assess(
            policy, measured, provider=RecordingProvider(self.provider, record, "raw")
        )

    @prompt(template="paper/prompts/feedback.j2", output_type=_Assessment)
    async def assess(
        self, policy: type[Policy], measured: EvaluationResult, *, generated: _Assessment
    ) -> EvaluationResult:
        if generated.metrics.keys() != self.criteria.keys():
            raise ValueError("Grader metrics must match exactly the rubric criteria")
        if generated.metrics.keys() & measured.metrics.keys():
            raise ValueError("Grader output must not replace measured metrics")
        return EvaluationResult(
            metrics=generated.metrics, feedback=generated.feedback, accepted=generated.accepted
        )
