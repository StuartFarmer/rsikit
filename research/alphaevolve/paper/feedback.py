"""Optional rubric-based LLM objectives after trusted measurements (paper §2.4)."""

import json

from pydantic import BaseModel, FiniteFloat
from slick import parse, render
from slick.providers import Provider

from rsikit.policy import PolicyDefinition

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

    async def __call__(
        self, policy: PolicyDefinition, measured: EvaluationResult
    ) -> EvaluationResult:
        if self.criteria.keys() & measured.metrics.keys():
            raise ValueError("Grader criteria must not replace measured metrics")
        record = {"policy_id": policy.id}
        self.attempts.append(record)
        return await self.assess(policy, measured, provider=self.provider, record=record)

    async def assess(
        self, policy: PolicyDefinition, measured: EvaluationResult, *, provider, record=None
    ) -> EvaluationResult:
        schema = _Assessment.model_json_schema()
        context = render(
            "paper/prompts/feedback.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            policy=policy,
            measured=measured,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, _Assessment)
        if generated.metrics.keys() != self.criteria.keys():
            raise ValueError("Grader metrics must match exactly the rubric criteria")
        if generated.metrics.keys() & measured.metrics.keys():
            raise ValueError("Grader output must not replace measured metrics")
        return EvaluationResult(
            metrics=generated.metrics, feedback=generated.feedback, accepted=generated.accepted
        )
