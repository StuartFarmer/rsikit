"""Trusted evaluator callbacks and maximized-threshold cascades (paper §2.4)."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rsikit.measurements import EvaluationResult


@dataclass(frozen=True)
class EvaluationStage:
    """Run evaluate(policy); each named objective must meet its lower bound."""

    evaluate: Callable[[Any], Awaitable[EvaluationResult]]
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.evaluate):
            raise ValueError("Evaluation stage requires a callable evaluator")
        object.__setattr__(self, "thresholds", EvaluationResult(self.thresholds).metrics)


async def evaluate_cascade(
    policy,
    stages: Sequence[EvaluationStage],
    *,
    feedback_evaluator: Callable[[Any, EvaluationResult], Awaitable[EvaluationResult]]
    | None = None,
) -> EvaluationResult:
    """Prune after cheap checks, then optionally grade (policy, combined_result).

    Later measurements overwrite earlier estimates. Feedback is concatenated;
    a rejection stops the cascade. Evaluator failures propagate unchanged.
    Callers isolate untrusted programs inside their evaluator callbacks.
    """
    stages = tuple(stages)
    if not stages or any(not isinstance(stage, EvaluationStage) for stage in stages):
        raise ValueError("A cascade requires at least one valid EvaluationStage")
    if feedback_evaluator is not None and not callable(feedback_evaluator):
        raise ValueError("Feedback evaluator must be callable")
    result = EvaluationResult({})

    def combine(next_result, thresholds):
        if not isinstance(next_result, EvaluationResult):
            raise ValueError("Evaluators must return EvaluationResult")
        metrics = {**result.metrics, **next_result.metrics}
        missing = thresholds.keys() - metrics.keys()
        if missing and next_result.failure is None:
            raise ValueError(f"Threshold metrics missing from evaluation: {sorted(missing)}")
        return EvaluationResult(
            metrics=metrics,
            features={**result.features, **next_result.features},
            feedback="\n".join(text for text in (result.feedback, next_result.feedback) if text),
            seed_scores={**result.seed_scores, **next_result.seed_scores},
            failure=next_result.failure,
            accepted=next_result.accepted
            and all(metrics[key] >= value for key, value in thresholds.items()),
        )

    for stage in stages:
        result = combine(await stage.evaluate(policy), stage.thresholds)
        if not result.accepted:
            return result
    if feedback_evaluator is not None:
        result = combine(await feedback_evaluator(policy, result), {})
    return result
