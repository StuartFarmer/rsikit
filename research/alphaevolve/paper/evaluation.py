"""Trusted evaluator callbacks and maximized-threshold cascades (paper §2.4)."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


def _numbers(values, *, seeds=False):
    if not isinstance(values, dict):
        raise ValueError("Evaluation values must be a dictionary")
    for name, value in values.items():
        if seeds:
            valid_name = isinstance(name, int) and not isinstance(name, bool)
        else:
            valid_name = isinstance(name, str) and bool(name.strip())
        if not valid_name:
            raise ValueError("Evaluation names must be nonempty strings; seed IDs must be integers")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("Evaluation values must be finite numbers")
    return dict(values)


@dataclass(frozen=True)
class EvaluationResult:
    """Measured objectives, numeric niche descriptors, and prompt feedback."""

    metrics: dict[str, float]
    features: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    seed_scores: dict[int, float] = field(default_factory=dict)
    accepted: bool = True

    def __post_init__(self):
        object.__setattr__(self, "metrics", _numbers(self.metrics))
        object.__setattr__(self, "features", _numbers(self.features))
        object.__setattr__(self, "seed_scores", _numbers(self.seed_scores, seeds=True))
        if not isinstance(self.feedback, str):
            raise ValueError("Evaluation feedback must be text")
        if not isinstance(self.accepted, bool):
            raise ValueError("Evaluation acceptance must be boolean")


@dataclass(frozen=True)
class EvaluationStage:
    """Run evaluate(policy); each named objective must meet its lower bound."""

    evaluate: Callable[[Any], Awaitable[EvaluationResult]]
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.evaluate):
            raise ValueError("Evaluation stage requires a callable evaluator")
        object.__setattr__(self, "thresholds", _numbers(self.thresholds))


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
        if missing:
            raise ValueError(f"Threshold metrics missing from evaluation: {sorted(missing)}")
        return EvaluationResult(
            metrics=metrics,
            features={**result.features, **next_result.features},
            feedback="\n".join(text for text in (result.feedback, next_result.feedback) if text),
            seed_scores={**result.seed_scores, **next_result.seed_scores},
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
