"""Trusted evaluator callbacks and maximized-threshold cascades (paper §2.4)."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from statistics import fmean
from typing import Any

from research.rewards import measure_rewards
from rsikit import Measurement


def _numbers(values, *, seeds=False):
    if not isinstance(values, dict):
        raise ValueError("Evaluation values must be a dictionary")
    for name, value in values.items():
        valid_name = (
            isinstance(name, int) and not isinstance(name, bool)
            if seeds
            else isinstance(name, str) and bool(name.strip())
        )
        if not valid_name:
            raise ValueError("Evaluation names must be nonempty strings; seed IDs must be integers")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(
                "Measurements must contain finite per-seed scores"
                if seeds
                else "Evaluation values must be finite numbers"
            )
    return dict(values)


@dataclass(frozen=True)
class EvaluationResult:
    """Measured evidence, screening rejection, or candidate execution failure.

    A failure always sets accepted=False. Infrastructure/provider errors raise
    instead of becoming results. Metrics use maximization; features are numeric
    descriptors. Callers choose their objective. Empty evidence is allowed for failed or
    intermediate evaluations; optimizers validate their required measurements.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    seed_scores: dict[int, float] = field(default_factory=dict)
    accepted: bool = True
    failure: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "metrics", _numbers(self.metrics))
        object.__setattr__(self, "features", _numbers(self.features))
        object.__setattr__(self, "seed_scores", _numbers(self.seed_scores, seeds=True))
        if not isinstance(self.feedback, str):
            raise ValueError("Evaluation feedback must be text")
        if not isinstance(self.accepted, bool):
            raise ValueError("Evaluation acceptance must be boolean")
        if self.failure is not None:
            if not isinstance(self.failure, str) or not self.failure.strip():
                raise ValueError("Evaluation failure must be nonempty text")
            object.__setattr__(self, "accepted", False)


@dataclass(frozen=True)
class EvaluationStage:
    """Run evaluate(policy); each named objective must meet its lower bound."""

    evaluate: Callable[[Any], Awaitable[Measurement]]
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.evaluate):
            raise ValueError("Evaluation stage requires a callable evaluator")
        object.__setattr__(
            self, "thresholds", Measurement(metrics=self.thresholds, accepted=False).metrics
        )


async def evaluate_cascade(
    policy,
    stages: Sequence[EvaluationStage],
    *,
    feedback_evaluator: Callable[[Any, Measurement], Awaitable[Measurement]] | None = None,
) -> Measurement:
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
    result = Measurement(accepted=False)

    def combine(next_result, thresholds):
        if not isinstance(next_result, Measurement):
            raise ValueError("Evaluators must return Measurement")
        metrics = {**result.metrics, **next_result.metrics}
        missing = thresholds.keys() - metrics.keys()
        if missing and next_result.failure is None:
            raise ValueError(f"Threshold metrics missing from evaluation: {sorted(missing)}")
        return Measurement(
            metrics=metrics,
            features={**result.features, **next_result.features},
            feedback="\n".join(text for text in (result.feedback, next_result.feedback) if text),
            scores={**result.scores, **next_result.scores},
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


async def assess(
    rollouts, policies, seeds=(0,), *, features=(), screening_seeds=(), screening_min_reward=None
) -> dict[str, Measurement]:
    """Return per-seed evidence; the optimizer derives objectives and descriptors.

    The features argument validates the requested reward-derived descriptors.
    Screening uses a cheap
    seed panel first; rejected/failed candidates skip the full panel. Full results
    contain only the requested evaluation seeds, even when Run has other scores.
    The caller owns rollout execution and run persistence.
    """
    seeds, screening_seeds, features = tuple(seeds), tuple(screening_seeds), tuple(features)
    if not seeds:
        raise ValueError("Evaluation requires at least one seed")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in (*seeds, *screening_seeds)
    ):
        raise ValueError("Evaluation seed IDs must be integers")
    if bool(screening_seeds) != (screening_min_reward is not None):
        raise ValueError("screening_seeds and screening_min_reward must be supplied together")
    if screening_min_reward is not None:
        _numbers({"screening_min_reward": screening_min_reward})
    unknown = set(features) - {"mean_reward", "reward_std"}
    if unknown:
        raise ValueError(f"Unsupported Gym descriptors: {', '.join(sorted(unknown))}")
    policies = list({policy.id: policy for policy in policies}.values())

    async def measure(batch, panel):
        if not batch:
            return {}
        measured = await measure_rewards(rollouts, batch, panel)
        return measured

    results = {}
    if screening_seeds:
        for policy_id, result in (await measure(policies, screening_seeds)).items():
            if result.failure is not None:
                results[policy_id] = result
            elif fmean(result.scores.values()) < screening_min_reward:
                results[policy_id] = replace(
                    result,
                    accepted=False,
                    feedback=f"Screening reward below {screening_min_reward}",
                )
        policies = [policy for policy in policies if policy.id not in results]
    results.update(await measure(policies, seeds))
    return results
