"""Trusted evaluator callbacks and maximized-threshold cascades (paper §2.4)."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import field
from numbers import Real
from statistics import fmean
from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field, StrictInt, TypeAdapter
from pydantic.dataclasses import dataclass

from rsikit import Episode
from rsikit.evaluation import episode_error, episode_scores


def _real_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Evaluation values must be real numbers")
    return value


NonemptyText = Annotated[str, Field(strict=True, pattern=r"\S")]
FiniteNumber = Annotated[
    float, BeforeValidator(_real_number), Field(strict=True, allow_inf_nan=False)
]
NamedValues = Annotated[dict[NonemptyText, FiniteNumber], Field(strict=True)]
SeedScores = Annotated[dict[StrictInt, FiniteNumber], Field(strict=True)]
NAMED_VALUES = TypeAdapter(NamedValues)


@dataclass(config=ConfigDict(strict=True, extra="forbid"))
class EvaluationResult:
    """Measured evidence, screening rejection, or candidate execution failure.

    A failure always sets accepted=False. Infrastructure/provider errors raise
    instead of becoming results. Metrics use maximization; features are numeric
    descriptors. Callers choose their objective. Empty evidence is allowed for failed or
    intermediate evaluations; optimizers validate their required measurements.
    """

    metrics: NamedValues = field(default_factory=dict)
    features: NamedValues = field(default_factory=dict)
    feedback: str = ""
    seed_scores: SeedScores = field(default_factory=dict)
    accepted: bool = True
    failure: NonemptyText | None = None

    def __post_init__(self):
        if self.failure is not None:
            self.accepted = False


@dataclass(config=ConfigDict(strict=True, extra="forbid"))
class EvaluationStage:
    """Run evaluate(policy); each named objective must meet its lower bound."""

    evaluate: Callable[[Any], Awaitable[EvaluationResult]]
    thresholds: NamedValues = field(default_factory=dict)


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
    stages = tuple(EvaluationStage(**vars(stage)) for stage in stages)
    if feedback_evaluator is not None and not callable(feedback_evaluator):
        raise ValueError("Feedback evaluator must be callable")
    result = EvaluationResult(accepted=False)

    def combine(next_result, thresholds):
        if not isinstance(next_result, EvaluationResult):
            raise ValueError("Evaluators must return EvaluationResult")
        next_result = EvaluationResult(**vars(next_result))
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


async def assess(
    evaluator, policies, seeds=(0,), *, features=(), screening_seeds=(), screening_min_reward=None
) -> dict[str, dict[int, Episode]]:
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
        NAMED_VALUES.validate_python({"screening_min_reward": screening_min_reward})
    unknown = set(features) - {"mean_reward", "reward_std"}
    if unknown:
        raise ValueError(f"Unsupported Gym descriptors: {', '.join(sorted(unknown))}")
    policies = list({policy.id: policy for policy in policies}.values())

    async def measure(batch, panel):
        if not batch:
            return {}
        measured = await evaluator(batch, seeds=panel)
        return measured

    results = {}
    if screening_seeds:
        for policy_id, result in (await measure(policies, screening_seeds)).items():
            if episode_error(result) is not None:
                results[policy_id] = result
            elif fmean(episode_scores(result).values()) < screening_min_reward:
                results[policy_id] = {}
        policies = [policy for policy in policies if policy.id not in results]
    results.update(await measure(policies, seeds))
    return results
