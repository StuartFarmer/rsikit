"""Candidate evaluation evidence and a Gym adapter backed by Run."""

import math
from dataclasses import dataclass, field, replace
from numbers import Real
from statistics import fmean, pstdev

from .evaluation import PolicyError


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

    @property
    def scores(self) -> dict[int, float]:
        """Compatibility spelling for the former research Measurement types."""
        return self.seed_scores


async def evaluate_gym(
    run, policies, seeds=(0,), *, features=(), screening_seeds=(), screening_min_reward=None
) -> dict[str, EvaluationResult]:
    """Evaluate through Run, preserving successful siblings and cached episodes.

    Return reward, worst_reward and stability (-population standard deviation).
    Optional descriptors are mean_reward and reward_std. Screening uses a cheap
    seed panel first; rejected/failed candidates skip the full panel. Full results
    contain only the requested evaluation seeds, even when Run has other scores.
    This adapter borrows Run and leaves its lifecycle and sandbox to the caller.
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
        failures = {}
        try:
            await run.evaluate(batch, seeds=panel)
        except PolicyError as exc:
            if not exc.failures or not set(exc.failures) <= {p.id for p in batch}:
                raise
            failures = exc.failures
        results = {}
        for policy in batch:
            if policy.id in failures:
                results[policy.id] = EvaluationResult(failure=failures[policy.id])
                continue
            stored = run.scores(policy)
            scores = _numbers({seed: stored[seed] for seed in panel}, seeds=True)
            mean, std = fmean(scores.values()), pstdev(scores.values())
            descriptors = {"mean_reward": mean, "reward_std": std}
            results[policy.id] = EvaluationResult(
                metrics={"reward": mean, "worst_reward": min(scores.values()), "stability": -std},
                features={name: descriptors[name] for name in features},
                seed_scores=scores,
            )
        return results

    results = {}
    if screening_seeds:
        for policy_id, result in (await measure(policies, screening_seeds)).items():
            if result.failure is not None:
                results[policy_id] = result
            elif result.metrics["reward"] < screening_min_reward:
                results[policy_id] = replace(
                    result,
                    accepted=False,
                    feedback=f"Screening reward below {screening_min_reward}",
                )
        policies = [policy for policy in policies if policy.id not in results]
    results.update(await measure(policies, seeds))
    return results
