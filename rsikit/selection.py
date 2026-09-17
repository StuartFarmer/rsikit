"""Direct sampling and adaptive allocation; recipes own eligibility and reward."""

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

from .strategies import Candidate

T = TypeVar("T")
Arm = TypeVar("Arm", bound=Hashable)


def uniform(options: Sequence[T], *, rng: random.Random) -> T:
    return rng.choice(options)


def weighted(options: Sequence[T], weights: Sequence[float], *, rng: random.Random) -> T:
    """Draw one option; the caller supplies nonnegative weights with positive total."""
    return rng.choices(options, weights=weights, k=1)[0]


def tournament(
    options: Sequence[T],
    *,
    key: Callable[[T], float],
    rng: random.Random,
    size: int = 3,
    maximize: bool = True,
) -> T:
    """Uniformly sample distinct entrants; best wins, sampled order breaks ties."""
    entrants = rng.sample(list(options), min(size, len(options)))
    return (max if maximize else min)(entrants, key=key)


def softmax(
    options: Sequence[T],
    *,
    key: Callable[[T], float],
    rng: random.Random,
    temperature: float = 1.0,
    maximize: bool = True,
) -> T:
    """Sample finite measured values; subtract the maximum before exponentiation."""
    values = [(1 if maximize else -1) * key(option) for option in options]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Softmax needs finite measured values")
    best = max(values)
    weights = []
    for value in values:
        difference = value - best
        scaled = (
            difference / temperature
            if math.isfinite(difference)
            else value / temperature - best / temperature
        )
        weights.append(math.exp(scaled))
    return weighted(options, weights, rng=rng)


def epsilon_greedy(
    options: Sequence[T],
    *,
    key: Callable[[T], float],
    rng: random.Random,
    epsilon: float = 0.1,
    maximize: bool = True,
) -> T:
    if rng.random() < epsilon:
        return uniform(options, rng=rng)
    return (max if maximize else min)(options, key=key)


class UCB1:
    """Choose by mean reward plus exploration * sqrt(log(total) / count).

    Rewards are measured utilities in [0, 1]. Unobserved eligible arms are chosen
    first; ties use the owned seeded RNG. choose() does not reserve an in-flight
    pull. Callers own scheduling, arm IDs, reward definitions and exactly-once
    attribution. Use a separate instance per role; no state resets between choices.
    """

    def __init__(
        self, *, role: str = "selection", seed: int = 0, exploration: float = math.sqrt(2)
    ):
        self.role, self.exploration = role, exploration
        self.rng = random.Random(seed)
        self.counts: Counter[Hashable] = Counter()
        self.means: dict[Hashable, float] = {}
        self.observations: list[dict] = []

    def choose(self, arms: Sequence[Arm]) -> Arm:
        unseen = [arm for arm in arms if self.counts[arm] == 0]
        if unseen:
            return self.rng.choice(unseen)
        total = sum(self.counts.values())
        values = [
            self.means[a] + self.exploration * math.sqrt(math.log(total) / self.counts[a])
            for a in arms
        ]
        best = max(values)
        return self.rng.choice([arm for arm, value in zip(arms, values) if value == best])

    def observe(
        self,
        arm: Hashable,
        reward: float | None,
        *,
        attempt_id: int | str,
        outcome: str = "evaluated",
        baseline: float | None = None,
    ) -> None:
        """Record attribution; None preserves an unscored event without a pull update."""
        if reward is not None:
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("UCB1 reward must be finite and in [0, 1]")
            count = self.counts[arm] + 1
            mean = self.means.get(arm, 0.0)
            self.means[arm] = mean + (reward - mean) / count
            self.counts[arm] = count
        self.observations.append(
            dict(
                role=self.role,
                arm=arm,
                reward=reward,
                attempt_id=attempt_id,
                outcome=outcome,
                baseline=baseline,
            )
        )


class ThompsonSampling:
    """Beta-Bernoulli posterior sampling for binary success, with a Beta(1,1) default.

    Supply reward 1 for the recipe's declared success, 0 for failure, or None for
    an unscored event. Continuous fitness needs a different likelihood or an
    explicit binary success rule. choose() never updates the posterior. Eligible
    arms may change; stable IDs retain observations when they become eligible again.
    """

    def __init__(
        self,
        *,
        role: str = "selection",
        seed: int = 0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.role, self.alpha, self.beta = role, alpha, beta
        self.rng = random.Random(seed)
        self.successes: Counter[Hashable] = Counter()
        self.failures: Counter[Hashable] = Counter()
        self.observations: list[dict] = []

    def choose(self, arms: Sequence[Arm]) -> Arm:
        return max(
            arms,
            key=lambda a: self.rng.betavariate(
                self.alpha + self.successes[a], self.beta + self.failures[a]
            ),
        )

    def observe(
        self,
        arm: Hashable,
        reward: float | None,
        *,
        attempt_id: int | str,
        outcome: str = "evaluated",
        baseline: float | None = None,
    ) -> None:
        if reward is not None:
            if reward not in (0, 1):
                raise ValueError("Beta-Bernoulli Thompson sampling requires binary reward")
            self.successes[arm] += reward
            self.failures[arm] += 1 - reward
        self.observations.append(
            dict(
                role=self.role,
                arm=arm,
                reward=reward,
                attempt_id=attempt_id,
                outcome=outcome,
                baseline=baseline,
            )
        )


def better(candidate: Candidate, incumbent: Candidate, *, objective: str, maximize=True) -> bool:
    left = candidate.evaluation.metrics[objective]
    right = incumbent.evaluation.metrics[objective]
    return left > right if maximize else left < right


def top_candidates(candidates, count: int, *, objective: str, maximize=True) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.evaluation.metrics[objective], reverse=maximize)[
        :count
    ]


def rank_parents(population, count: int, *, rng) -> list[Candidate]:
    size = len(population)
    return rng.choices(
        population, weights=[1 / (rank + size) for rank in range(1, size + 1)], k=count
    )


def lineage_weights(
    archive, *, objective: str, maximize=True, score_bounds=(0.0, 1.0)
) -> list[float]:
    lower, upper = score_bounds
    children = Counter(c.parent_id for c in archive)
    weights = []
    for candidate in archive:
        score = candidate.evaluation.metrics[objective]
        quality = min(1.0, max(0.0, (score - lower) / (upper - lower)))
        if not maximize:
            quality = 1 - quality
        weights.append(1 / (1 + math.exp(-10 * (quality - 0.5))) / (1 + children[candidate.id]))
    return weights
