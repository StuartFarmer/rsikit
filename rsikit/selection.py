"""Selection decisions over measured, valid candidates; recipes own admission timing."""

import math
from collections import Counter

from .strategies import Candidate


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
