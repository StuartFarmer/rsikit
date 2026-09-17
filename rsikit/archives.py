"""Measured candidate retention, independent of selection and population topology."""

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

from .evaluation import EvaluationError
from .selection import better, top_candidates
from .strategies import Candidate


def _eligible(candidate: Candidate, objective: str, incumbents: Sequence[Candidate]) -> bool:
    evaluation = candidate.evaluation
    if evaluation is None:
        raise EvaluationError("Archive admission requires a completed evaluation")
    if not evaluation.valid:
        return False
    if objective not in evaluation.metrics or not math.isfinite(evaluation.metrics[objective]):
        raise EvaluationError(f"Archive requires finite objective {objective!r}")
    # ponytail: linear ID scan; add an index if archive size makes admission costly.
    for incumbent in incumbents:
        if incumbent.id == candidate.id:
            if incumbent != candidate:
                raise EvaluationError("Candidate ID already identifies different evidence")
            return False
    return True


class EliteArchive:
    """Keep up to capacity best candidates, retaining incumbents on score ties."""

    def __init__(self, capacity: int, *, objective: str = "score", maximize: bool = True):
        self.capacity, self.objective, self.maximize = capacity, objective, maximize
        self.candidates: tuple[Candidate, ...] = ()

    def add(self, candidate: Candidate) -> bool:
        if not _eligible(candidate, self.objective, self.candidates):
            return False
        self.candidates = tuple(
            top_candidates(
                (*self.candidates, candidate),
                self.capacity,
                objective=self.objective,
                maximize=self.maximize,
            )
        )
        return candidate in self.candidates


class SteppingStoneArchive:
    """Retain every valid, measured candidate, including regressions; no size bound."""

    def __init__(self, *, objective: str = "score"):
        self.objective = objective
        self.candidates: tuple[Candidate, ...] = ()

    def add(self, candidate: Candidate) -> bool:
        if not _eligible(candidate, self.objective, self.candidates):
            return False
        self.candidates = (*self.candidates, candidate)
        return True


class QDArchive:
    """Keep one quality champion per measured cell; lower global quality can enter.

    The callback computes a hashable cell from completed evidence. Descriptor errors
    propagate before mutation. cell_count is the finite descriptor-space capacity;
    omit it when unknown. Coverage is then None. Candidate IDs are run-global.
    """

    def __init__(
        self,
        cell: Callable[[Candidate], Hashable],
        *,
        objective: str = "score",
        maximize: bool = True,
        cell_count: int | None = None,
    ):
        self.cell, self.objective, self.maximize = cell, objective, maximize
        self.cell_count = cell_count
        self.elites: dict[Hashable, Candidate] = {}

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self.elites.values())

    @property
    def coverage(self) -> float | None:
        return len(self.elites) / self.cell_count if self.cell_count is not None else None

    def add(self, candidate: Candidate) -> bool:
        if not _eligible(candidate, self.objective, self.candidates):
            return False
        cell = self.cell(candidate)
        incumbent = self.elites.get(cell)
        if incumbent is not None and not better(
            candidate, incumbent, objective=self.objective, maximize=self.maximize
        ):
            return False
        self.elites[cell] = candidate
        return True


@dataclass(frozen=True)
class FeatureGrid:
    """Equal-width bins over closed bounds; the upper endpoint belongs to the last bin."""

    bounds: tuple[tuple[float, float], ...]
    bins: tuple[int, ...]

    @property
    def size(self) -> int:
        return math.prod(self.bins)

    def locate(self, features: Sequence[float]) -> tuple[int, ...]:
        if len(features) != len(self.bounds):
            raise EvaluationError("Descriptor dimension differs from feature grid")
        cell = []
        for value, (lower, upper), count in zip(features, self.bounds, self.bins, strict=True):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise EvaluationError("Measured descriptor is non-finite or outside grid bounds")
            span = upper - lower
            position = (
                (value - lower) / span
                if math.isfinite(span)
                else (value / 2 - lower / 2) / (upper / 2 - lower / 2)
            )
            cell.append(min(count - 1, int(position * count)))
        return tuple(cell)
