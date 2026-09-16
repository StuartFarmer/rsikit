"""Serial search recipes adapted from slick-bits AlphaEvolve, EoH, and DGM.

These strategies optimize artifacts. DGMArchive does not execute self-modifying
agents. See README and NOTICE for the scope and attribution of these adaptations.
"""

import math
import random
from collections import Counter
from collections.abc import Callable

from .strategies import Candidate, SequentialStrategy


class AlphaEvolve(SequentialStrategy):
    """Single-objective island/cell champions, inspirations, and periodic reseeding."""

    def __init__(
        self,
        initial,
        evaluation,
        propose,
        *,
        objective="score",
        maximize=True,
        islands=4,
        cell: Callable[[Candidate], tuple] = lambda candidate: (),
        exploration=0.2,
        inspirations=3,
        reset_interval=20,
        seed=0,
    ):
        super().__init__(initial, evaluation, propose, objective=objective, maximize=maximize)
        self.rng = random.Random(seed)
        self.cell, self.exploration = cell, exploration
        self.inspirations, self.reset_interval = inspirations, reset_interval
        self.cells = {0: cell(self.best)}
        self.islands = [{self.cells[0]: self.best} for _ in range(islands)]
        self.events: list[dict] = []

    def select_parent(self) -> Candidate:
        island = self.rng.randrange(len(self.islands))
        population = list(self.islands[island].values())
        champion = sorted(
            population, key=lambda c: self._score(c.evaluation), reverse=self.maximize
        )[0]
        parent = self.rng.choice(population) if self.rng.random() < self.exploration else champion
        pool = {c.id: c for archive in self.islands for c in archive.values() if c.id != parent.id}
        inspirations = self.rng.sample(list(pool.values()), min(len(pool), self.inspirations))
        self.context = {
            "operation": "alphaevolve",
            "parents": (parent,),
            "inspirations": tuple(inspirations),
            "island": island,
        }
        return parent

    def observe(self, candidate: Candidate) -> None:
        if candidate.evaluation.valid:
            island = self.islands[self.context["island"]]
            key = self.cell(candidate)
            incumbent = island.get(key)
            self.cells[candidate.id] = key
            if incumbent is None or (
                self._score(candidate.evaluation) > self._score(incumbent.evaluation)
                if self.maximize
                else self._score(candidate.evaluation) < self._score(incumbent.evaluation)
            ):
                island[key] = candidate
        completed = len(self.history)
        if self.reset_interval and completed % self.reset_interval == 0:
            self.reset_islands(attempt=completed)

    def reset_islands(self, *, attempt=None) -> None:
        # FunSearch seed-all-islands / weaker-half reseeding; see NOTICE.
        champions = [
            sorted(a.values(), key=lambda c: self._score(c.evaluation), reverse=self.maximize)[0]
            for a in self.islands
        ]
        ranked = list(range(len(self.islands)))
        self.rng.shuffle(ranked)
        ranked.sort(key=lambda i: self._score(champions[i].evaluation), reverse=not self.maximize)
        count = len(ranked) // 2
        for island in ranked[:count]:
            donor = self.rng.choice(ranked[count:])
            founder = champions[donor]
            self.islands[island] = {self.cells[founder.id]: founder}
            self.events.append(
                {
                    "attempt": len(self.history) - 1 if attempt is None else attempt,
                    "island": island,
                    "founder": founder.id,
                }
            )


class EoH(SequentialStrategy):
    """Seeded EoH population with fixed-snapshot operator cycles and elitist selection.

    Fill the population from the seed first. Then emit one offspring per call,
    visiting all five operators N times each before admitting the cycle's elites.
    The global best remains available immediately, even during a partial cycle.
    """

    OPERATORS = ("E1", "E2", "M1", "M2", "M3")

    def __init__(
        self,
        initial,
        evaluation,
        propose,
        *,
        objective="score",
        maximize=True,
        population_size=4,
        parents=3,
        seed=0,
    ):
        super().__init__(initial, evaluation, propose, objective=objective, maximize=maximize)
        self.rng = random.Random(seed)
        self.population_size, self.parent_count = population_size, parents
        self.population = [self.best]
        self.offspring: list[Candidate] = []
        self.cycle_step = 0
        self.cycles = 0

    def select_parent(self) -> Candidate:
        if len(self.population) < self.population_size:
            self.context = {"operation": "INIT", "parents": (self.best,)}
            return self.best
        operation = self.OPERATORS[self.cycle_step // self.population_size]
        count = self.parent_count if operation in ("E1", "E2") else 1
        size = len(self.population)
        weights = [1 / (rank + size) for rank in range(1, size + 1)]
        parents = self.rng.choices(self.population, weights=weights, k=count)
        self.context = {"operation": operation, "parents": tuple(parents)}
        return parents[0]

    def observe(self, candidate: Candidate) -> None:
        if self.context["operation"] == "INIT":
            if candidate.evaluation.valid:
                self.population.append(candidate)
                self.population.sort(key=lambda c: self._score(c.evaluation), reverse=self.maximize)
            return
        if candidate.evaluation.valid:
            self.offspring.append(candidate)
        self.cycle_step += 1
        if self.cycle_step == self.population_size * len(self.OPERATORS):
            self.population = sorted(
                self.population + self.offspring,
                key=lambda c: self._score(c.evaluation),
                reverse=self.maximize,
            )[: self.population_size]
            self.offspring = []
            self.cycle_step = 0
            self.cycles += 1


class DGMArchive(SequentialStrategy):
    """Archive all feasible artifacts; sample by normalized quality and child count.

    This adapts DGM's archive policy, not its executable self-modification. The
    caller's proposer can compose diagnosis and revision. Score bounds normalize
    the objective to [0, 1] for the original sigmoid-based selection weighting.
    """

    def __init__(
        self,
        initial,
        evaluation,
        propose,
        *,
        objective="score",
        maximize=True,
        score_bounds=(0.0, 1.0),
        seed=0,
    ):
        super().__init__(initial, evaluation, propose, objective=objective, maximize=maximize)
        self.rng = random.Random(seed)
        self.score_bounds = score_bounds
        self.archive = [self.best]

    def selection_weights(self) -> list[float]:
        lower, upper = self.score_bounds
        children = Counter(c.parent_id for c in self.archive)
        weights = []
        for candidate in self.archive:
            quality = min(
                1.0, max(0.0, (self._score(candidate.evaluation) - lower) / (upper - lower))
            )
            if not self.maximize:
                quality = 1 - quality
            weights.append(1 / (1 + math.exp(-10 * (quality - 0.5))) / (1 + children[candidate.id]))
        return weights

    def select_parent(self) -> Candidate:
        parent = self.rng.choices(self.archive, weights=self.selection_weights(), k=1)[0]
        self.context = {"operation": "dgm-archive", "parents": (parent,)}
        return parent

    def observe(self, candidate: Candidate) -> None:
        if candidate.evaluation.valid:
            self.archive.append(candidate)
