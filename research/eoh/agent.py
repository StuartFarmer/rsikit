"""Evolve heuristic descriptions and implementations with EoH's five operators."""

import random
from collections.abc import Awaitable, Callable
from statistics import fmean
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator
from slick import Session, prompt
from slick.providers import Provider

from rsikit import Episode, PolicyDefinition
from rsikit.optimization import SequentialOptimizer, search

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OPERATORS = ("E1", "E2", "M1", "M2", "M3")


class Proposal(BaseModel, extra="forbid", frozen=True):
    name: Text
    description: Text
    implementation: str = Field(min_length=1)

    @field_validator("implementation")
    @classmethod
    def nonblank_content(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("candidate content must not be blank")
        return content


class Individual(Proposal):
    id: int
    fitness: float = Field(allow_inf_nan=False)


class CandidateRejected(Exception):
    """The evaluator found an infeasible candidate; consume its attempt and continue."""


class EoH(SequentialOptimizer):
    """Own ranked evolution over arbitrary task content and a caller's evaluator."""

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[list[PolicyDefinition]], Awaitable[dict[str, dict[int, Episode]]]]
        | None = None,
        *,
        maximize: bool = True,
        population_size=20,
        generations=20,
        parents=5,
        seed=0,
        operators=OPERATORS,
        init_attempts=None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        super().__init__()
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.population_size, self.generations, self.parents = population_size, generations, parents
        self.seed, self.operators, self.init_attempts = seed, operators, init_attempts
        self.session = None
        self._best = None
        self.maximize = maximize
        self.on_event = on_event
        self.population = []
        self.score_direction = "Higher" if maximize else "Lower"
        self.attempts: list[dict] = []
        self.history: list[tuple[Individual, ...]] = []
        self.evaluations = 0

    @property
    def best(self):
        if self._best is None:
            return None
        return PolicyDefinition.from_text(
            self._best.implementation, name=self._best.name, description=self._best.description
        )

    def emit(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, *, generated: Proposal) -> Proposal:
        """Generate a fresh idea and candidate."""
        return generated

    @prompt(template="explore_diverse.j2", output_type=Proposal)
    async def explore_diverse(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Explore ideas unlike the selected parents (E1)."""
        return generated

    @prompt(template="explore_shared.j2", output_type=Proposal)
    async def explore_shared(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Explore a new variation of the parents' common idea (E2)."""
        return generated

    @prompt(template="modify_structure.j2", output_type=Proposal)
    async def modify_structure(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Modify a parent's structure (M1)."""
        return generated

    @prompt(template="tune_settings.j2", output_type=Proposal)
    async def tune_settings(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Tune a parent's choices while retaining its structure (M2)."""
        return generated

    @prompt(template="simplify.j2", output_type=Proposal)
    async def simplify(self, parents: list[Individual], *, generated: Proposal) -> Proposal:
        """Remove redundant components from a parent (M3)."""
        return generated

    async def run(
        self,
        population_size=None,
        generations=None,
        parents=None,
        seed=None,
        operators=None,
        init_attempts=None,
        *,
        session: Session | None = None,
    ):
        for key, value in dict(
            population_size=population_size,
            generations=generations,
            parents=parents,
            seed=seed,
            operators=operators,
            init_attempts=init_attempts,
        ).items():
            if value is not None:
                setattr(self, key, value)
        self.session = session
        if self.evaluate is None:
            raise ValueError("run requires an evaluator; otherwise use propose/update")

        async def evaluate(policies):
            try:
                measured = await self.evaluate(policies)
            except CandidateRejected as exc:
                measured = {p.id: {0: Episode(error=str(exc))} for p in policies}
            except Exception as exc:
                self._current[2].update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            return measured

        await search(self, evaluate)
        return self.population

    async def _proposals(self):
        self.rng = random.Random(self.seed)
        attempts = 3 * self.population_size if self.init_attempts is None else self.init_attempts
        self._accepted = []
        for _ in range(attempts):
            policy = await self._attempt("INIT", self.initialize, [], 0)
            if policy is not None:
                yield policy
            if len(self._accepted) == self.population_size:
                break
        if len(self._accepted) != self.population_size:
            raise RuntimeError("could not initialize a full valid population; inspect attempts")
        population = self._select_survivors(self._accepted, self.population_size)
        strategies = {
            "E1": (self.explore_diverse, self.parents),
            "E2": (self.explore_shared, self.parents),
            "M1": (self.modify_structure, 1),
            "M2": (self.tune_settings, 1),
            "M3": (self.simplify, 1),
        }
        for generation in range(1, self.generations + 1):
            self._accepted = []
            for operation in self.operators:
                method, count = strategies[operation]
                for _ in range(len(population)):
                    selected = self._select_parents(population, count)
                    policy = await self._attempt(operation, method, selected, generation)
                    if policy is not None:
                        yield policy
            population = self._select_survivors(population + self._accepted, self.population_size)

    def _accept(self, returns, *, seed_scores=None, error=None, feedback=""):
        policy, proposal, record = self._current
        if error is not None:
            record.update(status="rejected", error=error)
            self.emit("rejected", dict(record))
            return
        fitness = fmean(returns)
        child = Individual(**proposal.model_dump(), id=record["id"], fitness=fitness)
        self._accepted.append(child)
        if self._best is None or (
            fitness > self._best.fitness if self.maximize else fitness < self._best.fitness
        ):
            self._best = child
        record.update(fitness=fitness, status="accepted")
        self.emit("candidate", dict(record, policy=policy.to_text()))

    def _select_parents(self, population: list[Individual], count: int) -> list[Individual]:
        # Official v0.1 prob_rank.py: one-based ranks, sampling with replacement.
        weights = [1 / (rank + len(population)) for rank in range(1, len(population) + 1)]
        return self.rng.choices(population, weights=weights, k=count)

    def _select_survivors(self, candidates: list[Individual], size: int) -> list[Individual]:
        population = sorted(candidates, key=lambda item: item.fitness, reverse=self.maximize)[:size]
        self.history.append(tuple(population))
        self.population = population
        self.emit("population", dict(ids=[row.id for row in population]))
        return population

    async def _attempt(self, operation, propose, selected, generation) -> PolicyDefinition | None:
        record = {
            "id": len(self.attempts) + 1,
            "generation": generation,
            "operation": operation,
            "parents": tuple(item.id for item in selected),
        }
        self.attempts.append(record)
        self.emit("attempt_started", dict(record))

        async def recorded_call(context, **kwargs):
            if self.session is None:
                response, requests = await self.provider.acall(context, **kwargs)
            else:
                # Finish the Session as text before Slick parses the proposal. A
                # malformed JSON response must not leave its conversation pending.
                response, requests = await self.session.arun(context), []
            record["raw_response"] = response
            self.emit(
                "raw_response",
                dict(id=record["id"], operation=operation, prompt=context, response=response),
            )
            return response, requests

        try:
            inputs = (selected,) if selected else ()
            try:
                proposal = await propose(*inputs, provider=SimpleNamespace(acall=recorded_call))
            except ValidationError as exc:
                raise CandidateRejected(f"invalid generated proposal: {exc}") from exc
            record["proposal"] = proposal.model_dump()
            try:
                policy = PolicyDefinition.from_text(
                    proposal.implementation, name=proposal.name, description=proposal.description
                )
                policy.validate()
            except ValueError as exc:
                raise CandidateRejected(str(exc)) from exc
            self.evaluations += 1
            self._current = policy, proposal, record
            return policy
        except CandidateRejected as exc:
            record["status"] = "rejected"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.emit("rejected", dict(record))
            return None
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
