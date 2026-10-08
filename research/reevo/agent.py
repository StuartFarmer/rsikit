"""Reflective evolution of policies, adapted from Slick Bits and ai4co/reevo."""

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from statistics import fmean
from types import SimpleNamespace

from pydantic import BaseModel, Field, ValidationError
from slick import Provider, prompt

from rsikit import Episode, PolicyDefinition
from rsikit.optimization import SequentialOptimizer
from rsikit.policy import InvalidPolicy


class Proposal(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    implementation: str = Field(min_length=1)

    def policy(self):
        try:
            policy = PolicyDefinition.from_text(
                self.implementation, name=self.name, description=self.description
            )
        except ValueError as exc:
            raise InvalidPolicy(str(exc)) from exc
        policy.validate()
        return policy


@dataclass(frozen=True)
class Config:
    population_size: int = 10
    initial_size: int = 30
    max_evaluations: int = 100
    crossover_rate: float = 1.0
    mutation_rate: float = 0.5
    short_reflection: bool = True
    long_reflection: bool = True
    seed: int = 0


@dataclass
class Individual:
    id: int
    stage: str
    generation: int
    parents: list[int] = field(default_factory=list)
    policy: PolicyDefinition | None = None
    score: float | None = None
    error: str = ""

    @property
    def candidate(self):
        return self.policy.to_text() if self.policy else ""


@dataclass
class Result:
    individuals: list[Individual] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    best: Individual | None = None
    stop_reason: str = ""


class ReEvo(SequentialOptimizer):
    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[PolicyDefinition]], Awaitable[dict[str, dict[int, Episode]]]]
        | None = None,
        *,
        config: Config | None = None,
        seed_policy: PolicyDefinition | None = None,
        reflector_provider: Provider | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        super().__init__()
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.config = config or Config()
        self.seed_policy = seed_policy
        self.reflector_provider = reflector_provider or provider
        self.on_event = on_event
        self.rng = random.Random(self.config.seed)
        self.result = Result()
        self.population = []
        self.long_term = self.initial_reflection = ""
        self.model_calls = 0

    @property
    def best(self):
        return self.result.best.policy if self.result.best else None

    @property
    def seed_text(self):
        return self.seed_policy.to_text() if self.seed_policy else ""

    @property
    def remaining(self):
        return self.config.max_evaluations - len(self.result.individuals)

    def emit(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    async def call(self, operation, *args, reflection=False):
        self.model_calls += 1
        record = dict(call=self.model_calls, operation=operation.__name__)
        self.emit("call_started", record)

        async def recorded(context, **kwargs):
            provider = self.reflector_provider if reflection else self.provider
            response, tools = await provider.acall(context, **kwargs)
            self.emit("raw_response", dict(record, prompt=context, response=response))
            return response, tools

        return await operation(*args, provider=SimpleNamespace(acall=recorded))

    @prompt(template="initial.j2", output_type=Proposal)
    async def initial(self, *, generated: Proposal) -> PolicyDefinition:
        return generated.policy()

    @prompt(template="crossover.j2", output_type=Proposal)
    async def crossover(
        self, worse, better, reflection, *, generated: Proposal
    ) -> PolicyDefinition:
        return generated.policy()

    @prompt(template="mutate.j2", output_type=Proposal)
    async def mutate(self, elite, reflection, *, generated: Proposal) -> PolicyDefinition:
        return generated.policy()

    @prompt(template="reflect_pair.j2")
    async def reflect_pair(self, worse, better, *, generated: str) -> str:
        return generated

    @prompt(template="reflect_long.j2")
    async def reflect_long(self, prior, insights, *, generated: str) -> str:
        return generated

    async def attempt(self, operation, *args, stage, generation, parents=(), policy=None):
        row = Individual(len(self.result.individuals), stage, generation, list(parents))
        self.result.individuals.append(row)
        self.emit(
            "attempt_started",
            dict(id=row.id, stage=stage, generation=generation, parents=row.parents),
        )
        try:
            row.policy = policy if policy is not None else await self.call(operation, *args)
            row.policy.validate()
        except (ValidationError, InvalidPolicy) as exc:
            row.error = str(exc)
        if row.error:
            self.record_candidate(row)
        self._current = row
        return row

    def record_candidate(self, row):
        self.emit(
            "candidate",
            dict(
                id=row.id,
                policy=row.candidate,
                score=row.score,
                error=row.error,
                parents=row.parents,
            ),
        )

    def _accept(self, returns, *, seed_scores=None, error=None, feedback=""):
        row = self._current
        row.error = error or ""
        if error is None:
            row.score = fmean(returns)
            if self.result.best is None or row.score > self.result.best.score:
                self.result.best = row
        self.record_candidate(row)

    def select_parents(self):
        valid = [row for row in self.population if row.score is not None]
        if self.result.best is not None and self.result.best not in valid:
            valid.append(self.result.best)
        if not valid:
            self.result.stop_reason = "no_valid_individuals"
            return []
        count = min(int(self.config.population_size * self.config.crossover_rate), self.remaining)
        # ponytail: quadratic pairs fit small populations; sample score groups at thousands.
        pairs = [(a, b) for a, b in combinations(valid, 2) if a.score != b.score]
        if count and not pairs:
            self.result.stop_reason = "no_distinct_parents"
            return []
        return [sorted(self.rng.choice(pairs), key=lambda row: row.score) for _ in range(count)]

    async def _proposals(self):
        if self.seed_policy is not None and self.remaining > 0:
            row = await self.attempt(None, stage="seed", generation=0, policy=self.seed_policy)
            if not row.error:
                yield row.policy
            if row.error:
                self.result.stop_reason = "invalid_seed"
                return
        for _ in range(min(self.config.initial_size, self.remaining)):
            row = await self.attempt(self.initial, stage="initial", generation=0)
            self.population.append(row)
            if not row.error:
                yield row.policy
        generation = 0
        while self.remaining > 0 and not self.result.stop_reason:
            generation += 1
            selected = self.select_parents()
            if self.result.stop_reason:
                break
            insights = [
                await self.call(self.reflect_pair, worse, better, reflection=True)
                if self.config.short_reflection
                else ""
                for worse, better in selected
            ]
            offspring = []
            for (worse, better), insight in zip(selected, insights):
                row = await self.attempt(
                    self.crossover,
                    worse,
                    better,
                    insight,
                    stage="crossover",
                    generation=generation,
                    parents=(worse.id, better.id),
                )
                offspring.append(row)
                if not row.error:
                    yield row.policy
            if self.config.long_reflection and any(insights):
                reflection = await self.call(
                    self.reflect_long, self.long_term, insights, reflection=True
                )
                self.long_term = " ".join(reflection.split()[:49])
            record = dict(
                generation=generation,
                pairs=[[a.id, b.id] for a, b in selected],
                short_term=insights,
                long_term=self.long_term,
            )
            self.result.reflections.append(record)
            self.emit("reflection", record)
            elite = self.result.best
            for _ in range(
                min(int(self.config.population_size * self.config.mutation_rate), self.remaining)
            ):
                row = await self.attempt(
                    self.mutate,
                    elite,
                    self.long_term,
                    stage="mutation",
                    generation=generation,
                    parents=(elite.id,),
                )
                offspring.append(row)
                if not row.error:
                    yield row.policy
            if not offspring:
                self.result.stop_reason = "no_offspring"
            self.population = offspring
        self.result.stop_reason = self.result.stop_reason or "budget"
        self.emit("finished", dict(reason=self.result.stop_reason, model_calls=self.model_calls))

    async def run(self) -> Result:
        await self._run_evaluated()
        return self.result
