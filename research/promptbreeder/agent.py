"""Co-evolve policies and mutation instructions through binary tournaments."""

import random
from dataclasses import dataclass, field
from statistics import fmean
from types import SimpleNamespace

from pydantic import BaseModel, Field, ValidationError
from slick import prompt

from rsikit import PolicyDefinition
from rsikit.optimization import SequentialOptimizer
from rsikit.policy import InvalidPolicy

INITIAL_MUTATION = "Improve this policy using its measured episode feedback"


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
class Individual:
    id: int
    policy: PolicyDefinition
    mutation: str
    score: float
    scores: dict[int, float]
    parent_id: int | None = None


@dataclass
class Result:
    population: list[Individual] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)
    best: Individual | None = None
    stop_reason: str = "budget"


class PromptBreeder(SequentialOptimizer):
    """Reduced policy adaptation: first, hyper_zero, and hyper_first operators."""

    operators = ("first", "hyper_zero", "hyper_first")

    def __init__(
        self,
        task,
        provider,
        evaluate=None,
        *,
        population_size=10,
        tournaments=50,
        seed=0,
        on_event=None,
    ):
        super().__init__()
        self.population_size, self.tournaments, self.seed = population_size, tournaments, seed
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.on_event = on_event
        self.result = Result()
        self.model_calls = self.evaluations = 0

    @property
    def best(self):
        return self.result.best.policy if self.result.best else None

    def emit(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    async def call(self, method, *args):
        self.model_calls += 1
        call_id = self.model_calls
        self.emit("call_started", dict(call=call_id, operation=method.__name__))

        async def recorded(context, **kwargs):
            response, calls = await self.provider.acall(context, **kwargs)
            self.emit("raw_response", dict(call=call_id, prompt=context, response=response))
            return response, calls

        return await method(*args, provider=SimpleNamespace(acall=recorded))

    @prompt(template="initialize.j2", output_type=Proposal)
    async def initialize(self, mutation, *, generated: Proposal) -> PolicyDefinition:
        return generated.policy()

    @prompt(template="first.j2", output_type=Proposal)
    async def first(self, parent, mutation, *, generated: Proposal) -> PolicyDefinition:
        return generated.policy()

    @prompt(template="hyper_zero.j2")
    async def hyper_zero(self, *, generated: str) -> str:
        return generated.strip()

    @prompt(template="hyper_first.j2")
    async def hyper_first(self, mutation, *, generated: str) -> str:
        return generated.strip()

    async def attempt(self, parent=None, operator="initialize"):
        record = dict(
            id=len(self.result.attempts),
            operator=operator,
            parent_id=parent.id if parent else None,
            error="",
        )
        self.result.attempts.append(record)
        self.emit("attempt_started", dict(record))
        self._child = None
        mutation = parent.mutation if parent else INITIAL_MUTATION
        if operator == "hyper_zero":
            mutation = await self.call(self.hyper_zero)
        elif operator == "hyper_first":
            mutation = await self.call(self.hyper_first, mutation)
        if not mutation:
            record["error"] = "Empty mutation instruction"
        else:
            try:
                policy = (
                    await self.call(self.first, parent, mutation)
                    if parent
                    else await self.call(self.initialize, mutation)
                )
            except (ValidationError, InvalidPolicy) as exc:
                record["error"] = str(exc)
            else:
                self.evaluations += 1
                self._current = policy, mutation, record
                return policy
        self.emit("rejected", dict(record))
        return None

    def _accept(self, returns, *, seed_scores=None, error=None, feedback=""):
        policy, mutation, record = self._current
        if error is not None:
            record["error"] = error
            self.emit("rejected", dict(record))
            return
        child = Individual(
            record["id"],
            policy,
            mutation,
            fmean(returns),
            dict(seed_scores or {}),
            record["parent_id"],
        )
        self._child = child
        if self.result.best is None or child.score > self.result.best.score:
            self.result.best = child
        record.update(
            policy=policy.to_text(), mutation=mutation, score=child.score, scores=child.scores
        )
        if self._slot is None:
            self.result.population.append(child)
        else:
            self.result.population[self._slot] = child
        self.emit("candidate", dict(record))

    async def _proposals(self):
        self.rng = random.Random(self.seed)
        self._slot = None
        for _ in range(3 * self.population_size):
            policy = await self.attempt()
            if policy is not None:
                yield policy
            if len(self.result.population) == self.population_size:
                break
        if len(self.result.population) != self.population_size:
            raise RuntimeError("Could not initialize a full population")
        while len(self.result.pairs) < self.tournaments:
            order = list(range(len(self.result.population)))
            self.rng.shuffle(order)
            pairs = list(zip(order[::2], order[1::2]))[: self.tournaments - len(self.result.pairs)]
            if not pairs:
                self.result.stop_reason = "no_pairs"
                break
            for pair in pairs:
                self.result.pairs.append(pair)
                winner, loser = sorted(
                    pair, key=lambda i: self.result.population[i].score, reverse=True
                )
                self._slot = loser
                operator = self.rng.choice(self.operators)
                policy = await self.attempt(self.result.population[winner], operator)
                if policy is not None:
                    yield policy
                self.emit(
                    "tournament",
                    dict(
                        pair=list(pair),
                        winner=winner,
                        loser=loser,
                        operator=operator,
                        replacement=self._child.id if self._child else None,
                    ),
                )
        self.emit(
            "finished",
            dict(
                model_calls=self.model_calls,
                evaluations=self.evaluations,
                reason=self.result.stop_reason,
            ),
        )

    async def run(self, *, population_size=None, tournaments=None, seed=None) -> Result:
        if population_size is not None:
            self.population_size = population_size
        if tournaments is not None:
            self.tournaments = tournaments
        if seed is not None:
            self.seed = seed
        await self._run_evaluated()
        return self.result
