"""Evolve Gymnasium policy classes using evaluated islands and Slick generation.

The island founding/reset policy adapts the official FunSearch program database;
see NOTICE. AlphaEvolve's unpublished database details are explicit local choices.
"""

import asyncio
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider, ProviderError

from ..policy import Policy, _policy_class
from .edits import (
    InvalidCandidate,
    Mutation,
    Program,
    apply_edits,
    check_program,
    check_rewrite,
)


class Guidance(BaseModel, extra="forbid"):
    instruction: str = Field(min_length=1)


class _RecordedProvider:
    """Retain raw responses even when Slick's Pydantic parsing rejects them."""

    def __init__(self, provider, record, key):
        self.provider, self.record, self.key = provider, record, key

    async def acall(self, *args, **kwargs):
        text, calls = await self.provider.acall(*args, **kwargs)
        self.record[self.key] = text
        return text, calls


@dataclass(frozen=True)
class _Candidate:
    policy: type[Policy]
    score: float


@dataclass
class PromptIdea:
    instruction: str
    reward: float = 0.0
    uses: int = 0

    @property
    def score(self) -> float:
        return self.reward / max(1, self.uses)


@dataclass(frozen=True)
class Config:
    islands: int = 4
    inspirations: int = 3
    exploration: float = 0.2
    reset_interval: int = 100
    meta_interval: int = 0
    mode: Literal["diff", "rewrite"] = "diff"
    generation_timeout: float | None = None


class AlphaEvolve:
    """Generate policies in memory; update selection from Run's measured scores.

    Configure Slick's template root once before use. A batch sees only previous
    updates. Generation is sequential, with no hidden retries or evaluation.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        context: str = "",
        ensemble: Sequence[tuple[Provider, float]] = (),
        prompt_variants: Sequence[tuple[str, float]] = (("", 1.0),),
        config: Config = Config(),
        seed: int = 0,
    ):
        self.task, self.context = task, context
        self.models = tuple(ensemble) or ((provider, 1.0),)
        self.variants = tuple(prompt_variants)
        self.config = config
        self.rng = random.Random(seed)
        self.islands: list[_Candidate | None] = [None] * config.islands
        self._best: _Candidate | None = None
        self._pending: dict[str, list[dict]] = {}
        self.prompt_ideas = [PromptIdea("")]
        # ponytail: in-memory attempt history; bound it if searches exceed RAM.
        self.attempts: list[dict] = []
        self.events: list[dict] = []
        self.generation_calls = self.meta_calls = self.completed = 0

    @property
    def best(self) -> type[Policy] | None:
        return None if self._best is None else self._best.policy

    @prompt(template="initialize.j2", output_type=Program)
    async def initialize(self, proposal: int, *, generated: Program) -> Program:
        """Create an initial named policy without a hand-written seed program."""
        return generated

    @prompt(template="mutate.j2", output_type=Mutation)
    async def mutate(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Mutation,
    ) -> Mutation:
        return generated

    @prompt(template="rewrite.j2", output_type=Program)
    async def rewrite(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Program,
    ) -> Program:
        return generated

    @prompt(template="evolve_prompt.j2", output_type=Guidance)
    async def evolve_prompt(
        self,
        parent: _Candidate,
        ideas: list[dict],
        failures: list[dict],
        *,
        generated: Guidance,
    ) -> str:
        return generated.instruction

    async def generate(self, n: int = 1) -> list[type[Policy]]:
        """Return n new proposals. Failure aborts the batch; no automatic retries.

        Proposals are neither executed nor saved. Only a successfully returned
        batch is eligible for update; rejected output stays in attempts.
        """
        records = []
        for _ in range(n):
            records.append(await self._propose())
        for record in records:
            self._pending.setdefault(record["policy"].id, []).append(record)
        return [record["policy"] for record in records]

    def update(self, scores: Mapping[str, float]) -> None:
        """Accept scores keyed by generated policy ID; larger scores are better."""
        # Validate the entire result before changing the archive or consuming pending work.
        for policy_id, score in scores.items():
            self._pending[policy_id]
            if not math.isfinite(score):
                raise ValueError("Policy scores must be finite")
        for policy_id, score in scores.items():
            for record in self._pending.pop(policy_id):
                child = _Candidate(record["policy"], score)
                parent, idea = record["parent"], record["idea"]
                if parent is None:
                    # Initial evaluated proposals can found every island.
                    for island_id in range(len(self.islands)):
                        self._register(child, island_id)
                else:
                    self._register(child, record["island"])
                    improvement = (score - parent.score) / max(1.0, abs(parent.score))
                    idea.reward += max(0.0, improvement)
                record.update(status="evaluated", score=score)
                self.completed += 1
                if self.config.reset_interval and self.completed % self.config.reset_interval == 0:
                    self.reset_islands()

    def sample(self) -> tuple[int, _Candidate, list[_Candidate]]:
        island_id = self.rng.randrange(len(self.islands))
        parent = self.islands[island_id]
        pool = {p.policy.id: p for p in self.islands if p is not None}
        if self.rng.random() < self.config.exploration:
            parent = self.rng.choice(list(pool.values()))
        pool[self._best.policy.id] = self._best
        pool.pop(parent.policy.id, None)
        inspirations = self.rng.sample(
            list(pool.values()), min(len(pool), self.config.inspirations)
        )
        return island_id, parent, inspirations

    async def _propose(self) -> dict:
        attempt_id = len(self.attempts) + 1
        model_id = self.rng.choices(range(len(self.models)), [w for _, w in self.models])[0]
        provider = self.models[model_id][0]
        failures = [
            {key: row[key] for key in ("id", "raw", "error") if key in row}
            for row in self.attempts
            if row.get("error")
        ][-3:]
        record = {"id": attempt_id, "model": model_id, "status": "generating"}
        self.attempts.append(record)
        try:
            parent = idea = None
            island_id = 0
            if self._best is None:
                operation = self.initialize
                arguments = (attempt_id,)
            else:
                island_id, parent, inspirations = self.sample()
                idea = await self._choose_guidance(attempt_id, parent, failures, provider, record)
                idea.uses += 1
                variant = self.rng.choices(
                    [v for v, _ in self.variants], [w for _, w in self.variants]
                )[0]
                guidance = "\n".join((variant, idea.instruction))
                record["guidance"] = guidance
                operation = {"diff": self.mutate, "rewrite": self.rewrite}[self.config.mode]
                arguments = (parent, inspirations, guidance, failures)
            self.generation_calls += 1
            proposal = await asyncio.wait_for(
                operation(*arguments, provider=_RecordedProvider(provider, record, "raw")),
                self.config.generation_timeout,
            )
            if parent is None:
                content = proposal.implementation
            elif self.config.mode == "diff":
                content = apply_edits(parent.policy._implementation, proposal.edits)
            else:
                content = check_rewrite(parent.policy._implementation, proposal.implementation)
            check_program(content)
            policy = _policy_class(proposal.name, content)
            record.update(
                status="generated", policy=policy, parent=parent, island=island_id, idea=idea
            )
            return record
        except (
            InvalidCandidate,
            ValidationError,
            ProviderError,
            TimeoutError,
            asyncio.TimeoutError,
        ) as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            raise
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    async def _choose_guidance(self, attempt_id, parent, failures, provider, record) -> PromptIdea:
        if self.config.meta_interval and attempt_id % self.config.meta_interval == 0:
            self.meta_calls += 1
            ideas = [
                {"instruction": idea.instruction, "score": idea.score, "uses": idea.uses}
                for idea in self.prompt_ideas
            ]
            try:
                raw = await asyncio.wait_for(
                    self.evolve_prompt(
                        parent,
                        ideas,
                        failures,
                        provider=_RecordedProvider(provider, record, "meta_raw"),
                    ),
                    self.config.generation_timeout,
                )
                if not raw.strip():
                    raise InvalidCandidate("Prompt guidance is blank")
                idea = PromptIdea(raw.strip())
                self.prompt_ideas.append(idea)
                return idea
            except (
                InvalidCandidate,
                ValidationError,
                ProviderError,
                TimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                # A failed optional meta call still permits the candidate attempt.
                record["meta_error"] = f"{type(exc).__name__}: {exc}"
        if self.rng.random() < self.config.exploration:
            return self.rng.choice(self.prompt_ideas)
        return max(self.prompt_ideas, key=lambda idea: idea.score)

    def _register(self, candidate: _Candidate, island_id: int) -> None:
        incumbent = self.islands[island_id]
        if incumbent is None or candidate.score > incumbent.score:
            self.islands[island_id] = candidate
        if self._best is None or candidate.score > self._best.score:
            self._best = candidate

    def reset_islands(self) -> None:
        """Reseed the weaker half from surviving champions, as in FunSearch."""
        ranked = list(range(len(self.islands)))
        self.rng.shuffle(ranked)
        ranked.sort(key=lambda i: self.islands[i].score)
        count = len(ranked) // 2
        for island_id in ranked[:count]:
            donor = self.rng.choice(ranked[count:])
            founder = self.islands[donor]
            self.islands[island_id] = founder
            self.events.append(
                {"completed": self.completed, "reset": island_id, "founder": founder.policy.id}
            )
