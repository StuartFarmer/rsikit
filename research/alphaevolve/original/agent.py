"""Evolve Gymnasium policy classes using evaluated islands and Slick generation.

The island founding/reset policy adapts the official FunSearch program database;
see NOTICE. AlphaEvolve's unpublished database details are explicit local choices.
"""

import asyncio
import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider, ProviderError

from rsikit.generation import WORKER_LIBRARIES, RecordingProvider
from rsikit.generation.edits import (
    InvalidCandidate,
    Mutation,
    Program,
    apply_edits,
    check_program,
    check_rewrite,
)
from rsikit.policy import Policy, _policy_class

logger = logging.getLogger(__name__)


class Guidance(BaseModel, extra="forbid"):
    instruction: str = Field(min_length=1)


@dataclass(frozen=True)
class _Candidate:
    policy: type[Policy]
    score: float
    seed_scores: dict[int, float] = field(default_factory=dict)


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
    max_repairs: int = 2


class AlphaEvolve:
    """Generate policies in memory; update selection from Run's measured scores.

    Configure Slick's template root once before use. A batch sees only previous
    updates. Proposals run concurrently, with bounded repair and no evaluation.
    """

    libraries = WORKER_LIBRARIES

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
        self._attempt_offset = 0
        self._prior_failures: list[dict] = []
        self.events: list[dict] = []
        self.generation_calls = self.repair_calls = self.meta_calls = self.completed = 0

    @property
    def best(self) -> type[Policy] | None:
        return None if self._best is None else self._best.policy

    @prompt(template="original/prompts/initialize.j2", output_type=Program)
    async def initialize(self, proposal: int, *, generated: Program) -> Program:
        """Create an initial named policy without a hand-written seed program."""
        return generated

    @prompt(template="original/prompts/mutate.j2", output_type=Mutation)
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

    @prompt(template="original/prompts/rewrite.j2", output_type=Program)
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

    @prompt(template="original/prompts/repair.j2", output_type=Program)
    async def fix(
        self, reference: str, failed: str, diagnostic: str, *, generated: Program
    ) -> Program:
        """Repair a full policy from validation or sandbox diagnostics."""
        return generated

    async def _repair_valid(self, record, reference, failed, diagnostic) -> type[Policy]:
        # Adapt main's check/repair/recheck loop; count all repairs for this proposal.
        repairs = record.setdefault("repairs", [])
        provider = self.models[record["model"]][0]
        while len(repairs) < self.config.max_repairs:
            call = {"input": failed, "diagnostic": diagnostic}
            repairs.append(call)
            self.repair_calls += 1
            logger.warning(
                "Repairing proposal %s (%s/%s): %s",
                record["id"],
                len(repairs),
                self.config.max_repairs,
                diagnostic,
            )
            try:
                proposal = await asyncio.wait_for(
                    self.fix(
                        reference,
                        failed,
                        diagnostic,
                        provider=RecordingProvider(provider, call, "raw"),
                    ),
                    self.config.generation_timeout,
                )
                content = proposal.implementation
                call["implementation"] = content
                if content == failed:
                    raise InvalidCandidate("Repair returned the unchanged implementation")
                if reference:
                    check_rewrite(reference, content)
                check_program(content)
                call["valid"] = True
                return _policy_class(proposal.name, content, proposal.description)
            except (InvalidCandidate, ValidationError) as exc:
                if isinstance(exc, ValidationError) and "raw" not in call:
                    raise  # Provider-side failures do not establish invalid model output.
                failed = call.get("implementation", call.get("raw", failed))
                diagnostic = f"{type(exc).__name__}: {exc}"
                call.update(valid=False, error=diagnostic)
        raise InvalidCandidate(f"Repair exhausted after {len(repairs)} repairs: {diagnostic}")

    async def repair(self, policy: type[Policy], diagnostic: str) -> type[Policy] | None:
        """Repair an unevaluated policy after a sandbox failure, preserving its ancestry.

        The same budget covers generation and runtime repairs. The caller evaluates
        the returned replacement through Run; failed versions remain in storage.
        Exhaustion discards pending copies of this policy and returns None.
        """
        records = self._pending[policy.id]
        record = max(records, key=lambda row: len(row.get("repairs", [])))
        parent = record["parent"]
        reference = policy._implementation if parent is None else parent.policy._implementation
        try:
            replacement = await self._repair_valid(
                record, reference, policy._implementation, diagnostic
            )
        except InvalidCandidate as exc:
            for row in self._pending.pop(policy.id):
                row.update(status="discarded", error=str(exc))
            logger.warning("Discarded %s: %s", policy.name, exc)
            return None
        for row in records:
            row.update(policy=replacement, status="repaired", revision=row.get("revision", 0) + 1)
        self._pending.pop(policy.id)
        self._pending.setdefault(replacement.id, []).extend(records)
        logger.info("Repaired %s → %s — %s", policy.name, replacement.name, replacement.description)
        return replacement

    @prompt(template="original/prompts/evolve_prompt.j2", output_type=Guidance)
    async def evolve_prompt(
        self,
        parent: _Candidate,
        ideas: list[dict],
        failures: list[dict],
        *,
        generated: Guidance,
    ) -> str:
        return generated.instruction

    async def generate(self, n: int = 1, *, concurrency: int = 4) -> list[type[Policy]]:
        """Attempt n proposals and return survivors after bounded repair.

        Proposals are neither executed nor saved. Only a successfully returned
        batch is eligible for update; discarded output stays in attempts.
        Each concurrency slot includes its proposal's repairs. Results retain
        proposal order. Invalid candidates do not cancel siblings or get replaced
        with new proposals. Provider errors and cancellation still stop the batch.
        """
        logger.info(
            "Generating %s policies (concurrency=%s)",
            n,
            concurrency,
            extra={"event": "generation_started", "total": n},
        )
        slots = asyncio.Semaphore(concurrency)

        async def propose():
            async with slots:
                return await self._propose()

        tasks = [asyncio.create_task(propose()) for _ in range(n)]
        try:
            records = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        records = [record for record in records if record is not None]
        for record in records:
            self._pending.setdefault(record["policy"].id, []).append(record)
        return [record["policy"] for record in records]

    def update(
        self,
        scores: Mapping[str, float],
        *,
        seed_scores: Mapping[str, Mapping[int, float]] | None = None,
    ) -> None:
        """Rank by scalar scores; retain optional per-seed rewards as model feedback."""
        details = {policy_id: dict((seed_scores or {}).get(policy_id, {})) for policy_id in scores}
        # Validate the entire result before changing the archive or consuming pending work.
        for policy_id, score in scores.items():
            self._pending[policy_id]
            if not math.isfinite(score):
                raise ValueError("Policy scores must be finite")
            if any(not math.isfinite(value) for value in details[policy_id].values()):
                raise ValueError("Per-seed scores must be finite")
        for policy_id, score in scores.items():
            for record in self._pending.pop(policy_id):
                child = _Candidate(record["policy"], score, details[policy_id])
                parent, idea = record["parent"], record["idea"]
                if parent is None:
                    self._register_founder(child, record["island"])
                else:
                    self._register(child, record["island"])
                    improvement = (score - parent.score) / max(1.0, abs(parent.score))
                    idea.reward += max(0.0, improvement)
                record.update(status="evaluated", score=score)
                self.completed += 1
                if self.config.reset_interval and self.completed % self.config.reset_interval == 0:
                    self.reset_islands()

    def sample(self) -> tuple[int, _Candidate, list[_Candidate]]:
        island_id = self.rng.choice(
            [i for i, island in enumerate(self.islands) if island is not None]
        )
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

    async def _propose(self) -> dict | None:
        attempt_id = self._attempt_offset + len(self.attempts) + 1
        model_id = self.rng.choices(range(len(self.models)), [w for _, w in self.models])[0]
        provider = self.models[model_id][0]
        failures = (
            self._prior_failures
            + [
                {key: row[key] for key in ("id", "raw", "error") if key in row}
                for row in self.attempts
                if row.get("error")
            ]
        )[-3:]
        record = {"id": attempt_id, "model": model_id, "status": "generating"}
        self.attempts.append(record)
        try:
            parent = idea = None
            island_id = self._founding_island(attempt_id)
            if island_id is not None:
                operation = self.initialize
                arguments = (attempt_id,)
            else:
                island_id, parent, inspirations = self.sample()
            record.update(parent=parent, island=island_id)
            if parent is not None:
                idea = await self._choose_guidance(attempt_id, parent, failures, provider, record)
                idea.uses += 1
                variant = self.rng.choices(
                    [v for v, _ in self.variants], [w for _, w in self.variants]
                )[0]
                guidance = "\n".join((variant, idea.instruction))
                record["guidance"] = guidance
                operation = {"diff": self.mutate, "rewrite": self.rewrite}[self.config.mode]
                arguments = (parent, inspirations, guidance, failures)
            logger.info("Requesting policy %s via %s", attempt_id, operation.__name__)
            self.generation_calls += 1
            reference = "" if parent is None else parent.policy._implementation
            try:
                proposal = await asyncio.wait_for(
                    operation(*arguments, provider=RecordingProvider(provider, record, "raw")),
                    self.config.generation_timeout,
                )
                if parent is None:
                    content = proposal.implementation
                elif self.config.mode == "diff":
                    content = apply_edits(reference, proposal.edits)
                else:
                    content = proposal.implementation
                    record["content"] = content
                    check_rewrite(reference, content)
                record["content"] = content
                check_program(content)
                policy = _policy_class(proposal.name, content, proposal.description)
            except (InvalidCandidate, ValidationError) as exc:
                if isinstance(exc, ValidationError) and "raw" not in record:
                    raise
                if self.config.max_repairs == 0:
                    raise InvalidCandidate(str(exc)) from exc
                policy = await self._repair_valid(
                    record,
                    reference,
                    record.get("content", record.get("raw", "")),
                    f"{type(exc).__name__}: {exc}",
                )
            record.update(
                status="generated", policy=policy, parent=parent, island=island_id, idea=idea
            )
            logger.info(
                "Generated %s — %s",
                policy.name,
                policy.description,
                extra={"event": "policy_generated", "policy_id": policy.id},
            )
            return record
        except InvalidCandidate as exc:
            record.update(status="discarded", error=f"{type(exc).__name__}: {exc}")
            logger.warning(
                "Discarded proposal %s: %s",
                attempt_id,
                exc,
                extra={"event": "proposal_discarded"},
            )
            return None
        except (
            ValidationError,
            ProviderError,
            TimeoutError,
            asyncio.TimeoutError,
        ) as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            logger.error("Policy proposal %s failed: %s: %s", attempt_id, type(exc).__name__, exc)
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
                        provider=RecordingProvider(provider, record, "meta_raw"),
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

    def _founding_island(self, attempt_id: int) -> int | None:
        return 0 if self._best is None else None

    def _register_founder(self, candidate: _Candidate, island_id: int) -> None:
        for target in range(len(self.islands)):
            self._register(candidate, target)

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
                {
                    "completed": self.completed,
                    "reset": island_id,
                    "donor": donor,
                    "founder": founder.policy.id,
                }
            )
