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
from statistics import fmean
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from rich.table import Column
from slick import parse, render
from slick.providers import Provider, ProviderError

from rsikit.generation import WORKER_LIBRARIES
from rsikit.optimization import Optimizer, validate_results
from rsikit.policy import InvalidPolicy, Policy, validate_policy

from ..generation import (
    InvalidCandidate,
    Mutation,
    _PolicyResponse,
    apply_edits,
    check_rewrite,
    evolution_regions,
)
from .healing import SelfHealer

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
    batch_size: int = 10
    proposals: int = 250
    generation_concurrency: int = 4
    islands: int = 4
    inspirations: int = 3
    exploration: float = 0.2
    reset_interval: int = 100
    meta_interval: int = 0
    mode: Literal["diff", "rewrite"] = "diff"
    generation_timeout: float | None = None
    max_repairs: int = 2

    def __post_init__(self):
        for name, minimum in (("batch_size", 1), ("proposals", 0), ("generation_concurrency", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class AlphaEvolve(Optimizer):
    """Propose policies in memory; update selection from caller-supplied episodes.

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
        self.healer = SelfHealer(task, provider, context=context, libraries=self.libraries)
        self.config = config
        self.rng = random.Random(seed)
        self.islands: list[_Candidate | None] = [None] * config.islands
        self._best: _Candidate | None = None
        self._pending: dict[str, list[dict]] = {}
        self._round = {}
        self._repairs = {}
        self._proposing = False
        self._seed_panel = None
        self.prompt_ideas = [PromptIdea("")]
        # ponytail: in-memory attempt history; bound it if searches exceed RAM.
        self.attempts: list[dict] = []
        self._attempt_offset = 0
        self._batch_number = 0
        self._streaming_progress = None
        self._prior_failures: list[dict] = []
        self.events: list[dict] = []
        self.generation_calls = self.repair_calls = self.meta_calls = self.completed = 0

    leaderboard_columns = {"island": Column("Island"), "parent": Column("Parent")}

    def _log_batch(self, batch, total, *, streaming=False):
        logger.info(
            "Starting %s %s",
            "batch" if streaming else "generation",
            batch,
            extra={
                "progress": dict(
                    kind="batch_started",
                    batch_id=str(batch),
                    label=f"{'Batch' if streaming else 'Generation'} {batch}",
                    total_candidates=total,
                    optimizer=f"AlphaEvolve ({type(self).__module__.split('.')[-2]})",
                    columns=self.leaderboard_columns,
                )
            },
        )

    def _log_candidate(self, row, *, status=None, restored=False):
        policy = row.get("policy")
        state = status or {
            "repaired": "generated",
            "rejected": "failed",
            "error": "failed",
            "execution_failed": "failed",
        }.get(row["status"], row["status"])
        logger.info(
            "%s: %s — %s",
            policy.name if policy else f"Attempt {row['id']}",
            state,
            policy.description if policy else "",
            extra={
                "progress": dict(
                    kind="candidate",
                    batch_id=str(row.get("batch", self._batch_number)),
                    attempt_id=str(row["id"]),
                    revision=row.get("revision", 0),
                    status=state,
                    proposal_done=policy is not None,
                    policy_id=policy.id if policy else "—",
                    name=policy.name if policy else "",
                    description=policy.description if policy else "",
                    score=row.get("score"),
                    error=row.get("error"),
                    restored=restored,
                )
            },
        )

    def evaluation_started(self, policies):
        for policy in policies:
            for row in self._pending.get(policy.id, ()):
                self._log_candidate(row, status="evaluating")

    def _log_leaderboard(self):
        rows = []
        seen = set()
        for island, candidate in sorted(
            enumerate(self.islands), key=lambda pair: -pair[1].score if pair[1] else math.inf
        ):
            if candidate is None or candidate.policy.id in seen:
                continue
            seen.add(candidate.policy.id)
            record = next(
                (
                    row
                    for row in reversed(self.attempts)
                    if row.get("policy") is not None and row["policy"].id == candidate.policy.id
                ),
                {},
            )
            parent = record.get("parent")
            rows.append(
                dict(
                    id=candidate.policy.id,
                    name=candidate.policy.name,
                    description=candidate.policy.description,
                    score=candidate.score,
                    generation=record.get("batch"),
                    extras=dict(island=island, parent=parent.policy.id[:6] if parent else "—"),
                )
            )
        logger.info(
            "AlphaEvolve leaderboard: %s entries",
            len(rows),
            extra={"progress": dict(kind="leaderboard", rows=rows)},
        )

    @property
    def best(self) -> type[Policy] | None:
        return None if self._best is None else self._best.policy

    async def initialize(self, proposal: int, *, provider, record=None) -> type[Policy]:
        """Create an initial named policy without a hand-written seed program."""
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "original/prompts/initialize.j2", instance=self, schema=schema, proposal=proposal
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    async def mutate(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        provider,
        record=None,
    ) -> type[Policy]:
        schema = Mutation.model_json_schema()
        context = render(
            "original/prompts/mutate.j2",
            instance=self,
            schema=schema,
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        mutation = parse(raw, Mutation)
        return Policy.from_text(
            apply_edits(parent.policy._implementation, mutation.edits),
            name=mutation.name,
            description=mutation.description,
        )

    async def rewrite(
        self,
        parent: _Candidate,
        inspirations: list[_Candidate],
        guidance: str,
        failures: list[dict],
        *,
        provider,
        record=None,
    ) -> type[Policy]:
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "original/prompts/rewrite.j2",
            instance=self,
            schema=schema,
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

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
                    self.healer.repair(
                        reference,
                        failed,
                        diagnostic,
                        provider=provider,
                        record=call,
                    ),
                    self.config.generation_timeout,
                )
                content = proposal._implementation
                call["implementation"] = content
                if content == failed:
                    raise InvalidCandidate("Repair returned the unchanged implementation")
                if reference:
                    check_rewrite(reference, content)
                evolution_regions(content)
                validate_policy(proposal)
                call["valid"] = True
                return proposal
            except (InvalidPolicy, ValidationError) as exc:
                if isinstance(exc, ValidationError) and "raw" not in call:
                    raise  # Provider-side failures do not establish invalid model output.
                failed = call.get("implementation", call.get("raw", failed))
                diagnostic = f"{type(exc).__name__}: {exc}"
                call.update(valid=False, error=diagnostic)
        raise InvalidCandidate(f"Repair exhausted after {len(repairs)} repairs: {diagnostic}")

    async def repair(self, policy: type[Policy], diagnostic: str) -> type[Policy] | None:
        """Repair an unevaluated policy after a episode failure, preserving its ancestry.

        The same budget covers generation and runtime repairs. The caller evaluates
        the returned replacement through Run; failed versions remain in storage.
        Exhaustion discards pending copies of this policy and returns None.
        """
        records = self._pending[policy.id]
        record = max(records, key=lambda row: len(row.get("repairs", [])))
        for row in records:
            self._log_candidate(row, status="repairing")
        parent = record["parent"]
        reference = policy._implementation if parent is None else parent.policy._implementation
        try:
            replacement = await self._repair_valid(
                record, reference, policy._implementation, diagnostic
            )
        except InvalidPolicy as exc:
            for row in self._pending.pop(policy.id):
                row.update(status="discarded", error=str(exc))
                self._log_candidate(row)
            logger.warning("Discarded %s: %s", policy.name, exc)
            return None
        for row in records:
            row.update(policy=replacement, status="repaired", revision=row.get("revision", 0) + 1)
            self._log_candidate(row)
        self._pending.pop(policy.id)
        self._pending.setdefault(replacement.id, []).extend(records)
        logger.info("Repaired %s → %s — %s", policy.name, replacement.name, replacement.description)
        return replacement

    async def evolve_prompt(
        self, parent: _Candidate, ideas: list[dict], failures: list[dict], *, provider, record=None
    ) -> str:
        schema = Guidance.model_json_schema()
        context = render(
            "original/prompts/evolve_prompt.j2",
            instance=self,
            schema=schema,
            parent=parent,
            ideas=ideas,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["meta_raw"] = raw
        generated = parse(raw, Guidance)
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
        if self._streaming_progress is None:
            self._batch_number += 1
            self._log_batch(self._batch_number, n)
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

    @property
    def done(self) -> bool:
        return (
            not self._proposing
            and not self._pending
            and not self._round
            and not self._repairs
            and self._attempt_offset + len(self.attempts) >= self.config.proposals
        )

    async def propose(self) -> list[type[Policy]]:
        if self._proposing or self._round:
            raise RuntimeError("Previous proposal round is still outstanding")
        self._proposing = True
        try:
            while True:
                policies = []
                if self._repairs:
                    slots = asyncio.Semaphore(self.config.generation_concurrency)

                    async def repair(policy_id, diagnostic):
                        async with slots:
                            policy = self._pending[policy_id][0]["policy"]
                            replacement = await self.repair(policy, diagnostic)
                            self._repairs.pop(policy_id, None)
                            return replacement

                    tasks = [
                        asyncio.create_task(repair(id, diagnostic))
                        for id, diagnostic in list(self._repairs.items())
                    ]
                    try:
                        policies = [p for p in await asyncio.gather(*tasks) if p is not None]
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                elif self._pending:
                    # Restored or interrupted proposals retain their original attempt identities.
                    policies = [rows[0]["policy"] for rows in self._pending.values()]
                else:
                    remaining = self.config.proposals - self._attempt_offset - len(self.attempts)
                    if remaining <= 0:
                        return []
                    policies = await self.generate(
                        min(self.config.batch_size, remaining),
                        concurrency=self.config.generation_concurrency,
                    )
                if policies:
                    self._round = {p.id: p for p in policies}
                    self.evaluation_started(self._round.values())
                    return list(self._round.values())
                logger.info("No surviving policies in proposal round")
        finally:
            self._proposing = False

    def discard(self, policy, reason):
        for record in self._pending.pop(policy.id, []):
            record.update(status="discarded", error=reason)
            self._log_candidate(record)

    def _accept_measurements(self, results):
        scores = {}
        for policy_id, result in results.items():
            if not result.scores:
                raise ValueError("AlphaEvolve reward feedback requires per-seed scores")
            scores[policy_id] = fmean(result.scores.values())
        self.update_scores(scores, seed_scores={id: r.scores for id, r in results.items()})

    def update(self, results) -> None:
        panel = validate_results(results, self._round, seed_panel=self._seed_panel)
        self._accept_measurements({id: r for id, r in results.items() if r.accepted})
        for policy_id, result in results.items():
            if result.failure is not None:
                self._repairs[policy_id] = result.failure
                for row in self._pending[policy_id]:
                    row.update(status="execution_failed", error=result.failure)
                    self._log_candidate(row)
            elif not result.accepted:
                self.discard(self._round[policy_id], result.feedback or "Evaluation rejected")
        self._seed_panel = panel
        self._round.clear()
        self._log_leaderboard()

    def update_scores(
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
                self._log_candidate(record)
                self.completed += 1
                if self.config.reset_interval and self.completed % self.config.reset_interval == 0:
                    self.reset_islands()

        self._log_leaderboard()

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
        if self._streaming_progress is not None:
            offset, size, total = self._streaming_progress
            group = (attempt_id - offset - 1) // size
            record["batch"] = f"stream-{offset}-{group + 1}"
            self._log_batch(record["batch"], min(size, total - group * size), streaming=True)
        else:
            record["batch"] = self._batch_number
        self.attempts.append(record)
        self._log_candidate(record)
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
                    operation(*arguments, provider=provider, record=record),
                    self.config.generation_timeout,
                )
                content = proposal._implementation
                record["content"] = content
                if parent is not None:
                    check_rewrite(reference, content)
                evolution_regions(content)
                validate_policy(proposal)
                policy = proposal
            except (InvalidPolicy, ValidationError) as exc:
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
            self._log_candidate(record)
            return record
        except InvalidPolicy as exc:
            record.update(status="discarded", error=f"{type(exc).__name__}: {exc}")
            logger.warning(
                "Discarded proposal %s: %s",
                attempt_id,
                exc,
                extra={"event": "proposal_discarded"},
            )
            self._log_candidate(record)
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
                        provider=provider,
                        record=record,
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
