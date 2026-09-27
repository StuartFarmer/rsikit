"""ShinkaEvolve's island archives, adaptive model allocation, and proposal operations.

Adapted from rsikit/shinka.py on main; see NOTICE for provenance and API changes.
The caller owns evaluation and storage. Generated implementations never run here.
"""

import asyncio
import logging
import math
import random
import statistics
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, StrictBool, ValidationError
from slick import prompt
from slick.providers import Provider, ProviderError
from sqlmodel import SQLModel

from rsikit import Policy
from rsikit.generation import WORKER_LIBRARIES, RecordingProvider
from rsikit.generation.edits import (
    InvalidCandidate,
    Mutation,
    Program,
    apply_edits,
    check_program,
    check_rewrite,
    evolution_regions,
)
from rsikit.policy import _policy_class

from .records import Evaluation, Generation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    islands: int = 2
    archive_size: int = 40
    elite_ratio: float = 0.3
    top_k: int = 2
    inspirations: int = 4
    parent_selection: Literal["weighted", "uniform", "best", "power"] = "weighted"
    selection_pressure: float = 10.0
    power_alpha: float = 1.0
    exploration: float = 1.0
    patch_types: tuple[tuple[str, float], ...] = (("diff", 0.45), ("full", 0.45), ("cross", 0.1))
    max_proposals: int = 3
    max_repairs: int = 2
    novelty_threshold: float = 0.95
    meta_interval: int = 10
    max_recommendations: int = 5
    migration_interval: int = 10
    migration_rate: float = 0.1
    generation_timeout: float | None = None


class Novelty(BaseModel, extra="forbid"):
    novel: StrictBool
    reason: str = Field(min_length=1)


class Recommendations(BaseModel, extra="forbid"):
    recommendations: list[str]


@dataclass(frozen=True)
class _Candidate:
    policy: type[Policy]
    score: float

    def evidence(self) -> dict:
        return dict(
            id=self.policy.id,
            name=self.policy.name,
            description=self.policy.description,
            implementation=self.policy._implementation,
            score=self.score,
        )


class ShinkaEvolve:
    """Generate Policy classes; learn selection only from externally measured rewards."""

    libraries = WORKER_LIBRARIES

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        context: str = "",
        ensemble: Sequence[Provider] = (),
        config: Config = Config(),
        seed: int = 0,
        embed: Callable[[str], Awaitable[Sequence[float]]] | None = None,
        novelty_provider: Provider | None = None,
        meta_provider: Provider | None = None,
    ):
        self.task, self.context, self.config = task, context, config
        self.models = tuple(ensemble) or (provider,)
        self.novelty_provider, self.meta_provider = novelty_provider, meta_provider or provider
        self.embed, self.rng = embed, random.Random(seed)
        self.islands: list[list[_Candidate]] = [[] for _ in range(config.islands)]
        self.initial: _Candidate | None = None
        self._best: _Candidate | None = None
        self._archive: dict[str, _Candidate] = {}
        self._policies: dict[str, type[Policy]] = {}
        self._pending: dict[str, list[Evaluation]] = {}
        self.offspring: dict[str, int] = {}
        self.model_gains: list[list[Decimal]] = [[] for _ in self.models]
        self.embeddings: dict[str, tuple[float, ...]] = {}
        self.scratchpad: list[str] = []
        self.evaluations: list[Evaluation] = []
        self.generations: list[Generation] = []
        self.events: list[dict] = []
        # ponytail: retain call text and archive evidence in memory; bound for very long searches.
        self.calls: list[dict] = []
        self._batch_start = self._event_start = self._reflected = self._attempt = 0
        self._counted: set[int] = set()
        self.completed = self.generation_calls = self.repair_calls = self.meta_calls = (
            self.novelty_calls
        ) = 0

    @property
    def best(self) -> type[Policy] | None:
        return None if self._best is None else self._best.policy

    @prompt(template="initialize.j2", output_type=Program)
    async def initialize(self, proposal: int, *, generated: Program) -> Program:
        return generated

    @prompt(template="diff.j2", output_type=Mutation)
    async def diff(self, data: dict, *, generated: Mutation) -> Mutation:
        return generated

    @prompt(template="full.j2", output_type=Program)
    async def rewrite(self, data: dict, *, generated: Program) -> Program:
        return generated

    @prompt(template="cross.j2", output_type=Program)
    async def crossover(self, data: dict, *, generated: Program) -> Program:
        return generated

    @prompt(template="repair.j2", output_type=Program)
    async def fix(
        self, reference: str, failed: str, diagnostic: str, *, generated: Program
    ) -> Program:
        return generated

    @prompt(template="novelty.j2", output_type=Novelty)
    async def judge(
        self, implementation: str, nearest: dict, similarity: float, *, generated: Novelty
    ) -> Novelty:
        return generated

    @prompt(template="meta.j2", output_type=Recommendations)
    async def reflect(
        self, recent: list[dict], previous: list[str], *, generated: Recommendations
    ) -> list[str]:
        if len(generated.recommendations) > self.config.max_recommendations or any(
            not item.strip() for item in generated.recommendations
        ):
            raise InvalidCandidate(
                "Recommendations must be nonblank and within the configured limit"
            )
        return generated.recommendations

    async def _call(self, operation, *args, provider, attempt=None, call=None):
        call = {} if call is None else call
        call.update(operation=operation.__name__, attempt=attempt)
        self.calls.append(call)
        try:
            return await asyncio.wait_for(
                operation(*args, provider=RecordingProvider(provider, call)),
                self.config.generation_timeout,
            )
        except ValidationError as exc:
            call["error"] = str(exc)
            if "raw" in call:
                raise InvalidCandidate(f"Invalid generated JSON: {exc}") from exc
            raise
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def selection_weights(self, population) -> list[float]:
        scores = [candidate.score for candidate in population]
        if self.config.parent_selection == "uniform":
            return [1.0] * len(scores)
        if self.config.parent_selection == "best":
            return [float(score == max(scores)) for score in scores]
        if self.config.parent_selection == "power":
            ranks = {
                i: rank
                for rank, i in enumerate(sorted(range(len(scores)), key=lambda i: -scores[i]), 1)
            }
            return [ranks[i] ** -self.config.power_alpha for i in range(len(scores))]
        median = statistics.median_low(scores) / 2 + statistics.median_high(scores) / 2
        weights = []
        for candidate, score in zip(population, scores):
            x = self.config.selection_pressure * (score / 2 - median / 2) * 2
            sigmoid = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
            weights.append(sigmoid / (1 + self.offspring.get(candidate.policy.id, 0)))
        return weights

    def model_weights(self) -> list[float]:
        if any(not gains for gains in self.model_gains):
            return [float(not gains) for gains in self.model_gains]
        scale = max(g for gains in self.model_gains for g in gains) or 1
        total = sum(map(len, self.model_gains))
        return [
            statistics.fmean(math.expm1(g / scale) for g in gains) / math.expm1(1)
            + self.config.exploration * math.sqrt(2 * math.log(total + 1) / len(gains))
            for gains in self.model_gains
        ]

    def _selection(self, attempt: int):
        weights = self.model_weights()
        model = self.rng.choices(range(len(weights)), weights if any(weights) else None)[0]
        if self.initial is None:
            return (attempt - 1) % len(self.islands), model, "initialize", [], []
        island = self.rng.randrange(len(self.islands))
        population = self.islands[island]
        parent = self.rng.choices(population, self.selection_weights(population))[0]
        ranked = sorted(
            (c for c in population if c.policy.id != parent.policy.id), key=lambda c: -c.score
        )
        rest = ranked[self.config.top_k :]
        inspirations = ranked[: self.config.top_k] + self.rng.sample(
            rest, min(len(rest), self.config.inspirations)
        )
        patch = self.rng.choices(
            [p for p, _ in self.config.patch_types], [w for _, w in self.config.patch_types]
        )[0]
        parents = [parent]
        if patch == "cross":
            if ranked:
                parents.append(self.rng.choice(ranked))
            else:
                patch = "full"
        return island, model, patch, parents, inspirations

    async def generate(self, n: int = 1, *, concurrency: int = 4) -> list[type[Policy]]:
        """Attempt n candidates, with bounded resampling and repair; return survivors."""
        if n < 0 or concurrency < 1:
            raise ValueError("n must be nonnegative and concurrency must be positive")
        self._batch_start, self._event_start = len(self.evaluations), len(self.events)
        self.generations.append(Generation(number=len(self.generations) + 1))
        logger.info(
            "Generating %s policies (concurrency=%s)",
            n,
            concurrency,
            extra={"event": "generation_started", "total": n},
        )
        if n:
            await self._reflect()
        slots, proposed = asyncio.Semaphore(concurrency), set()

        async def propose():
            async with slots:
                return await self._propose(proposed)

        tasks = [asyncio.create_task(propose()) for _ in range(n)]
        try:
            rows = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        policies = []
        for row in rows:
            if row.status == "discarded":
                self._finish(row)
            else:
                self._pending.setdefault(row.policy_id, []).append(row)
                policies.append(self._policies[row.policy_id])
        return policies

    async def _propose(self, proposed: set) -> Evaluation:
        self._attempt += 1
        island, model, patch, parents, inspirations = self._selection(self._attempt)
        row = Evaluation(
            generation=len(self.generations),
            attempt=self._attempt,
            island=island,
            model=model,
            patch=patch,
            parents=[p.policy.id for p in parents],
        )
        self.evaluations.append(row)
        reference = parents[0].policy._implementation if parents else ""
        failures = []
        try:
            for _ in range(self.config.max_proposals):
                row.proposals += 1
                self.generation_calls += 1
                logger.info(
                    "Requesting policy %s via %s (proposal %s/%s)",
                    row.attempt,
                    patch,
                    row.proposals,
                    self.config.max_proposals,
                )
                data = dict(
                    parents=[p.evidence() for p in parents],
                    inspirations=[p.evidence() for p in inspirations],
                    outcomes=[c.evidence() for c in list(self._archive.values())[-5:]],
                    guidance=self.scratchpad,
                    failures=failures,
                )
                operation = {
                    "initialize": self.initialize,
                    "diff": self.diff,
                    "full": self.rewrite,
                    "cross": self.crossover,
                }[patch]
                arguments = (row.attempt,) if patch == "initialize" else (data,)
                content, call = "", {}
                try:
                    try:
                        proposal = await self._call(
                            operation,
                            *arguments,
                            provider=self.models[model],
                            attempt=row.attempt,
                            call=call,
                        )
                        content = (
                            apply_edits(reference, proposal.edits)
                            if patch == "diff"
                            else proposal.implementation
                        )
                        if reference and patch != "diff":
                            check_rewrite(reference, content)
                        check_program(content)
                        policy = _policy_class(proposal.name, content, proposal.description)
                    except InvalidCandidate as exc:
                        failed = content or call.get("raw", "")
                        policy = await self._repair_valid(row, reference, failed, str(exc))
                    await self.check_novelty(policy._implementation, self.islands[island], row)
                    key = (island, policy._implementation)
                    if key in proposed:
                        raise InvalidCandidate(
                            "Duplicate program in this island's generation batch"
                        )
                    proposed.add(key)
                    self._policies[policy.id] = policy
                    row.policy_id, row.status, row.error = policy.id, "generated", None
                    logger.info(
                        "Generated %s — %s",
                        policy.name,
                        policy.description,
                        extra={"event": "policy_generated", "policy_id": policy.id},
                    )
                    return row
                except InvalidCandidate as exc:
                    row.error = str(exc)
                    failures.append(dict(implementation=content, error=str(exc)))
                    logger.warning("Rejected proposal for attempt %s: %s", row.attempt, exc)
            row.status = "discarded"
            logger.warning(
                "Discarded proposal %s: %s",
                row.attempt,
                row.error,
                extra={"event": "proposal_discarded"},
            )
            return row
        except asyncio.CancelledError:
            row.status = "cancelled"
            raise
        except Exception as exc:
            row.status, row.error = "error", f"{type(exc).__name__}: {exc}"
            logger.error("Policy proposal %s failed: %s", row.attempt, row.error)
            raise

    async def _repair_valid(self, row, reference, failed, diagnostic) -> type[Policy]:
        while row.repairs < self.config.max_repairs:
            row.repairs += 1
            self.repair_calls += 1
            logger.warning(
                "Repairing proposal %s (%s/%s): %s",
                row.attempt,
                row.repairs,
                self.config.max_repairs,
                diagnostic,
            )
            call = {}
            try:
                proposal = await self._call(
                    self.fix,
                    reference,
                    failed,
                    diagnostic,
                    provider=self.models[row.model],
                    attempt=row.attempt,
                    call=call,
                )
                content = proposal.implementation
                if content == failed:
                    raise InvalidCandidate("Repair returned the unchanged implementation")
                if reference:
                    check_rewrite(reference, content)
                check_program(content)
                return _policy_class(proposal.name, content, proposal.description)
            except InvalidCandidate as exc:
                failed = call.get("raw", failed)
                diagnostic = str(exc)
        raise InvalidCandidate(f"Repair exhausted after {row.repairs} repairs: {diagnostic}")

    def evaluation_failed(self, failures: Mapping[str, str]) -> None:
        for policy_id, diagnostic in failures.items():
            for row in self._pending[policy_id]:
                row.status, row.error = "failed", diagnostic

    async def repair(self, policy: type[Policy], diagnostic: str) -> type[Policy] | None:
        self.evaluation_failed({policy.id: diagnostic})
        rows = self._pending[policy.id]
        row = max(rows, key=lambda item: item.repairs)
        reference = (
            self._policies[row.parents[0]]._implementation
            if row.parents
            else policy._implementation
        )
        try:
            replacement = await self._repair_valid(
                row, reference, policy._implementation, diagnostic
            )
            await self.check_novelty(replacement._implementation, self.islands[row.island], row)
        except InvalidCandidate as exc:
            for item in self._pending.pop(policy.id):
                item.status, item.error = "discarded", str(exc)
                self._finish(item)
            logger.warning("Discarded %s: %s", policy.name, exc)
            return None
        replacements = [
            Evaluation(
                **{
                    **item.model_dump(),
                    "revision": item.revision + 1,
                    "policy_id": replacement.id,
                    "status": "repaired",
                    "error": None,
                    "repairs": row.repairs,
                }
            )
            for item in rows
        ]
        self._pending.pop(policy.id)
        self._pending.setdefault(replacement.id, []).extend(replacements)
        self.evaluations.extend(replacements)
        self._policies[replacement.id] = replacement
        logger.info("Repaired %s → %s — %s", policy.name, replacement.name, replacement.description)
        return replacement

    def update(self, scores: Mapping[str, float]) -> None:
        """Rank by finite mean rewards; provider/infrastructure errors are not bad fitness."""
        for policy_id, score in scores.items():
            self._pending[policy_id]
            if not math.isfinite(score):
                raise ValueError("Policy scores must be finite")
        for policy_id, score in scores.items():
            for row in self._pending.pop(policy_id):
                row.score, row.status, row.error = score, "evaluated", None
                self._finish(row, score)

    def _finish(self, row: Evaluation, score: float | None = None) -> None:
        if row.attempt in self._counted:
            return
        gain = Decimal(0)
        if row.parents and row.policy_id is not None:
            parent_id = row.parents[0]
            self.offspring[parent_id] = self.offspring.get(parent_id, 0) + 1
        if score is not None:
            candidate = _Candidate(self._policies[row.policy_id], score)
            if self.initial is None:
                self.initial = candidate
                self.islands = [[candidate] for _ in self.islands]
            if row.parents:
                baseline = max(self._archive[row.parents[0]].score, self.initial.score)
                gain = max(Decimal(score) - Decimal(baseline), Decimal(0))
            self._archive[row.policy_id] = candidate
            population = self.islands[row.island]
            if all(item.policy.id != row.policy_id for item in population):
                population.append(candidate)
                self.prune(population)
            if self._best is None or score > self._best.score:
                self._best = candidate
        row.model_gain = str(gain)
        self.model_gains[row.model].append(gain)
        self._counted.add(row.attempt)
        self.completed += 1
        if (
            self.initial is not None
            and self.config.migration_interval
            and self.completed % self.config.migration_interval == 0
        ):
            self.migrate()

    def prune(self, island) -> None:
        if len(island) > self.config.archive_size:
            ranked = sorted(island, key=lambda candidate: -candidate.score)
            count = max(1, math.ceil(self.config.archive_size * self.config.elite_ratio))
            island[:] = ranked[:count] + self.rng.sample(
                ranked[count:], self.config.archive_size - count
            )

    def migrate(self) -> None:
        if len(self.islands) < 2:
            return
        batches = []
        for island in self.islands:
            best = max(island, key=lambda candidate: candidate.score)
            pool = [candidate for candidate in island if candidate.policy.id != best.policy.id]
            batches.append(
                self.rng.sample(
                    pool, min(len(pool), math.floor(len(island) * self.config.migration_rate))
                )
            )
        for origin, migrants in enumerate(batches):
            target = (origin + 1) % len(self.islands)
            island = self.islands[target]
            additions = [
                c for c in migrants if all(c.policy.id != other.policy.id for other in island)
            ]
            island.extend(additions)
            self.prune(island)
            if additions:
                self.events.append(
                    dict(
                        completed=self.completed,
                        migration=[c.policy.id for c in additions],
                        origin=origin,
                        target=target,
                    )
                )

    async def embedding(self, implementation: str) -> tuple[float, ...]:
        if implementation not in self.embeddings:
            regions, _ = evolution_regions(implementation)
            vector = tuple(
                await self.embed("\n".join(implementation[start:end] for start, end in regions))
            )
            norm = math.hypot(*vector)
            if not vector or not math.isfinite(norm) or norm == 0:
                raise ValueError("Embedding must be a finite nonzero vector")
            self.embeddings[implementation] = tuple(value / norm for value in vector)
        return self.embeddings[implementation]

    async def check_novelty(self, implementation, population, row) -> None:
        if any(candidate.policy._implementation == implementation for candidate in population):
            raise InvalidCandidate("Duplicate island program")
        if self.embed is None or not population:
            return
        vector = await self.embedding(implementation)
        similarities = []
        for candidate in population:
            reference = await self.embedding(candidate.policy._implementation)
            if len(vector) != len(reference):
                raise ValueError("Embedding dimensions changed")
            similarities.append(sum(a * b for a, b in zip(vector, reference)))
        index = max(range(len(population)), key=similarities.__getitem__)
        nearest, similarity = population[index], similarities[index]
        row.similarity, row.nearest_id = similarity, nearest.policy.id
        if similarity > self.config.novelty_threshold:
            if self.novelty_provider is None:
                raise InvalidCandidate("Embedding similarity exceeds threshold")
            self.novelty_calls += 1
            judgment = await self._call(
                self.judge,
                implementation,
                nearest.evidence(),
                similarity,
                provider=self.novelty_provider,
                attempt=row.attempt,
            )
            row.novelty_reason = judgment.reason
            if not judgment.novel:
                raise InvalidCandidate(judgment.reason)

    async def _reflect(self) -> None:
        interval = self.config.meta_interval
        if (
            not interval
            or not self.completed
            or self.completed // interval <= self._reflected // interval
        ):
            return
        self._reflected = self.completed
        recent = self.evaluations[-interval:]
        ids = dict.fromkeys([self.initial.policy.id] if self.initial else [])
        for row in recent:
            ids.update(dict.fromkeys([*row.parents, row.policy_id]))
        evidence = [self._archive[key].evidence() for key in ids if key in self._archive]
        evidence.extend(
            dict(status=row.status, error=row.error, parents=row.parents)
            for row in recent
            if row.error
        )
        event = dict(meta_after=self.completed)
        self.events.append(event)
        self.meta_calls += 1
        try:
            self.scratchpad = await self._call(
                self.reflect, evidence, self.scratchpad, provider=self.meta_provider
            )
            event["recommendations"] = self.scratchpad
        except (InvalidCandidate, ProviderError, TimeoutError) as exc:
            event["error"] = str(exc)
            logger.warning("Keeping previous search guidance: %s", exc)

    def records(self, *, seeds=(), complete=False) -> list[SQLModel]:
        """Return the actual typed batch records and a current island snapshot for Run.save."""
        generation = self.generations[-1]
        generation.complete, generation.seeds = complete, list(seeds)
        generation.islands = [
            [dict(policy_id=c.policy.id, score=c.score) for c in population]
            for population in self.islands
        ]
        generation.events = self.events[self._event_start :]
        generation.recommendations = list(self.scratchpad)
        generation.model_weights = self.model_weights()
        generation.model_counts = list(map(len, self.model_gains))
        return [*self.evaluations[self._batch_start :], generation]
