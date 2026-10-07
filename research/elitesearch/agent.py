"""Breed a fixed population from measured elites and retain the best across generations."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from statistics import fmean

from pydantic import ConfigDict, Field, ValidationError, model_validator
from pydantic.dataclasses import dataclass
from rich.table import Column
from slick import parse, render
from slick.providers import Provider

from research.rewards import episode_error, episode_scores
from rsikit import Episode, Policy, search
from rsikit.generation import WORKER_LIBRARIES
from rsikit.optimization import validate_results
from rsikit.policy import InvalidPolicy, validate_policy

from .generation import (
    InvalidCandidate,
    Mutation,
    _PolicyResponse,
    apply_edits,
    check_rewrite,
)
from .healing import SelfHealer
from .records import Generation, Organism

logger = logging.getLogger(__name__)


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid", allow_inf_nan=False))
class Config:
    elite_size: int = Field(default=10, ge=1)
    population_size: int = Field(default=50, ge=1)
    generations: int = Field(default=20, ge=0)
    new_fraction: float = Field(default=0.2, ge=0, le=1)
    remix_fraction: float = Field(default=0.4, ge=0, le=1)
    remix_parents: int = Field(default=3, ge=2)
    generation_concurrency: int = Field(default=100, ge=1)
    generation_timeout: float = Field(default=120, gt=0)
    max_repairs: int = Field(default=2, ge=0)
    target_score: float | None = None

    @model_validator(mode="after")
    def valid_fractions(self):
        if self.new_fraction + self.remix_fraction > 1:
            raise ValueError("new_fraction + remix_fraction must not exceed 1")
        return self


class EliteSearch:
    """One run per instance. Evaluation is injected and must isolate generated code.

    Each population reads the prior generation's elites. Mean reward ranks first;
    ties prefer current edits, remixes, incumbents, then new ideas. Configure Slick's
    template root to this package's prompts directory before running.
    Evaluation callbacks run concurrently as candidates arrive; use Executor to
    bound episode workers. Each generation still ranks only after all candidates finish.
    """

    leaderboard_columns = {"operation": Column("Operation"), "parents": Column("Parents")}
    optimizer_name = "EliteSearch"

    def _log_candidate(self, row, *, restored=False):
        status = {"rejected": "repairing", "execution_failed": "repairing", "error": "failed"}.get(
            row.status, row.status
        )
        logger.info(
            "%s: %s — %s",
            row.name or f"Attempt {row.id}",
            status,
            row.description,
            extra={
                "progress": dict(
                    kind="candidate",
                    batch_id=str(row.generation),
                    attempt_id=str(row.id),
                    revision=row.repairs,
                    status=status,
                    proposal_done=bool(row.policy_id),
                    policy_id=row.policy_id or "—",
                    name=row.name,
                    description=row.description,
                    score=row.score,
                    error=row.error,
                    restored=restored,
                )
            },
        )

    def _log_leaderboard(self):
        logger.info(
            "Elite leaderboard: %s entries",
            len(self.elites),
            extra={
                "progress": dict(
                    kind="leaderboard",
                    rows=[
                        dict(
                            id=row.policy_id or str(row.id),
                            name=row.name,
                            description=row.description,
                            score=row.score,
                            generation=row.generation,
                            extras=dict(
                                operation=row.kind,
                                parents=", ".join(map(str, row.parent_ids)) or "—",
                            ),
                        )
                        for row in self.elites
                    ],
                )
            },
        )

    libraries = WORKER_LIBRARIES

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[type[Policy]]], Awaitable[dict[str, dict[int, Episode]]]]
        | None = None,
        *,
        context: str = "",
        config: Config = Config(),
        seed: int = 0,
        on_checkpoint: Callable[[EliteSearch], None] | None = None,
    ):
        self.healer = SelfHealer(task, provider, context=context, libraries=self.libraries)
        self.task, self.context, self.provider = task, context, provider
        self.evaluate, self.config, self.on_checkpoint = evaluate, config, on_checkpoint
        self.rng = random.Random(seed)
        self.elites: list[Organism] = []
        self.organisms: list[Organism] = []
        self.generations: list[Generation] = []
        self.reason = "ready"
        self._round = {}
        self._proposing = False
        self._policies: dict[int, type[Policy]] = {}
        self._sources: set[str] = set()
        self._seed_panel: set[int] | None = None
        self._call_slots = asyncio.Semaphore(config.generation_concurrency)

    @property
    def best(self) -> type[Policy] | None:
        return self._policies[self.elites[0].id] if self.elites else None

    def records(self):
        return [*self.generations, *self.organisms]

    def restore(self, organisms: list[Organism], generations: list[Generation]) -> None:
        """Restore trusted records into a fresh agent; replay only parent allocation.

        Replaying allocation recovers RNG state for legacy runs without an RNG
        checkpoint. No model calls, policy execution, or score updates occur here.
        """
        if self.organisms or self.generations:
            raise ValueError("Restore requires a fresh agent")
        organisms = sorted(organisms, key=lambda row: row.id)
        generations = sorted(generations, key=lambda row: row.number)
        if [row.id for row in organisms] != list(range(1, len(organisms) + 1)):
            raise ValueError("Checkpoint organism IDs must be contiguous")
        if [g.number for g in generations] != list(range(1, len(generations) + 1)):
            raise ValueError("Checkpoint generation numbers must be contiguous")
        if any(g.status != "completed" for g in generations[:-1]):
            raise ValueError("Only the last checkpoint generation may be unfinished")
        if len(organisms) != len(generations) * self.config.population_size:
            raise ValueError("Checkpoint population does not match the search config")
        for generation in generations:
            expected = self._population(generation)
            rows = [row for row in organisms if row.generation == generation.number]
            if [(r.id, r.kind, r.parent_ids) for r in rows] != [
                (r.id, r.kind, r.parent_ids) for r in expected
            ]:
                raise ValueError("Checkpoint parents do not match the search seed/config")
            self.organisms[-len(rows) :] = rows
            self.generations.append(generation)
            for row in rows:
                if row.score is not None:
                    panel = set(map(int, row.seed_scores))
                    if (
                        row.status != "evaluated"
                        or not row.policy_id
                        or not panel
                        or not math.isfinite(row.score)
                        or any(not math.isfinite(v) for v in row.seed_scores.values())
                        or (self._seed_panel is not None and panel != self._seed_panel)
                    ):
                        raise ValueError("Invalid checkpoint scores")
                    self._seed_panel = panel
                if row.policy_id is not None:
                    policy = Policy.from_text(
                        row.implementation, name=row.name, description=row.description
                    )
                    if policy.id != row.policy_id:
                        raise ValueError("Checkpoint policy ID does not match its source")
                    self._policies[row.id] = policy
                implementations = [row.implementation] if row.policy_id else []
                implementations.extend(
                    revision["implementation"]
                    for revision in row.revisions
                    if revision.get("policy_id") is not None
                )
                for implementation in implementations:
                    validate_policy(Policy.from_text(implementation))
                    self._sources.add(ast.dump(ast.parse(implementation), include_attributes=False))
            if generation.status == "completed":
                ranked = self._rank(rows)
                if [row.id for row in ranked] != generation.elite_ids:
                    # Preserve pre-priority checkpoints and the parent order they used.
                    ranked = sorted(
                        self.elites + [row for row in rows if row.score is not None],
                        key=lambda row: (-row.score, row.id),
                    )[: self.config.elite_size]
                    if [row.id for row in ranked] != generation.elite_ids:
                        raise ValueError("Checkpoint elites do not match recorded scores")
                self.elites = ranked

    def _checkpoint(self):
        # ponytail: full snapshots; use incremental writes when history persistence dominates runtime.
        if self.on_checkpoint is not None:
            self.on_checkpoint(self)

    async def invent(
        self, proposal: int, elites: list[Organism], *, provider, record=None
    ) -> type[Policy]:
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "new.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            proposal=proposal,
            elites=elites,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    async def edit(self, parent: Organism, *, provider, record=None) -> type[Policy]:
        schema = Mutation.model_json_schema()
        context = render(
            "edit.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parent=parent,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        mutation = parse(raw, Mutation)
        if record is not None:
            record.update(name=mutation.name, description=mutation.description)
        return Policy.from_text(
            apply_edits(parent.implementation, mutation.edits),
            name=mutation.name,
            description=mutation.description,
        )

    async def remix(self, parents: list[Organism], *, provider, record=None) -> type[Policy]:
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "remix.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parents=parents,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    async def _call(self, row, operation, *args):
        call = dict(operation=operation.__name__)
        row.calls.append(call)
        self._checkpoint()  # Persist an in-flight call and its consumed repair budget.
        try:
            async with self._call_slots:
                return await asyncio.wait_for(
                    operation(*args, provider=self.provider, record=call),
                    self.config.generation_timeout,
                )
        except ValidationError as exc:
            call["error"] = str(exc)
            if "raw" not in call:
                raise
            raise InvalidCandidate(f"Invalid generated output: {exc}") from exc
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if "name" in call:
                # Keep parsed identity even when applying the mutation fails.
                row.name, row.description = call["name"], call["description"]

    def _population(self, generation):
        n = self.config.population_size
        new = int(n * self.config.new_fraction) if self.elites else n
        remix = int(n * self.config.remix_fraction) if len(self.elites) >= 2 else 0
        kinds = ["new"] * new + ["remix"] * remix + ["edit"] * (n - new - remix)
        self.rng.shuffle(kinds)
        rows = []
        for kind in kinds:
            count = (
                0
                if kind == "new"
                else 1
                if kind == "edit"
                else min(len(self.elites), self.config.remix_parents)
            )
            parents = self.rng.sample(self.elites, count)
            rows.append(
                Organism(
                    id=len(self.organisms) + len(rows) + 1,
                    generation=generation.number,
                    kind=kind,
                    parent_ids=[p.id for p in parents],
                )
            )
        self.organisms.extend(rows)
        return rows

    async def _generate(self, row, *, repairing=False):
        parents = [self.organisms[i - 1] for i in row.parent_ids]
        reference = parents[0].implementation if row.kind == "edit" else ""
        while True:
            if repairing:
                if row.repairs >= self.config.max_repairs:
                    row.status = "discarded"
                    self._log_candidate(row)
                    logger.warning(
                        "Discarded organism %s: %s",
                        row.id,
                        row.error,
                        extra={"event": "proposal_discarded"},
                    )
                    return row
                failed = row.implementation or row.calls[-1].get("raw", "")
                row.revisions.append(
                    dict(
                        implementation=failed,
                        policy_id=row.policy_id,
                        error=row.error,
                        status=row.status,
                    )
                )
                row.repairs += 1
                operation, args = self.healer.repair, (reference, failed, row.error)
                logger.warning(
                    "Repairing organism %s (%s/%s): %s",
                    row.id,
                    row.repairs,
                    self.config.max_repairs,
                    row.error,
                )
            elif row.kind == "new":
                operation, args = self.invent, (row.id, list(self.elites))
            elif row.kind == "edit":
                operation, args = self.edit, (parents[0],)
            else:
                operation, args = self.remix, (parents,)
            row.status, row.policy_id, row.implementation = "generating", None, ""
            self._policies.pop(row.id, None)
            self._log_candidate(row)
            try:
                proposal = await self._call(row, operation, *args)
                row.name, row.description = proposal.name, proposal.description
                row.implementation = proposal._implementation
                validate_policy(proposal)
                if reference:
                    check_rewrite(reference, row.implementation)
                key = ast.dump(ast.parse(row.implementation), include_attributes=False)
                if key in self._sources:
                    raise InvalidCandidate("Duplicate program; make a substantive change")
                self._sources.add(key)
                policy = proposal
                self._policies[row.id] = policy
                row.policy_id, row.status, row.error = policy.id, "generated", None
                logger.info(
                    "Generated %s (%s) — %s",
                    row.name,
                    row.kind,
                    row.description,
                    extra={"event": "policy_generated"},
                )
                self._log_candidate(row)
                self._checkpoint()
                return row
            except InvalidPolicy as exc:
                row.status, row.error = "rejected", str(exc)
                repairing = True
            except BaseException as exc:
                row.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                row.error = f"{type(exc).__name__}: {exc}"
                raise

    async def _prepare(self, rows):
        logger.info(
            "Generating population of %s",
            len(rows),
            extra={"event": "generation_started", "total": len(rows)},
        )

        async def produce(row):
            if row.status in ("evaluated", "discarded"):
                return
            repairing = row.status in ("rejected", "execution_failed")
            if not repairing and row.id in self._policies:
                row.status = "generated"  # Reuse cached episodes; evaluate only missing seeds.
            else:
                if row.calls and row.calls[-1]["operation"] == "repair" and row.revisions:
                    repairing = True
                    if not row.implementation:
                        row.implementation = row.revisions[-1]["implementation"]
                        row.error = row.revisions[-1]["error"]
                await self._generate(row, repairing=repairing)

        tasks = [asyncio.create_task(produce(row)) for row in rows]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _rank(self, rows):
        incumbents = {row.id for row in self.elites}
        return sorted(
            self.elites + [row for row in rows if row.score is not None],
            key=lambda row: (
                -row.score,
                2 if row.id in incumbents else {"edit": 0, "remix": 1, "new": 3}[row.kind],
                row.id,
            ),
        )[: self.config.elite_size]

    def _promote(self, generation, rows):
        previous = {row.id for row in self.elites}
        self.elites = self._rank(rows)
        generation.elite_ids = [row.id for row in self.elites]
        generation.promoted_ids = [row.id for row in self.elites if row.id not in previous]
        generation.status = "completed"
        logger.info(
            "Generation %s: %s promotions; %s/%s elites; best=%s",
            generation.number,
            len(generation.promoted_ids),
            len(self.elites),
            self.config.elite_size,
            self.elites[0].score if self.elites else "unfilled",
        )

    @property
    def done(self):
        if self._round or self._proposing:
            return False
        if self.generations and self.generations[-1].status != "completed":
            return False
        return (
            len(self.generations) >= self.config.generations
            or self.config.target_score is not None
            and bool(self.elites)
            and self.elites[0].score >= self.config.target_score
        )

    def _settle_generation(self):
        generation = self.generations[-1]
        rows = [row for row in self.organisms if row.generation == generation.number]
        if generation.status == "completed" or any(
            row.status not in ("evaluated", "discarded") for row in rows
        ):
            return
        self._promote(generation, rows)
        self._log_leaderboard()
        logger.info(
            "Finished generation %s",
            generation.number,
            extra={
                "progress": dict(
                    kind="batch_finished", batch_id=str(generation.number), status="completed"
                )
            },
        )
        self.reason = (
            "target_reached"
            if self.config.target_score is not None
            and self.elites
            and self.elites[0].score >= self.config.target_score
            else "completed"
            if len(self.generations) >= self.config.generations
            else "running"
        )
        self._checkpoint()

    async def propose(self):
        if self._proposing or self._round:
            raise RuntimeError("Previous proposal round is still outstanding")
        if self.done:
            return []
        self._proposing = True
        self.reason = "running"
        try:
            while True:
                if not self.generations or self.generations[-1].status == "completed":
                    if (
                        len(self.generations) >= self.config.generations
                        or self.config.target_score is not None
                        and self.elites
                        and self.elites[0].score >= self.config.target_score
                    ):
                        return []
                    generation = Generation(
                        number=len(self.generations) + 1, elite_ids=[row.id for row in self.elites]
                    )
                    self.generations.append(generation)
                    self._population(generation)
                generation = self.generations[-1]
                generation.status, generation.error = "running", None
                rows = [row for row in self.organisms if row.generation == generation.number]
                logger.info(
                    "Generation %s",
                    generation.number,
                    extra={
                        "progress": dict(
                            kind="batch_started",
                            batch_id=str(generation.number),
                            label=f"Generation {generation.number}",
                            total_candidates=len(rows),
                        )
                    },
                )
                self._checkpoint()
                await self._prepare(rows)
                self._round = {row.policy_id: row for row in rows if row.status == "generated"}
                if self._round:
                    for row in self._round.values():
                        row.status = "evaluating"
                        self._log_candidate(row)
                    return [self._policies[row.id] for row in self._round.values()]
                self._settle_generation()
        finally:
            self._proposing = False

    def update(self, results):
        panel = validate_results(results, self._round, seed_panel=self._seed_panel)
        score_panels = {id: episode_scores(r) for id, r in results.items()}
        for id, result in results.items():
            error = episode_error(result)
            row = self._round[id]
            if error is not None:
                row.status, row.error = "execution_failed", error
            elif not result:
                row.status, row.error = "discarded", "Evaluation rejected"
            else:
                row.score = fmean(score_panels[id].values())
                row.seed_scores = {str(seed): value for seed, value in score_panels[id].items()}
                row.status, row.error = "evaluated", None
            self._log_candidate(row)
        self._round.clear()
        self._seed_panel = panel
        self._checkpoint()
        self._settle_generation()

    async def run(self) -> list[Organism]:
        """Compatibility wrapper; new callers pass this optimizer to rsikit.search."""
        if self.evaluate is None:
            raise ValueError("Pass an evaluator to rsikit.search(optimizer, evaluate)")
        logger.info(
            "Starting %s",
            self.optimizer_name,
            extra={
                "progress": dict(
                    kind="search_started",
                    optimizer=self.optimizer_name,
                    total_candidates=self.config.population_size * self.config.generations,
                    total_generations=self.config.generations,
                    leaderboard_size=self.config.elite_size,
                    columns=self.leaderboard_columns,
                    resumed=bool(self.generations),
                )
            },
        )
        for row in self.organisms:
            self._log_candidate(row, restored=True)
        self._log_leaderboard()
        try:
            await search(self, self.evaluate, on_checkpoint=lambda agent: agent._checkpoint())
            if self.reason in ("ready", "running"):
                self.reason = "completed"
            return self.elites
        except BaseException as exc:
            self.reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            if self.generations and self.generations[-1].status != "completed":
                self.generations[-1].status = self.reason
                self.generations[-1].error = f"{type(exc).__name__}: {exc}"
            try:
                self._checkpoint()
            except BaseException:
                logger.exception("Checkpoint failed while handling search error")
            raise
        finally:
            logger.info(
                "Search %s",
                self.reason,
                extra={
                    "progress": dict(
                        kind="search_finished",
                        status={"error": "failed", "target_reached": "stopped"}.get(
                            self.reason, self.reason
                        ),
                        reason=self.reason,
                    )
                },
            )
