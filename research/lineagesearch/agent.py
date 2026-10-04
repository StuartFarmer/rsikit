"""Generate distinct approach families and expand them using measured evidence.

The caller supplies isolated evaluation on a fixed seed panel and owns persistence.
Generated Python is parsed, never imported or executed by this optimizer.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, stdev
from typing import Annotated

from pydantic import BaseModel, Field, StrictInt, StringConstraints, ValidationError
from rich.table import Column
from slick import parse, render
from slick.providers import Provider
from sqlmodel import SQLModel

from research.rewards import episode_error, episode_scores
from rsikit import Episode, Policy, search
from rsikit.generation import WORKER_LIBRARIES
from rsikit.optimization import validate_results
from rsikit.policy import InvalidPolicy, validate_policy

from .generation import InvalidCandidate, _PolicyResponse, check_rewrite, evolution_regions
from .healing import SelfHealer
from .records import Family, Study, Trial

logger = logging.getLogger(__name__)
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True)
class Config:
    families: int = 10
    initial_per_family: int = 10
    batch_size: int = 10
    frontier_per_family: int | None = None
    cull_percent: float = 90.0
    exploration: float = 0.2
    bonus_batches: int = 2
    patience: int = 3
    min_delta: float = 0.0
    uncertainty: float = 2.0
    max_attempts: int = 500
    generation_concurrency: int = 100
    generation_timeout: float = 120
    discovery_attempts: int = 3
    max_repairs: int = 2
    decomposition_k: int = 3
    decomposition_max_votes: int = 40

    def __post_init__(self):
        if not 0 <= self.cull_percent < 100:
            raise ValueError("cull_percent must be between 0 (inclusive) and 100 (exclusive)")


class FamilyBrief(BaseModel, extra="forbid"):
    name: Text
    mechanism: Text


class Families(BaseModel, extra="forbid"):
    families: list[FamilyBrief] = Field(min_length=1)


class Approach(BaseModel, extra="forbid"):
    hypothesis: Text
    mechanism: Text
    change: Text
    test: Text


class Approaches(BaseModel, extra="forbid"):
    approaches: list[Approach] = Field(min_length=1)


class Vote(BaseModel, extra="forbid"):
    candidate: StrictInt
    reason: Text


class LineageSearch:
    """One study per instance; run() returns its terminal Study record.

    Score comparisons maximize paired per-seed rewards. `uncertainty` times the
    standard error is a practical noise margin, not a multiple-testing guarantee.
    With one seed, comparisons assume a deterministic evaluator.
    """

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
        on_checkpoint: Callable[[LineageSearch], None] | None = None,
    ):
        self.healer = SelfHealer(task, provider, context=context, libraries=self.libraries)
        self.task, self.context = task, context
        self.provider, self.evaluate = provider, evaluate
        self.config, self.rng, self.on_checkpoint = config, random.Random(seed), on_checkpoint
        self.study = Study(task=task, context=context, config={**asdict(config), "seed": seed})
        self.families: list[Family] = []
        self.trials: list[Trial] = []
        self._policies: dict[int, type[Policy]] = {}
        self._sources: set[str] = set()
        self._seed_panel: set[int] | None = None
        self._call_slots: asyncio.Semaphore | None = None
        self._round = {}
        self._expansions = []
        self._proposing = False
        self._proposal_error = None
        self._phase = "discover"
        self._bonus_remaining = 0

    @property
    def best(self) -> type[Policy] | None:
        incumbents = [self.trials[f.best_id - 1] for f in self.families if f.best_id is not None]
        return self._policies[max(incumbents, key=lambda t: t.score).id] if incumbents else None

    def records(self) -> list[SQLModel]:
        return [self.study, *self.families, *self.trials]

    leaderboard_columns = {
        "family": Column("Family"),
        "operation": Column("Operation"),
        "parent": Column("Parent"),
    }

    def _log_candidate(self, row, *, terminal=False):
        status = {
            "planning": "planned",
            "rejected": "repairing",
            "execution_failed": "repairing",
        }.get(row.status, row.status)
        if terminal and status != "evaluated":
            status = "discarded"
        logger.info(
            "%s: %s — %s",
            row.name or f"Attempt {row.id}",
            status,
            row.description,
            extra={
                "progress": dict(
                    kind="candidate",
                    batch_id=str(row.batch),
                    attempt_id=str(row.id),
                    revision=row.repairs,
                    status=status,
                    proposal_done=bool(row.policy_id),
                    policy_id=row.policy_id or "—",
                    name=row.name,
                    description=row.description,
                    score=row.score,
                    error=row.error,
                )
            },
        )

    def _log_leaderboard(self):
        rows = sorted(
            (self.trials[f.best_id - 1] for f in self.families if f.best_id is not None),
            key=lambda row: (-row.score, row.id),
        )
        logger.info(
            "Family leaderboard: %s entries",
            len(rows),
            extra={
                "progress": dict(
                    kind="leaderboard",
                    rows=[
                        dict(
                            id=row.policy_id or str(row.id),
                            name=row.name,
                            description=row.description,
                            score=row.score,
                            generation=row.batch,
                            extras=dict(
                                family=self.families[row.family_id - 1].name,
                                operation=row.kind,
                                parent=row.parent_id,
                            ),
                        )
                        for row in rows
                    ],
                )
            },
        )

    def _checkpoint(self):
        # ponytail: full in-memory history snapshots; use incremental writes for very long runs.
        if self.on_checkpoint is not None:
            self.on_checkpoint(self)

    async def discover(self, failures: list[dict], *, provider, record=None) -> list[FamilyBrief]:
        schema = Families.model_json_schema()
        context = render("families.j2", instance=self, schema=schema, failures=failures)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, Families)
        items = generated.families
        if len(items) != self.config.families:
            raise InvalidCandidate(
                f"Expected exactly {self.config.families} families; received {len(items)}"
            )
        for field in ("name", "mechanism"):
            if len({getattr(item, field).casefold() for item in items}) != len(items):
                raise InvalidCandidate(f"Duplicate family {field}")
        return items

    async def found(self, data: dict, count: int, *, provider, record=None) -> list[Approach]:
        schema = Approaches.model_json_schema()
        context = render("founders.j2", instance=self, schema=schema, data=data, count=count)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, Approaches)
        return self._check_approaches(generated, count)

    async def refine(self, data: dict, count: int, *, provider, record=None) -> list[Approach]:
        schema = Approaches.model_json_schema()
        context = render("refine.j2", instance=self, schema=schema, data=data, count=count)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, Approaches)
        return self._check_approaches(generated, count)

    async def pivot(self, data: dict, count: int, *, provider, record=None) -> list[Approach]:
        schema = Approaches.model_json_schema()
        context = render("pivot.j2", instance=self, schema=schema, data=data, count=count)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, Approaches)
        return self._check_approaches(generated, count)

    async def elect(self, decision: dict, data: dict, *, provider, record=None) -> int:
        schema = Vote.model_json_schema()
        context = render("elect.j2", instance=self, schema=schema, decision=decision, data=data)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        generated = parse(raw, Vote)
        if not 1 <= generated.candidate <= len(decision["proposals"]):
            raise InvalidCandidate("Vote must identify one of the numbered proposals")
        return generated.candidate - 1

    async def implement(self, data: dict, approach: dict, *, provider, record=None) -> type[Policy]:
        schema = _PolicyResponse.model_json_schema()
        context = render("implement.j2", instance=self, schema=schema, data=data, approach=approach)
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    @staticmethod
    def _check_approaches(generated, count):
        items = generated.approaches
        if len(items) != count:
            raise InvalidCandidate(f"Expected exactly {count} approaches; received {len(items)}")
        if len({item.mechanism.casefold() for item in items}) != len(items):
            raise InvalidCandidate("Each approach must have a distinct mechanism")
        return items

    async def _parallel(self, calls):
        if self.config.generation_concurrency == 1:
            return [await call for call in calls]
        tasks = [asyncio.create_task(call) for call in calls]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call(self, operation, *args, family_id=None, trial_id=None, repair=0, record=None):
        call = {} if record is None else record
        call.update(
            operation=operation.__name__, family_id=family_id, trial_id=trial_id, repair=repair
        )
        self.study.calls.append(call)
        if self._call_slots is None:
            self._call_slots = asyncio.Semaphore(self.config.generation_concurrency)
        try:
            async with self._call_slots:
                return await asyncio.wait_for(
                    operation(*args, provider=self.provider, record=call),
                    timeout=self.config.generation_timeout,
                )
        except ValidationError as exc:
            call["error"] = str(exc)
            if "raw" not in call:
                raise
            raise InvalidCandidate(f"Invalid generated output: {exc}") from exc
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _decompose(self, operation, *, data, count, retries, family=None):
        """MAKER proposal and vote waves, bounded globally and repaired independently."""
        wanted = 2 * self.config.decomposition_k - 1
        family_id = None if family is None else family.id
        label = "families" if family is None else family.name
        decision = dict(
            operation=operation.__name__,
            family_id=family_id,
            facet="mechanism family" if family is None else "experimental approach",
            coordinates={} if family is None else {"mechanism family": family.name},
            proposals=[],
            votes=[],
            winner=None,
            outcome="sampling",
        )
        self.study.decompositions.append(decision)
        slots = [None] * wanted
        errors = []
        logger.info(
            "%s: sampling %s partitions (up to %s concurrent model calls)",
            label,
            wanted,
            self.config.generation_concurrency,
        )

        async def sample(index):
            feedback = []
            args = (
                (feedback,)
                if family is None
                else (
                    dict(data, decomposition_failures=feedback),
                    count,
                )
            )
            for repair in range(retries + 1):
                record = dict(partition=index + 1)
                try:
                    slots[index] = await self._call(
                        operation, *args, family_id=family_id, repair=repair, record=record
                    )
                except InvalidPolicy as exc:
                    errors.append(str(exc))
                    # Keep full history in study.calls, not in every future prompt.
                    feedback[:] = [dict(error=str(exc), raw=record.get("raw", ""))]
                    if repair < retries:
                        logger.warning(
                            "%s: repairing partition %s (%s/%s): %s",
                            label,
                            index + 1,
                            repair + 1,
                            retries,
                            exc,
                        )
                    else:
                        logger.warning(
                            "%s: partition %s exhausted repairs: %s", label, index + 1, exc
                        )
                else:
                    decision["proposals"] = [
                        [item.model_dump() for item in proposal]
                        for proposal in slots
                        if proposal is not None
                    ]
                    logger.info(
                        "%s: partition %s accepted (%s/%s ready)",
                        label,
                        index + 1,
                        len(decision["proposals"]),
                        wanted,
                    )
                    self._checkpoint()
                    return
                self._checkpoint()

        await self._parallel(sample(index) for index in range(wanted))
        proposals = [proposal for proposal in slots if proposal is not None]
        if not proposals:
            decision["outcome"] = "generation_exhausted"
            raise InvalidCandidate(errors[-1])
        decision["votes"] = [0] * len(proposals)
        winner = 0
        decision["outcome"] = "single_candidate"
        if len(proposals) > 1:
            tally = Counter()
            decision["outcome"] = "voting"

            async def vote():
                try:
                    return await self._call(self.elect, decision, data, family_id=family_id)
                except InvalidPolicy as exc:
                    logger.warning("%s: flagged discriminator vote: %s", label, exc)
                    return None

            drawn = 0
            while drawn < self.config.decomposition_max_votes:
                wave = min(
                    self.config.decomposition_k,
                    self.config.generation_concurrency,
                    self.config.decomposition_max_votes - drawn,
                )
                choices = await self._parallel(vote() for _ in range(wave))
                drawn += wave
                for choice in choices:
                    if choice is not None:
                        tally[choice] += 1
                        decision["votes"][choice] += 1
                winner = tally.most_common(1)[0][0] if tally else 0
                runner_up = max(v for i, v in enumerate(decision["votes"]) if i != winner)
                logger.info(
                    "%s: discriminator votes %s (%s/%s calls)",
                    label,
                    decision["votes"],
                    drawn,
                    self.config.decomposition_max_votes,
                )
                self._checkpoint()
                # Like MAKER, ingest the entire wave before deciding.
                if tally[winner] - runner_up >= self.config.decomposition_k:
                    decision["outcome"] = "elected"
                    break
            else:
                decision["outcome"] = "fallback"
                logger.warning("%s: vote cap reached; using partition %s", label, winner + 1)
        decision["winner"] = winner + 1
        logger.info("%s: selected partition %s (%s)", label, winner + 1, decision["outcome"])
        self._checkpoint()
        return proposals[winner]

    async def _discover(self):
        self.study.phase = "planning_families"
        logger.info("Discovering %s approach families", self.config.families)
        try:
            briefs = await self._decompose(
                self.discover,
                data={},
                count=self.config.families,
                retries=self.config.discovery_attempts - 1,
            )
        except InvalidPolicy as exc:
            self.study.reason = "generation_exhausted"
            self.study.error = str(exc)
            return
        self.families = [Family(id=i, **brief.model_dump()) for i, brief in enumerate(briefs, 1)]
        self._checkpoint()

    async def _plan_approaches(self, family, operation, data, count):
        return await self._decompose(
            operation,
            data=data,
            count=count,
            retries=self.config.max_repairs,
            family=family,
        )

    async def _plan_founders(self):
        self.study.phase = "planning_approaches"

        async def plan(family):
            try:
                approaches = await self._plan_approaches(
                    family, self.found, self._evidence(family, None), self.config.initial_per_family
                )
            except InvalidPolicy as exc:
                family.status = "generation_exhausted"
                logger.warning("%s: initial decomposition exhausted: %s", family.name, exc)
            else:
                family.initial_approaches = [item.model_dump() for item in approaches]
            self._checkpoint()

        await self._parallel(plan(family) for family in self.families)
        self.study.phase = "search"
        self._checkpoint()

    def _evidence(self, family, parent):
        history = [
            trial.model_dump(exclude={"implementation"})
            for trial in self.trials
            if trial.family_id == family.id and trial.status != "planning"
        ]
        return dict(
            family=family.model_dump(),
            parent={} if parent is None else parent.model_dump(),
            history=history,
            siblings=[
                dict(name=f.name, mechanism=f.mechanism) for f in self.families if f.id != family.id
            ],
        )

    def _select_parent(self, family):
        if not family.frontier:
            return None
        ranked = sorted((self.trials[i - 1] for i in family.frontier), key=lambda t: -t.score)
        if self.rng.random() < self.config.exploration:
            return self.rng.choice(ranked)
        return self.rng.choices(ranked, weights=range(len(ranked), 0, -1))[0]

    def _accept_policy(self, row, policy, parent):
        row.policy_id = None
        self._policies.pop(row.id, None)
        row.name, row.description = policy.name, policy.description
        row.implementation = policy._implementation
        evolution_regions(row.implementation)
        validate_policy(policy)
        if parent is not None:
            check_rewrite(parent.implementation, row.implementation)
        key = ast.dump(ast.parse(row.implementation), include_attributes=False)
        if key in self._sources:
            raise InvalidCandidate("Duplicate program AST across lineages")
        self._sources.add(key)
        self._policies[row.id] = policy
        row.policy_id, row.status, row.error = policy.id, "generated", None
        row.score, row.seed_scores, row.feedback = None, {}, ""
        self._log_candidate(row)
        logger.info(
            "Generated %s — %s",
            policy.name,
            policy.description,
            extra={"event": "policy_generated", "policy_id": policy.id},
        )

    async def _repair(self, row, data, parent):
        while row.repairs < self.config.max_repairs:
            # Keep every failed version before overwriting its program or diagnostic.
            row.revisions.append(row.model_dump(exclude={"revisions"}))
            failed = next(
                (
                    call.get("raw", "")
                    for call in reversed(self.study.calls)
                    if call.get("trial_id") == row.id
                ),
                row.implementation,
            )
            diagnostic = row.model_dump(exclude={"revisions"})
            row.repairs += 1
            row.status = "repairing"
            self._log_candidate(row)
            self._checkpoint()
            logger.warning(
                "Repairing attempt %s (%s/%s): %s",
                row.id,
                row.repairs,
                self.config.max_repairs,
                row.error,
            )
            try:
                policy = await self._call(
                    self.healer.repair,
                    data,
                    diagnostic,
                    failed,
                    family_id=row.family_id,
                    trial_id=row.id,
                    repair=row.repairs,
                )
                self._accept_policy(row, policy, parent)
                return
            except InvalidPolicy as exc:
                row.status, row.error = "rejected", str(exc)

    async def _generate(self, rows, data, parent):
        slots = asyncio.Semaphore(self.config.generation_concurrency)
        logger.info(
            "Generating %s policies (including repairs)",
            len(rows),
            extra={"event": "generation_started", "total": len(rows)},
        )

        async def generate(row):
            async with slots:
                if row.status != "execution_failed":
                    row.status = "generating"
                    self._log_candidate(row)
                    try:
                        approach = row.model_dump(
                            include={"hypothesis", "mechanism", "change", "test"}
                        )
                        policy = await self._call(
                            self.implement, data, approach, family_id=row.family_id, trial_id=row.id
                        )
                        self._accept_policy(row, policy, parent)
                    except InvalidPolicy as exc:
                        row.status, row.error = "rejected", str(exc)
                if row.error is not None:
                    await self._repair(row, data, parent)
                if row.status != "generated":
                    logger.warning(
                        "Discarded attempt %s after %s repairs: %s",
                        row.id,
                        row.repairs,
                        row.error,
                        extra={"event": "proposal_discarded"},
                    )

        await self._parallel(generate(row) for row in rows)

    def _improves(self, child, parent, minimum=0.0):
        differences = [
            score - parent.seed_scores[seed] for seed, score in child.seed_scores.items()
        ]
        error = stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0
        return fmean(differences) > minimum + self.config.uncertainty * error

    def _update(self, family, rows, full_batch):
        measured = sorted((row for row in rows if row.score is not None), key=lambda t: -t.score)
        had_incumbent = family.best_id is not None
        family.last_gain = 0
        for row in measured:
            if family.best_id is None or self._improves(row, self.trials[family.best_id - 1]):
                family.best_id = row.id
        if family.best_id is not None:
            best = self.trials[family.best_id - 1]
            checkpoint = (
                None if family.checkpoint_id is None else self.trials[family.checkpoint_id - 1]
            )
            if checkpoint is None or self._improves(best, checkpoint, self.config.min_delta):
                family.last_gain = 0 if checkpoint is None else best.score - checkpoint.score
                family.checkpoint_id = best.id
                family.stale_batches = 0
                family.pivot_tested = False
            elif full_batch and len(measured) >= math.ceil(len(rows) / 2):
                family.stale_batches += 1
                family.pivot_tested |= rows[0].kind == "pivot"
        if full_batch:
            family.failed_batches = (
                family.failed_batches + 1 if len(measured) < math.ceil(len(rows) / 2) else 0
            )
            if family.failed_batches >= self.config.patience:
                family.status = "generation_exhausted"
            elif (
                had_incumbent
                and family.stale_batches >= self.config.patience
                and family.pivot_tested
            ):
                family.status = "stagnated"
        family.batches += 1
        logger.info(
            "%s: %s, best=%s, stale batches=%s",
            family.name,
            family.status,
            family.best_id,
            family.stale_batches,
        )

    def _cull(self, rows):
        active = [family for family in self.families if family.status == "active"]
        family_ids = {family.id for family in active}
        measured = [row for row in rows if row.score is not None and row.family_id in family_ids]
        if not measured:
            return
        pool = sorted(
            [self.trials[i - 1] for family in active for i in family.frontier] + measured,
            key=lambda t: (-t.score, t.id),
        )
        keep = max(1, math.ceil(len(pool) * (100 - self.config.cull_percent) / 100))
        survivors = pool[:keep]
        for family in active:
            had_candidates = bool(family.frontier) or any(
                t.family_id == family.id for t in measured
            )
            family.frontier = [t.id for t in survivors if t.family_id == family.id][
                : self.config.frontier_per_family
            ]
            if had_candidates and not family.frontier:
                family.status = "culled"
                logger.info("%s: culled (no surviving candidates)", family.name)
        retained = sum(len(family.frontier) for family in active)
        logger.info(
            "Global cull: dropped %s/%s candidates; %s survivors for next decomposition",
            len(pool) - retained,
            len(pool),
            retained,
        )
        self._checkpoint()

    async def _prepare_expansion(self, family, initial_approaches=None):
        parent = self._select_parent(family)
        wanted = self.config.initial_per_family if parent is None else self.config.batch_size
        count = min(wanted, self.config.max_attempts - self.study.attempts)
        if count <= 0:
            return
        if parent is None:
            kind, operation = "founder", self.found
        elif family.stale_batches >= self.config.patience - 1 or family.batches % 2 == 0:
            kind, operation = "pivot", self.pivot
        else:
            kind, operation = "refine", self.refine
        self.study.batches += 1
        rows = [
            Trial(
                id=len(self.trials) + i + 1,
                family_id=family.id,
                parent_id=None if parent is None else parent.id,
                batch=self.study.batches,
                kind=kind,
            )
            for i in range(count)
        ]
        data = self._evidence(family, parent)
        self._expansions.append((family, rows, count == wanted, data, parent))
        logger.info(
            "Starting batch %s",
            rows[0].batch,
            extra={
                "progress": dict(
                    kind="batch_started",
                    batch_id=str(rows[0].batch),
                    label=f"Batch {rows[0].batch}",
                    total_candidates=count,
                )
            },
        )
        self.trials.extend(rows)
        for row in rows:
            self._log_candidate(row)
        self.study.attempts += count
        logger.info("%s: %s batch %s (%s attempts)", family.name, kind, self.study.batches, count)
        self._checkpoint()
        try:
            approaches = (
                [Approach(**item) for item in initial_approaches[:count]]
                if initial_approaches is not None
                else await self._plan_approaches(family, operation, data, count)
            )
        except InvalidPolicy as exc:
            approaches = []
            for row in rows:
                row.status = "rejected"
                row.error = str(exc)
        if approaches:
            for row, approach in zip(rows, approaches):
                for key, value in approach.model_dump().items():
                    setattr(row, key, value)
            await self._generate(rows, data, parent)
            self._checkpoint()

    def _sample_family(self):
        active = [f for f in self.families if f.status == "active"]
        ranked = sorted(
            active,
            key=lambda f: -math.inf if f.best_id is None else self.trials[f.best_id - 1].score,
            reverse=True,
        )
        # Rank weights handle negative rewards; recent confirmed progress gets extra trials.
        weights = [
            rank + (len(ranked) if f.last_gain > 0 else 0)
            for rank, f in zip(range(len(ranked), 0, -1), ranked)
        ]
        return self.rng.choices(ranked, weights=weights)[0]

    @property
    def done(self):
        return (
            not self._proposing
            and not self._round
            and not self._expansions
            and self.study.reason in ("completed", "budget_exhausted", "generation_exhausted")
        )

    def _finish_study(self):
        if self._expansions or self._round:
            return
        if self.study.reason == "generation_exhausted":
            return
        active = any(f.status == "active" for f in self.families)
        if not active or self.study.attempts >= self.config.max_attempts:
            self.study.reason = "completed" if self.families and not active else "budget_exhausted"

    def _settle_expansions(self):
        if self._proposal_error is not None:
            return
        if any(
            row.status == "execution_failed" and row.repairs < self.config.max_repairs
            for _, rows, _, _, _ in self._expansions
            for row in rows
        ):
            return
        all_rows = []
        for family, rows, full_batch, _, _ in self._expansions:
            for row in rows:
                self._log_candidate(row, terminal=True)
            self._update(family, rows, full_batch)
            all_rows.extend(rows)
            logger.info(
                "Finished batch %s",
                rows[0].batch,
                extra={
                    "progress": dict(
                        kind="batch_finished", batch_id=str(rows[0].batch), status="completed"
                    )
                },
            )
        self._expansions.clear()
        self._log_leaderboard()
        self._cull(all_rows)
        if self._phase == "explore":
            self._bonus_remaining = self.config.bonus_batches
            self._phase = "bonus" if self._bonus_remaining else "explore"
        else:
            self._bonus_remaining -= 1
            if not self._bonus_remaining:
                self._phase = "explore"
        self._finish_study()
        self._checkpoint()

    async def propose(self):
        if self._proposing or self._round:
            raise RuntimeError("Previous proposal round is still outstanding")
        if self.done:
            return []
        if self._proposal_error is not None:
            raise self._proposal_error
        self._proposing = True
        try:
            if self._phase == "discover":
                self.study.reason = "running"
                if self.config.max_attempts:
                    await self._discover()
                    if self.study.reason == "generation_exhausted":
                        return []
                    await self._plan_founders()
                self._phase = "explore"
            while True:
                if self._expansions:
                    await self._parallel(
                        self._generate(
                            [
                                row
                                for row in rows
                                if row.status == "execution_failed"
                                and row.repairs < self.config.max_repairs
                            ],
                            data,
                            parent,
                        )
                        for _, rows, _, data, parent in self._expansions
                    )
                else:
                    self._finish_study()
                    if self.study.reason in (
                        "completed",
                        "budget_exhausted",
                        "generation_exhausted",
                    ):
                        return []
                    if self._phase == "explore":
                        await self._parallel(
                            self._prepare_expansion(
                                family, family.initial_approaches if family.batches == 0 else None
                            )
                            for family in self.families
                            if family.status == "active"
                        )
                    else:
                        await self._prepare_expansion(self._sample_family())
                self._round = {
                    row.policy_id: row
                    for _, rows, _, _, _ in self._expansions
                    for row in rows
                    if row.status == "generated"
                }
                if self._round:
                    for row in self._round.values():
                        row.status = "evaluating"
                        self._log_candidate(row)
                    return [self._policies[row.id] for row in self._round.values()]
                self._settle_expansions()
        except BaseException as exc:
            self._proposal_error = exc
            if isinstance(exc, Exception):
                self._round = {
                    row.policy_id: row for row in self.trials if row.status == "generated"
                }
                if self._round:
                    for row in self._round.values():
                        row.status = "evaluating"
                        self._log_candidate(row)
                    return [self._policies[row.id] for row in self._round.values()]
            raise
        finally:
            self._proposing = False

    def update(self, results):
        panel = validate_results(results, self._round, seed_panel=self._seed_panel)
        score_panels = {id: episode_scores(r) for id, r in results.items()}
        for id, result in results.items():
            error = episode_error(result)
            row = self._round[id]
            row.feedback = ""
            if error is not None:
                row.status, row.error = "execution_failed", error
            elif not result:
                row.status, row.error = "rejected", "Evaluation rejected"
            else:
                row.seed_scores = {str(seed): score for seed, score in score_panels[id].items()}
                row.score, row.status = fmean(score_panels[id].values()), "evaluated"
            self._log_candidate(row)
        self._seed_panel = panel
        self._round.clear()
        self._checkpoint()
        self._settle_expansions()

    async def run(self) -> Study:
        """Compatibility wrapper around the shared proposal/evaluation runner."""
        if self.evaluate is None:
            raise ValueError("Pass an evaluator to rsikit.search(optimizer, evaluate)")
        logger.info(
            "Starting LineageSearch",
            extra={
                "progress": dict(
                    kind="search_started",
                    optimizer="LineageSearch",
                    total_candidates=self.config.max_attempts,
                    columns=self.leaderboard_columns,
                    resumed=False,
                )
            },
        )
        try:
            await search(self, self.evaluate, on_checkpoint=lambda agent: agent._checkpoint())
            return self.study
        except BaseException as exc:
            self.study.reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            self.study.error = f"{type(exc).__name__}: {exc}"
            try:
                self._checkpoint()
            except BaseException:
                logger.exception("Checkpoint failed while handling search error")
            raise
        finally:
            logger.info(
                "Search %s",
                self.study.reason,
                extra={
                    "progress": dict(
                        kind="search_finished",
                        status="failed"
                        if self.study.reason == "error"
                        else "cancelled"
                        if self.study.reason == "cancelled"
                        else "completed"
                        if self.study.attempts >= self.config.max_attempts
                        else "stopped",
                        reason=self.study.reason,
                    )
                },
            )
