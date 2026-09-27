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
from slick import prompt
from slick.providers import Provider
from sqlmodel import SQLModel

from rsikit import Policy
from rsikit.generation import WORKER_LIBRARIES, RecordingProvider
from rsikit.generation.edits import InvalidCandidate, Program, check_program, check_rewrite
from rsikit.policy import _policy_class

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


@dataclass(frozen=True)
class Measurement:
    """Comparable per-seed rewards, or an explicit candidate execution failure.

    An infrastructure failure must raise instead of returning a failure here.
    """

    scores: dict[int, float]
    feedback: str = ""
    failure: str | None = None


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
        evaluate: Callable[[Sequence[type[Policy]]], Awaitable[dict[str, Measurement]]],
        *,
        context: str = "",
        config: Config = Config(),
        seed: int = 0,
        on_checkpoint: Callable[[LineageSearch], None] | None = None,
    ):
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
        self._evaluation_lock = asyncio.Lock()

    @property
    def best(self) -> type[Policy] | None:
        incumbents = [self.trials[f.best_id - 1] for f in self.families if f.best_id is not None]
        return self._policies[max(incumbents, key=lambda t: t.score).id] if incumbents else None

    def records(self) -> list[SQLModel]:
        return [self.study, *self.families, *self.trials]

    def _checkpoint(self):
        # ponytail: full in-memory history snapshots; use incremental writes for very long runs.
        if self.on_checkpoint is not None:
            self.on_checkpoint(self)

    @prompt(template="families.j2", output_type=Families)
    async def discover(self, failures: list[dict], *, generated: Families) -> list[FamilyBrief]:
        items = generated.families
        if len(items) != self.config.families:
            raise InvalidCandidate(
                f"Expected exactly {self.config.families} families; received {len(items)}"
            )
        for field in ("name", "mechanism"):
            if len({getattr(item, field).casefold() for item in items}) != len(items):
                raise InvalidCandidate(f"Duplicate family {field}")
        return items

    @prompt(template="founders.j2", output_type=Approaches)
    async def found(self, data: dict, count: int, *, generated: Approaches) -> list[Approach]:
        return self._check_approaches(generated, count)

    @prompt(template="refine.j2", output_type=Approaches)
    async def refine(self, data: dict, count: int, *, generated: Approaches) -> list[Approach]:
        return self._check_approaches(generated, count)

    @prompt(template="pivot.j2", output_type=Approaches)
    async def pivot(self, data: dict, count: int, *, generated: Approaches) -> list[Approach]:
        return self._check_approaches(generated, count)

    @prompt(template="elect.j2", output_type=Vote)
    async def elect(self, decision: dict, data: dict, *, generated: Vote) -> int:
        if not 1 <= generated.candidate <= len(decision["proposals"]):
            raise InvalidCandidate("Vote must identify one of the numbered proposals")
        return generated.candidate - 1

    @prompt(template="implement.j2", output_type=Program)
    async def implement(self, data: dict, approach: dict, *, generated: Program) -> Program:
        return generated

    @prompt(template="repair.j2", output_type=Program)
    async def repair(self, data: dict, trial: dict, failed: str, *, generated: Program) -> Program:
        return generated

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
                    operation(*args, provider=RecordingProvider(self.provider, call)),
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
                except InvalidCandidate as exc:
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
                except InvalidCandidate as exc:
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
        except InvalidCandidate as exc:
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
            except InvalidCandidate as exc:
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

    def _accept_program(self, row, program, parent):
        row.policy_id = None
        self._policies.pop(row.id, None)
        row.name, row.description = program.name, program.description
        row.implementation = program.implementation
        check_program(row.implementation)
        if parent is not None:
            check_rewrite(parent.implementation, row.implementation)
        key = ast.dump(ast.parse(row.implementation), include_attributes=False)
        if key in self._sources:
            raise InvalidCandidate("Duplicate program AST across lineages")
        self._sources.add(key)
        policy = _policy_class(row.name, row.implementation, row.description)
        self._policies[row.id] = policy
        row.policy_id, row.status, row.error = policy.id, "generated", None
        row.score, row.seed_scores, row.feedback = None, {}, ""
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
            self._checkpoint()
            logger.warning(
                "Repairing attempt %s (%s/%s): %s",
                row.id,
                row.repairs,
                self.config.max_repairs,
                row.error,
            )
            try:
                program = await self._call(
                    self.repair,
                    data,
                    diagnostic,
                    failed,
                    family_id=row.family_id,
                    trial_id=row.id,
                    repair=row.repairs,
                )
                self._accept_program(row, program, parent)
                return
            except InvalidCandidate as exc:
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
                    try:
                        approach = row.model_dump(
                            include={"hypothesis", "mechanism", "change", "test"}
                        )
                        program = await self._call(
                            self.implement, data, approach, family_id=row.family_id, trial_id=row.id
                        )
                        self._accept_program(row, program, parent)
                    except InvalidCandidate as exc:
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

    async def _measure(self, rows, data, parent):
        while any(row.status == "generated" for row in rows):
            await self._evaluate_pending(rows)
            failed = [
                row
                for row in rows
                if row.status == "execution_failed" and row.repairs < self.config.max_repairs
            ]
            self._checkpoint()
            if failed:
                await self._generate(failed, data, parent)
                self._checkpoint()

    async def _evaluate_pending(self, rows):
        generated = [row for row in rows if row.status == "generated"]
        if not generated:
            return
        # The evaluator owns its worker pool; model calls in other families keep running.
        async with self._evaluation_lock:
            for row in generated:
                row.status = "evaluating"
            self._checkpoint()
            results = await self.evaluate([self._policies[row.id] for row in generated])
        if set(results) != {row.policy_id for row in generated}:
            raise ValueError("Evaluator must return exactly the requested policy IDs")
        panel = self._seed_panel
        for result in results.values():
            if result.failure is not None:
                continue
            if not result.scores or any(not math.isfinite(s) for s in result.scores.values()):
                raise ValueError("Measurements must contain finite per-seed scores")
            if panel is not None and set(result.scores) != panel:
                raise ValueError("Every measurement must use the same seed panel")
            panel = set(result.scores)
        self._seed_panel = panel
        for row in generated:
            result = results[row.policy_id]
            row.feedback = result.feedback
            if result.failure is not None:
                row.status, row.error = "execution_failed", result.failure
                logger.warning("Policy %s failed: %s", row.name, row.error)
            else:
                row.seed_scores = {str(seed): score for seed, score in result.scores.items()}
                row.score, row.status = fmean(result.scores.values()), "evaluated"

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

    async def _expand(self, family, initial_approaches=None):
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
        self.trials.extend(rows)
        self.study.attempts += count
        logger.info("%s: %s batch %s (%s attempts)", family.name, kind, self.study.batches, count)
        self._checkpoint()
        try:
            approaches = (
                [Approach(**item) for item in initial_approaches[:count]]
                if initial_approaches is not None
                else await self._plan_approaches(family, operation, data, count)
            )
        except InvalidCandidate as exc:
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
            await self._measure(rows, data, parent)
        self._update(family, rows, count == wanted)
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

    async def run(self) -> Study:
        """Elect the complete initial tree, then expand by measured fitness until stopped."""
        self.study.reason = "running"
        try:
            if self.config.max_attempts > 0:
                await self._discover()
            if self.study.reason == "generation_exhausted":
                return self.study
            await self._plan_founders()
            while self.study.attempts < self.config.max_attempts:
                active = [f for f in self.families if f.status == "active"]
                if not active:
                    break
                start = len(self.trials)
                await self._parallel(
                    self._expand(
                        family,
                        family.initial_approaches if family.batches == 0 else None,
                    )
                    for family in active
                )
                self._cull(self.trials[start:])
                for _ in range(self.config.bonus_batches):
                    if self.study.attempts >= self.config.max_attempts or all(
                        f.status != "active" for f in self.families
                    ):
                        break
                    start = len(self.trials)
                    await self._expand(self._sample_family())
                    self._cull(self.trials[start:])
            self.study.reason = (
                "completed"
                if self.families and all(f.status != "active" for f in self.families)
                else "budget_exhausted"
            )
            return self.study
        except BaseException as exc:
            self.study.reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            self.study.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._checkpoint()
