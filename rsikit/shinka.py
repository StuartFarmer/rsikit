"""ShinkaEvolve archive policy and Slick proposal operations for the RSIKit lifecycle."""

import math
import random
import statistics
from collections.abc import Awaitable, Callable, Sequence
from decimal import Decimal

from pydantic import BaseModel, Field, StrictBool, ValidationError
from slick import prompt

from .edits import InvalidCandidate, ProposalRejected, apply_diff, mutable_regions, validate_source
from .proposer import PromptProposer, _RecordedProvider, recent_context
from .selection import top_candidates
from .strategies import Candidate, SequentialStrategy


class ShinkaEvolve(SequentialStrategy):
    """Bounded islands, novelty rejection, and model credit from measured improvement.

    The proposer reads context['model'] and context['patch']; the host owns
    evaluation, provider construction, deadlines, and persistence. Optional
    novelty and reflection callbacks can be supplied by ShinkaProposer.
    """

    def __init__(
        self,
        initial,
        evaluation,
        propose,
        *,
        objective="score",
        maximize=True,
        islands=2,
        archive_size=40,
        elite_ratio=0.3,
        top_k=2,
        inspirations=4,
        parent_selection="weighted",
        selection_pressure=10.0,
        power_alpha=1.0,
        models=1,
        exploration=1.0,
        patch_types=(("diff", 0.45), ("full", 0.45), ("cross", 0.1)),
        max_proposals=3,
        embed: Callable[[str], Awaitable[Sequence[float]]] | None = None,
        novelty: Callable[[str, Candidate, float], Awaitable[tuple[bool, str]]] | None = None,
        novelty_threshold=0.95,
        reflect: Callable[..., Awaitable[list[str]]] | None = None,
        meta_interval=10,
        migration_interval=10,
        migration_rate=0.1,
        seed=0,
    ):
        super().__init__(initial, evaluation, self._propose, objective=objective, maximize=maximize)
        self.proposer = propose
        self.rng = random.Random(seed)
        self.islands = [[self.best] for _ in range(islands)]
        self.archive_size, self.elite_ratio = archive_size, elite_ratio
        self.top_k, self.inspiration_count = top_k, inspirations
        self.parent_selection, self.selection_pressure = parent_selection, selection_pressure
        self.power_alpha, self.exploration = power_alpha, exploration
        self.patch_types, self.max_proposals = patch_types, max_proposals
        self.embed, self.novelty, self.novelty_threshold = embed, novelty, novelty_threshold
        self.reflect, self.meta_interval = reflect, meta_interval
        self.migration_interval, self.migration_rate = migration_interval, migration_rate
        self.model_gains: list[list[Decimal]] = [[] for _ in range(models)]
        self.offspring: dict[int, int] = {}
        self.embeddings: dict[str, tuple[float, ...]] = {}
        self.scratchpad: tuple[str, ...] = ()
        self._reflected = 0
        self.proposals: list[dict] = []
        self.events: list[dict] = []

    def selection_weights(self, population) -> list[float]:
        scores = [self._score(c.evaluation) * (1 if self.maximize else -1) for c in population]
        if self.parent_selection == "uniform":
            return [1.0] * len(scores)
        if self.parent_selection == "best":
            return [float(s == max(scores)) for s in scores]
        if self.parent_selection == "power":
            ranks = {
                i: rank
                for rank, i in enumerate(sorted(range(len(scores)), key=lambda i: -scores[i]), 1)
            }
            return [ranks[i] ** -self.power_alpha for i in range(len(scores))]
        median = statistics.median_low(scores) / 2 + statistics.median_high(scores) / 2
        weights = []
        for candidate, score in zip(population, scores):
            x = self.selection_pressure * (score / 2 - median / 2) * 2
            sigmoid = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
            weights.append(sigmoid / (1 + self.offspring.get(candidate.id, 0)))
        return weights

    def model_weights(self) -> list[float]:
        if any(not gains for gains in self.model_gains):
            return [float(not gains) for gains in self.model_gains]
        scale = max(g for gains in self.model_gains for g in gains) or 1
        total = sum(map(len, self.model_gains))
        # ponytail: rescan rewards for normalization; aggregate if long runs make this costly.
        return [
            statistics.fmean(math.expm1(g / scale) for g in gains) / math.expm1(1)
            + self.exploration * math.sqrt(2 * math.log(total + 1) / len(gains))
            for gains in self.model_gains
        ]

    def select_parent(self) -> Candidate:
        island = self.rng.randrange(len(self.islands))
        population = self.islands[island]
        parent = self.rng.choices(population, self.selection_weights(population))[0]
        ranked = top_candidates(
            [c for c in population if c.id != parent.id],
            len(population),
            objective=self.objective,
            maximize=self.maximize,
        )
        rest = ranked[self.top_k :]
        inspirations = ranked[: self.top_k] + self.rng.sample(
            rest, min(len(rest), self.inspiration_count)
        )
        weights = self.model_weights()
        model = self.rng.choices(range(len(weights)), weights if any(weights) else None)[0]
        patch = self.rng.choices(
            [p for p, _ in self.patch_types], [w for _, w in self.patch_types]
        )[0]
        parents = (parent, self.rng.choice(ranked)) if patch == "cross" and ranked else (parent,)
        if patch == "cross" and len(parents) == 1:
            patch = "full"
        self.context = {
            "operation": "shinkaevolve",
            "patch": patch,
            "model": model,
            "parents": parents,
            "inspirations": tuple(inspirations),
            "island": island,
            "objective": self.objective,
            "maximize": self.maximize,
            "guidance": self.scratchpad,
            "failures": [],
        }
        return parent

    async def _propose(self, parent, history) -> str:
        completed = len(history) - 1
        if (
            self.reflect
            and self.meta_interval
            and completed
            and completed % self.meta_interval == 0
            and self._reflected != completed
        ):
            event = {"meta_after": completed}
            self.events.append(event)
            try:
                recent = history[-self.meta_interval :]
                evidence = tuple(
                    {
                        c.id: c
                        for c in (history[0], *(history[c.parent_id] for c in recent), *recent)
                    }.values()
                )
                self.scratchpad = tuple(
                    await self.reflect(
                        evidence, self.scratchpad, objective=self.objective, maximize=self.maximize
                    )
                )
                event["recommendations"] = self.scratchpad
            except ProposalRejected as exc:
                event["error"] = str(exc)
            self._reflected = completed
        self.context["guidance"] = self.scratchpad
        source = ""
        for _ in range(self.max_proposals):
            record = {
                "candidate_id": len(history),
                "model": self.context["model"],
                "patch": self.context["patch"],
            }
            self.proposals.append(record)
            try:
                source = await self.proposer(parent, history)
                record["source"] = source
                validate_source(source, parent.source)
                await self.check_novelty(source, self.islands[self.context["island"]], record)
                return source
            except InvalidCandidate as exc:
                source = getattr(exc, "source", source)
                record.update(source=source, error=str(exc))
                self.context["failures"].append({"source": source, "error": str(exc)})
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
        raise ProposalRejected(source, "Proposal budget exhausted: " + self.proposals[-1]["error"])

    async def embedding(self, source) -> tuple[float, ...]:
        if source not in self.embeddings:
            mutable = "\n".join(source[start:end] for start, end in mutable_regions(source))
            vector = tuple(await self.embed(mutable))
            norm = math.hypot(*vector)
            if not vector or not math.isfinite(norm) or norm == 0:
                raise ValueError("Embedding must be a finite nonzero vector")
            self.embeddings[source] = tuple(v / norm for v in vector)
        return self.embeddings[source]

    async def check_novelty(self, source, population, record) -> None:
        if any(c.source == source for c in population):
            raise InvalidCandidate("Duplicate island program")
        if self.embed is None:
            return
        vector = await self.embedding(source)
        similarities = []
        for candidate in population:
            reference = await self.embedding(candidate.source)
            if len(vector) != len(reference):
                raise ValueError("Embedding dimensions changed")
            similarities.append(sum(a * b for a, b in zip(vector, reference)))
        index = max(range(len(population)), key=similarities.__getitem__)
        similarity, nearest = similarities[index], population[index]
        record.update(similarity=similarity, nearest_id=nearest.id)
        if similarity > self.novelty_threshold:
            novel, reason = (
                await self.novelty(source, nearest, similarity)
                if self.novelty
                else (False, "Embedding similarity exceeds threshold")
            )
            record.update(novel=novel, novelty_reason=reason)
            if not novel:
                raise InvalidCandidate(reason)

    def observe(self, candidate: Candidate) -> None:
        gain = Decimal(0)
        if self.pending is not None:
            self.offspring[candidate.parent_id] = self.offspring.get(candidate.parent_id, 0) + 1
        if candidate.evaluation.valid:
            direction = 1 if self.maximize else -1
            parent = self.history[candidate.parent_id]
            baseline = max(direction * self._score(c.evaluation) for c in (parent, self.history[0]))
            gain = max(
                Decimal(direction * self._score(candidate.evaluation)) - Decimal(baseline),
                Decimal(0),
            )
            island = self.islands[self.context["island"]]
            island.append(candidate)
            self.prune(island)
        self.model_gains[self.context["model"]].append(gain)
        if self.migration_interval and len(self.history) % self.migration_interval == 0:
            self.migrate()

    def _record(self, candidate: Candidate) -> None:
        super()._record(candidate)
        self.selections[-1].update(model=self.context["model"], patch=self.context["patch"])

    def prune(self, island) -> None:
        if len(island) > self.archive_size:
            ranked = top_candidates(
                island, len(island), objective=self.objective, maximize=self.maximize
            )
            count = max(1, math.ceil(self.archive_size * self.elite_ratio))
            island[:] = ranked[:count] + self.rng.sample(ranked[count:], self.archive_size - count)

    def migrate(self) -> None:
        if len(self.islands) < 2:
            return
        batches = []
        for island in self.islands:
            best = top_candidates(island, 1, objective=self.objective, maximize=self.maximize)[0]
            pool = [c for c in island if c.id != best.id]
            batches.append(
                self.rng.sample(pool, min(len(pool), math.floor(len(island) * self.migration_rate)))
            )
        for origin, migrants in enumerate(batches):
            target = (origin + 1) % len(self.islands)
            island = self.islands[target]
            additions = [c for c in migrants if all(c.id != other.id for other in island)]
            island.extend(additions)
            self.prune(island)
            if additions:
                self.events.append(
                    {"migration": [c.id for c in additions], "from": origin, "to": target}
                )


class Novelty(BaseModel, extra="forbid"):
    novel: StrictBool
    reason: str = Field(min_length=1)


class Recommendations(BaseModel, extra="forbid"):
    recommendations: list[str]


class ShinkaProposer(PromptProposer):
    """Separate diff/full/crossover, novelty, and meta prompts; reuse RSIKit repair.

    Configure Slick's root to the checkout root. Providers are selected through
    strategy.context['model']; auxiliary calls and repair use their own providers.
    """

    def __init__(
        self,
        task,
        provider,
        *,
        ensemble=(),
        novelty_provider=None,
        meta_provider=None,
        max_recommendations=5,
        instructions=None,
    ):
        super().__init__(task, provider, instructions=instructions)
        self.models = tuple(ensemble) or (provider,)
        self.novelty_provider = provider if novelty_provider is None else novelty_provider
        self.meta_provider = provider if meta_provider is None else meta_provider
        self.max_recommendations = max_recommendations

    async def __call__(self, parent, history, *, context=None) -> str:
        context = context or {}
        patch, model = context.get("patch", "full"), context.get("model", 0)
        record = {
            "candidate_id": len(history),
            "operation": "shinkaevolve",
            "patch": patch,
            "model": model,
            "parent_ids": [c.id for c in context.get("parents", (parent,))],
        }
        self.records.append(record)
        data = {
            "parents": [c.model_dump(mode="json") for c in context.get("parents", (parent,))],
            "inspirations": [c.model_dump(mode="json") for c in context.get("inspirations", ())],
            "outcomes": [c.model_dump(mode="json") for c in recent_context(history)],
            "guidance": context.get("guidance", ()),
            "failures": context.get("failures", ()),
            "evidence": context.get("evidence", ()),
            "instruction": self.instructions.get(patch, ""),
            "objective": context.get("objective", "score"),
            "direction": "maximize" if context.get("maximize", True) else "minimize",
        }
        operation = {"diff": self.diff, "full": self.rewrite, "cross": self.crossover}[patch]
        raw = ""
        try:
            raw = await operation(data, provider=_RecordedProvider(self.models[model], record))
            source = apply_diff(parent.source, raw) if patch == "diff" else raw
            validate_source(source, parent.source)
            record["source"] = source
            return source
        except InvalidCandidate as exc:
            record["error"] = str(exc)
            raise ProposalRejected(raw, str(exc)) from exc
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def _structured(self, operation, args, provider, record, source=""):
        self.records.append(record)
        try:
            return await operation(*args, provider=_RecordedProvider(provider, record))
        except ValidationError as exc:
            record["error"] = str(exc)
            if "error" in record.get("calls", [{}])[-1]:
                raise
            raise ProposalRejected(source, f"Invalid generated JSON: {exc}") from exc
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def assess_novelty(self, source, nearest, similarity) -> tuple[bool, str]:
        record = {"operation": "novelty", "nearest_id": nearest.id, "similarity": similarity}
        result = await self._structured(
            self.judge, (source, nearest, similarity), self.novelty_provider, record, source
        )
        record["judgment"] = result.model_dump(mode="json")
        return result.novel, result.reason

    async def summarize(self, recent, previous, *, objective="score", maximize=True) -> list[str]:
        record = {
            "operation": "meta",
            "candidate_ids": [c.id for c in recent],
            "objective": objective,
            "maximize": maximize,
        }
        result = await self._structured(
            self.reflect,
            (recent, previous, objective, "maximize" if maximize else "minimize"),
            self.meta_provider,
            record,
        )
        record["recommendations"] = result
        return result

    @prompt(template="rsikit/prompts/shinka/diff.j2")
    async def diff(self, data, *, generated: str) -> str:
        return generated

    @prompt(template="rsikit/prompts/shinka/full.j2")
    async def rewrite(self, data, *, generated: str) -> str:
        return generated

    @prompt(template="rsikit/prompts/shinka/cross.j2")
    async def crossover(self, data, *, generated: str) -> str:
        return generated

    @prompt(template="rsikit/prompts/shinka/novelty.j2", output_type=Novelty)
    async def judge(self, source, nearest, similarity, *, generated: Novelty) -> Novelty:
        return generated

    @prompt(template="rsikit/prompts/shinka/meta.j2", output_type=Recommendations)
    async def reflect(
        self, recent, previous, objective, direction, *, generated: Recommendations
    ) -> list[str]:
        if len(generated.recommendations) > self.max_recommendations or any(
            not s.strip() for s in generated.recommendations
        ):
            raise ProposalRejected(
                "", "Recommendations must be nonblank and within the configured limit"
            )
        return generated.recommendations
