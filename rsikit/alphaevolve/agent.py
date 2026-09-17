"""Evolve Gymnasium policy classes using evaluated islands and Slick generation.

The island founding/reset policy adapts the official FunSearch program database;
see NOTICE. AlphaEvolve's unpublished database details are explicit local choices.
"""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider, ProviderError

from .edits import (
    InvalidCandidate,
    Mutation,
    Program,
    apply_edits,
    check_program,
    check_rewrite,
    evolution_regions,
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
class Evaluation:
    """Measured objectives (all maximized), diagnostics, and a discrete diversity cell."""

    metrics: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    cell: tuple[int, ...] = ()
    error: str = ""


Evaluator = Callable[[str], Awaitable[Evaluation]]


@dataclass(frozen=True)
class EvaluationStage:
    evaluate: Evaluator
    minimums: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    id: int
    content: str
    metrics: dict[str, float]
    feedback: str
    cell: tuple[int, ...]
    parent_id: int | None = None


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
    evaluation_timeout: float | None = None


class AlphaEvolve:
    """Own one asynchronous search; callers own model retries and isolated execution.

    Configure Slick's process-global template root before use. Each prompt call
    has independent context and exactly one provider; no mutable Session is shared.
    Use a fresh instance for each run. `run` returns the best on its target metric;
    `best_by_metric`, `programs`, `islands`, and `attempts` expose the other results.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Evaluator,
        *,
        context: str = "",
        ensemble: Sequence[tuple[Provider, float]] = (),
        stages: Sequence[EvaluationStage] = (),
        prompt_variants: Sequence[tuple[str, float]] = (("", 1.0),),
        config: Config = Config(),
    ):
        self.task, self.context, self.evaluate = task, context, evaluate
        self.models = tuple(ensemble) or ((provider, 1.0),)
        self.stages = (*stages, EvaluationStage(evaluate))
        self.variants = tuple(prompt_variants)
        self.config = config
        # ponytail: in-memory run history; persist externally for runs exceeding RAM.
        self.programs: list[Candidate] = []
        self.islands: list[dict[tuple[tuple[int, ...], str], Candidate]] = [
            {} for _ in range(config.islands)
        ]
        self.best_by_metric: dict[str, Candidate] = {}
        self.prompt_ideas = [PromptIdea("")]
        self.attempts: list[dict] = []
        self.events: list[dict] = []
        self.generation_calls = self.meta_calls = self.evaluations = self.completed = 0

    @prompt(template="mutate.j2", output_type=Mutation)
    async def mutate(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Mutation,
    ) -> Mutation:
        """Generate validated exact edits; the search loop enforces edit boundaries."""
        return generated

    @prompt(template="rewrite.j2", output_type=Program)
    async def rewrite(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        guidance: str,
        failures: list[dict],
        *,
        generated: Program,
    ) -> Program:
        """Generate a complete Solution program as a structured response."""
        return generated

    @prompt(template="evolve_prompt.j2", output_type=Guidance)
    async def evolve_prompt(
        self,
        parent: Candidate,
        ideas: list[dict],
        failures: list[dict],
        *,
        generated: Guidance,
    ) -> str:
        """Propose task-specific guidance to be scored by subsequent offspring."""
        return generated.instruction

    async def run(
        self,
        initial: str,
        *,
        attempts: int = 100,
        concurrency: int = 4,
        target_metric: str | None = None,
        seed: int = 0,
    ) -> Candidate:
        """Evaluate the seed, then spend exactly `attempts` candidate attempts.

        Seed evaluation is additional. Meta calls are additional and counted
        separately. Known candidate/provider failures consume attempts; unexpected
        errors abort and cancel workers. Completion order influences parallel runs.
        """
        self.rng = random.Random(seed)
        await self.initialize(initial, target_metric)
        work = iter(range(1, attempts + 1))
        workers = [
            asyncio.create_task(self._worker(work)) for _ in range(min(concurrency, attempts))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        return self.best_by_metric[self.target_metric]

    async def initialize(self, initial: str, target_metric: str | None) -> None:
        evolution_regions(initial)
        check_program(initial)
        evaluation = await self._evaluate(initial, {})
        self.objectives = tuple(evaluation.metrics)
        self.target_metric = target_metric or self.objectives[0]
        evaluation.metrics[self.target_metric]
        candidate = Candidate(0, initial, evaluation.metrics, evaluation.feedback, evaluation.cell)
        self.programs.append(candidate)
        # Official FunSearch policy: every island starts from the evaluated seed.
        for island in self.islands:
            self._register(candidate, island)

    async def _worker(self, work) -> None:
        for attempt_id in work:
            await self._step(attempt_id)
            self.completed += 1
            if self.config.reset_interval and self.completed % self.config.reset_interval == 0:
                self.reset_islands()

    def sample(self) -> tuple[int, Candidate, list[Candidate], str]:
        """Sample a cell champion or a metric champion, then distinct inspirations."""
        island_id = self.rng.randrange(len(self.islands))
        population = list({p.id: p for p in self.islands[island_id].values()}.values())
        objective = self.rng.choice(self.objectives)
        parent = (
            self.rng.choice(population)
            if self.rng.random() < self.config.exploration
            else max(population, key=lambda p: p.metrics[objective])
        )
        pool = {p.id: p for p in (*population, *self.best_by_metric.values()) if p.id != parent.id}
        inspirations = self.rng.sample(
            list(pool.values()), min(len(pool), self.config.inspirations)
        )
        return island_id, parent, inspirations, objective

    async def _step(self, attempt_id: int) -> None:
        island_id, parent, inspirations, objective = self.sample()
        model_id = self.rng.choices(range(len(self.models)), [w for _, w in self.models])[0]
        provider = self.models[model_id][0]
        failures = [
            {
                **{key: row[key] for key in ("id", "raw", "error") if key in row},
                "evaluations": [
                    {"metrics": stage.metrics, "feedback": stage.feedback, "error": stage.error}
                    for stage in row.get("stages", [])
                ],
            }
            for row in self.attempts
            if row.get("error")
        ][-3:]
        record = {
            "id": attempt_id,
            "parent_id": parent.id,
            "island": island_id,
            "model": model_id,
            "objective": objective,
            "inspirations": tuple(p.id for p in inspirations),
            "status": "running",
        }
        self.attempts.append(record)
        try:
            idea = await self._choose_guidance(attempt_id, parent, failures, provider, record)
            idea.uses += 1
            variant = self.rng.choices(
                [v for v, _ in self.variants], [w for _, w in self.variants]
            )[0]
            guidance = "\n".join((variant, idea.instruction))
            record["guidance"] = guidance
            operation = {"diff": self.mutate, "rewrite": self.rewrite}[self.config.mode]
            self.generation_calls += 1
            try:
                proposal = await asyncio.wait_for(
                    operation(
                        parent,
                        inspirations,
                        guidance,
                        failures,
                        provider=_RecordedProvider(provider, record, "raw"),
                    ),
                    self.config.generation_timeout,
                )
            except ValidationError as exc:
                raise InvalidCandidate(f"Invalid generated proposal: {exc}") from exc
            content = (
                apply_edits(parent.content, proposal.edits)
                if self.config.mode == "diff"
                else check_rewrite(parent.content, proposal.source)
            )
            record["content"] = content
            check_program(content)
            evaluation = await self._evaluate(content, record)
            if set(evaluation.metrics) != set(self.objectives):
                raise InvalidCandidate("Final evaluator changed the objective names")
            child = Candidate(
                attempt_id,
                content,
                evaluation.metrics,
                evaluation.feedback,
                evaluation.cell,
                parent.id,
            )
            self.programs.append(child)
            self._register(child, self.islands[island_id])
            improvement = child.metrics[objective] / max(
                1.0, abs(parent.metrics[objective])
            ) - parent.metrics[objective] / max(1.0, abs(parent.metrics[objective]))
            idea.reward += max(0.0, improvement)
            record.update(status="evaluated", candidate=child)
        except (InvalidCandidate, ProviderError, TimeoutError, asyncio.TimeoutError) as exc:
            record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
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

    async def _evaluate(self, content: str, record: dict) -> Evaluation:
        feedback = []
        record["stages"] = []
        for stage in self.stages:
            self.evaluations += 1
            result = await asyncio.wait_for(stage.evaluate(content), self.config.evaluation_timeout)
            record["stages"].append(result)
            if result.error:
                raise InvalidCandidate(result.error)
            metrics = {name: float(value) for name, value in result.metrics.items()}
            if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                raise InvalidCandidate("Evaluation metrics must be nonempty and finite")
            if any(
                name not in metrics or metrics[name] < floor
                for name, floor in stage.minimums.items()
            ):
                raise InvalidCandidate(
                    f"Evaluation cascade threshold failed: {metrics}; "
                    f"required {stage.minimums}. {result.feedback}"
                )
            feedback.append(result.feedback)
        return Evaluation(metrics, "\n".join(filter(None, feedback)), result.cell)

    def _register(self, candidate: Candidate, island: dict) -> None:
        """Keep one champion per (diversity cell, objective), retaining ties."""
        for metric, value in candidate.metrics.items():
            key = candidate.cell, metric
            if key not in island or value > island[key].metrics[metric]:
                island[key] = candidate
            if (
                metric not in self.best_by_metric
                or value > self.best_by_metric[metric].metrics[metric]
            ):
                self.best_by_metric[metric] = candidate

    def reset_islands(self) -> None:
        """Reseed the weaker half from surviving champions, as in official FunSearch.

        Local adaptation: use the selected target metric and a completion interval.
        In-flight children are admitted to the current island when they finish.
        """
        ranked = list(range(len(self.islands)))
        self.rng.shuffle(ranked)
        ranked.sort(
            key=lambda i: max(p.metrics[self.target_metric] for p in self.islands[i].values())
        )
        count = len(ranked) // 2
        for island_id in ranked[:count]:
            donor = self.rng.choice(ranked[count:])
            founder = max(self.islands[donor].values(), key=lambda p: p.metrics[self.target_metric])
            self.islands[island_id] = {}
            self._register(founder, self.islands[island_id])
            self.events.append(
                {"completed": self.completed, "reset": island_id, "founder": founder.id}
            )
