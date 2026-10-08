"""Recursively improve optimizer source against a local downstream meta-utility."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from types import SimpleNamespace

from pydantic import BaseModel, Field, ValidationError
from slick import prompt
from slick.providers import Provider

from rsikit import PolicyDefinition
from rsikit.optimization import SequentialOptimizer

SEED_IMPROVER = """async def improve(initial, capabilities):
    candidates = []
    count = min(capabilities.generations_left, capabilities.evaluations_left)
    for _ in range(count):
        candidate = await capabilities.suggest(initial)
        candidates.append((await capabilities.evaluate(candidate), candidate))
    return max(candidates, key=lambda pair: pair[0])[1]
"""


class Program(BaseModel, extra="forbid"):
    code: str = Field(min_length=1)


class ImproverExecutionError(Exception):
    """The isolated executor reports generated-code failure, including timeout."""


class BudgetExhausted(ImproverExecutionError):
    pass


@dataclass(frozen=True)
class Problem:
    initial: str
    utility_description: str
    evaluate: Callable[[str], Awaitable[float]]


class Capabilities:
    """Expose budgeted generation/evaluation endpoints to an isolated improver."""

    def __init__(self, suggest, evaluate, generations, evaluations):
        self._suggest, self._evaluate = suggest, evaluate
        self.generations_left, self.evaluations_left = generations, evaluations

    async def suggest(self, initial: str, guidance: str = "") -> str:
        if self.generations_left <= 0:
            raise BudgetExhausted("generation budget exhausted")
        self.generations_left -= 1
        return await self._suggest(initial, guidance)

    async def evaluate(self, source: str) -> float:
        if self.evaluations_left <= 0:
            raise BudgetExhausted("utility budget exhausted")
        self.evaluations_left -= 1
        score = await self._evaluate(source)
        if not math.isfinite(score):
            raise ValueError("utility must be finite")
        return score


class STOP(SequentialOptimizer):
    """The executor runs source with RPC access to capabilities in caller-owned isolation.

    It must not execute source in the host process. Only explicit generated-program
    failures are converted to zero utility; transport and evaluator failures propagate.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        execute: Callable[[str, str, Capabilities], Awaitable[str]],
        problems: Sequence[Problem],
        *,
        on_event: Callable[[str, dict], None] | None = None,
        initial=SEED_IMPROVER,
        rounds=3,
        generations=4,
        evaluations=4,
        inner_generations=4,
        inner_evaluations=4,
        target_problem=0,
        deployments=1,
    ):
        super().__init__()
        if type(deployments) is not int or deployments < 0:
            raise ValueError("deployments must be a nonnegative integer")
        self.target_problem, self.deployments = target_problem, deployments
        self._search_options = dict(
            initial=initial,
            rounds=rounds,
            generations=generations,
            evaluations=evaluations,
            inner_generations=inner_generations,
            inner_evaluations=inner_evaluations,
        )
        self._improver = None
        self._best_policy, self._best_score = None, -math.inf
        self.task, self.provider, self.execute = task, provider, execute
        self.problems = tuple(problems)
        self.on_event = on_event

    @property
    def best(self):
        return self._best_policy

    async def _proposals(self):
        if self._improver is None:
            await self.run(**self._search_options)
        problem = self.problems[self.target_problem]
        for _ in range(self.deployments):
            initial = self.best.to_text() if self.best else problem.initial
            caps = self.capabilities(
                problem.evaluate,
                problem.utility_description,
                self.inner_generations,
                self.inner_evaluations,
            )
            try:
                source = await self.execute(self._improver, initial, caps)
            except ImproverExecutionError as exc:
                self.emit(
                    "deployment_failed",
                    dict(
                        problem=self.target_problem,
                        optimizer=self._improver,
                        source=initial,
                        error=str(exc),
                    ),
                )
                return
            try:
                policy = PolicyDefinition.from_text(source)
                policy.validate()
            except ValueError as exc:
                self.emit(
                    "deployment_failed",
                    dict(
                        problem=self.target_problem,
                        optimizer=self._improver,
                        source=source,
                        error=str(exc),
                    ),
                )
                return
            self.emit("proposal", dict(problem=self.target_problem, policy=policy.to_text()))
            yield policy

    def _accept(self, returns, *, seed_scores=None, error=None, feedback=""):
        policy = self._pending_policy
        score = fmean(returns) if error is None else None
        if score is not None and score > self._best_score:
            self._best_policy, self._best_score = policy, score
        self.emit(
            "deployment_feedback",
            dict(
                problem=self.target_problem,
                policy=policy.to_text(),
                score=score,
                scores=dict(seed_scores or {}),
                error=error,
            ),
        )

    def emit(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    @prompt(template="improve.j2", output_type=Program)
    async def improve(
        self, initial: str, utility: str, guidance: str, *, generated: Program
    ) -> str:
        if not generated.code.strip():
            raise ImproverExecutionError("empty generated program")
        return generated.code

    def capabilities(self, utility, description, generations, evaluations):
        async def suggest(initial, guidance):
            self.generation_calls += 1
            self.emit("call_started", dict(call=self.generation_calls, utility=description))

            async def recorded(context, **kwargs):
                response, calls = await self.provider.acall(context, **kwargs)
                self.emit("raw_response", dict(prompt=context, response=response))
                return response, calls

            try:
                return await self.improve(
                    initial, description, guidance, provider=SimpleNamespace(acall=recorded)
                )
            except ValidationError as exc:
                raise ImproverExecutionError(f"Invalid generated program: {exc}") from exc

        async def measure(source):
            self.utility_calls += 1
            result = await utility(source)
            self.emit("utility", dict(source=source, score=result))
            return result

        return Capabilities(suggest, measure, generations, evaluations)

    async def meta_utility(self, source):
        self.meta_calls += 1
        scores = []
        for index, problem in enumerate(self.problems):
            caps = self.capabilities(
                problem.evaluate,
                problem.utility_description,
                self.inner_generations,
                self.inner_evaluations,
            )
            try:
                improved = await self.execute(source, problem.initial, caps)
            except ImproverExecutionError as exc:
                self.emit(
                    "rejected",
                    dict(problem=index, optimizer=source, source=problem.initial, error=str(exc)),
                )
                return 0.0
            if not improved.strip():
                self.emit(
                    "rejected",
                    dict(
                        problem=index,
                        optimizer=source,
                        source=problem.initial,
                        error="Empty returned program",
                    ),
                )
                return 0.0
            # Final checking is separate from the improver's search budget, as upstream.
            self.downstream_checks += 1
            self.utility_calls += 1
            score = await problem.evaluate(improved)
            if not math.isfinite(score):
                raise ValueError("downstream utility must be finite")
            scores.append(score)
            self.emit(
                "downstream", dict(problem=index, improver=source, policy=improved, score=score)
            )
        return fmean(scores)

    async def self_improve(self, optimizer, current, description, generations, evaluations):
        caps = self.capabilities(self.meta_utility, description, generations, evaluations)
        try:
            candidate = await self.execute(optimizer, current, caps)
        except ImproverExecutionError as exc:
            self.emit("rejected", dict(optimizer=optimizer, source=current, error=str(exc)))
            return current, None
        if not candidate.strip():
            self.emit(
                "rejected",
                dict(optimizer=optimizer, source=current, error="Empty returned program"),
            )
            return current, None
        self.improver_checks += 1
        score = await self.meta_utility(candidate)
        return (candidate, score) if score != 0 else (current, None)

    async def run(
        self,
        initial: str = SEED_IMPROVER,
        *,
        rounds: int = 3,
        generations: int = 4,
        evaluations: int = 4,
        inner_generations: int = 4,
        inner_evaluations: int = 4,
    ) -> dict:
        self.inner_generations, self.inner_evaluations = inner_generations, inner_evaluations
        self.generation_calls = self.utility_calls = self.meta_calls = 0
        self.downstream_checks = self.improver_checks = 0
        current = optimizer = previous_optimizer = initial
        history = []
        description = (
            "Mean downstream utility after this improver improves the supplied initial "
            "solutions using budgeted capabilities. Implement async improve(initial, capabilities)."
        )
        for _ in range(rounds):
            candidate, score = await self.self_improve(
                optimizer, current, description, generations, evaluations
            )
            if score is None:
                optimizer = previous_optimizer
            else:
                previous_optimizer, optimizer, current = optimizer, candidate, candidate
            history.append({"source": current, "optimizer": optimizer, "checked_score": score})
            self.emit(
                "round",
                dict(
                    history[-1],
                    round=len(history),
                    generation_calls=self.generation_calls,
                    utility_calls=self.utility_calls,
                    meta_calls=self.meta_calls,
                    downstream_checks=self.downstream_checks,
                    improver_checks=self.improver_checks,
                ),
            )
        self._improver = current
        return {
            "improver": current,
            "history": history,
            "generation_calls": self.generation_calls,
            "utility_calls": self.utility_calls,
            "meta_calls": self.meta_calls,
            "downstream_checks": self.downstream_checks,
            "improver_checks": self.improver_checks,
        }
