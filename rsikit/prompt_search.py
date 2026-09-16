"""Improve plain-text instructions by comparing their measured downstream results."""

import math
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from slick import prompt
from slick.providers import Provider

from .evaluation import EvaluationError
from .proposer import _RecordedProvider


@dataclass(frozen=True)
class PromptTrial:
    """Ordered case/repeat utilities, including the task's declared failure utility."""

    case_ids: tuple[str, ...]
    utilities: tuple[float, ...]
    feedback: tuple[str, ...]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.utilities)


EvaluateInstruction = Callable[[str, tuple[str, ...]], Awaitable[PromptTrial]]


class PromptSearch:
    """Measured instruction hill climbing; no model training or evaluator ownership.

    Evaluators use fresh state and a fixed task-defined utility (higher is better).
    The caller supplies disjoint development/selection cohorts and withholds final
    test data. Each run resets records. One run at a time; configure the checkout
    template root once. Errors propagate with incumbent and partial evidence kept.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: EvaluateInstruction,
        *,
        before_comparison: Callable[[], None] | None = None,
    ):
        self.task, self.provider, self.evaluate = task, provider, evaluate
        self.before_comparison = before_comparison
        self.best = ""
        self.history: list[dict] = []
        self.measurements: list[dict] = []

    async def run(
        self,
        initial_instruction: str,
        *,
        development: tuple[str, ...],
        selection: tuple[str, ...],
        revisions: int = 3,
    ) -> str:
        """Revise using development evidence, then require two complete strict wins."""
        self.best, self.history, self.measurements = initial_instruction, [], []
        if set(development) & set(selection):
            raise EvaluationError("Development and selection evidence must be disjoint")
        await self.assess(self.best, development, "development")
        for iteration in range(revisions):
            record = {"revision": iteration, "incumbent": self.best, "decision": "pending"}
            self.history.append(record)
            try:
                evidence = [
                    m for m in self.measurements if m["split"] == "development" and "trial" in m
                ]
                raw = await self.revise(
                    self.best, evidence, provider=_RecordedProvider(self.provider, record)
                )
                challenger = raw.strip()
                record["challenger"] = challenger
                if not challenger or challenger == self.best.strip():
                    record["decision"] = "unchanged" if challenger else "blank"
                    continue
                if self.before_comparison is not None:
                    self.before_comparison()
                development_win = await self.compare(
                    record, development, "development", challenger_first=bool(iteration % 2)
                )
                selection_win = development_win and await self.compare(
                    record, selection, "selection", challenger_first=bool(iteration % 2)
                )
                record["decision"] = "promote" if selection_win else "retain"
                if selection_win:
                    self.best = challenger
            except BaseException as exc:
                record.update(decision="interrupted", error=f"{type(exc).__name__}: {exc}")
                raise
        return self.best

    async def assess(self, instruction: str, cases: tuple[str, ...], split: str) -> PromptTrial:
        """Record partial attempts; reject invalid callback evidence without scoring it."""
        record = {"instruction": instruction, "case_ids": cases, "split": split}
        self.measurements.append(record)
        try:
            trial = await self.evaluate(instruction, cases)
            if (
                trial.case_ids != cases
                or not cases
                or len(trial.utilities) != len(cases)
                or len(trial.feedback) != len(cases)
                or not all(math.isfinite(value) for value in trial.utilities)
            ):
                raise EvaluationError(
                    "Prompt trial must contain matching cases and finite utilities"
                )
            record["trial"] = asdict(trial)
            return trial
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    async def compare(
        self, record: dict, cases: tuple[str, ...], split: str, *, challenger_first: bool
    ) -> bool:
        """Re-measure both arms, preserving partial evidence if either call fails."""
        pair = record[split] = {}
        arms = ("challenger", "incumbent") if challenger_first else ("incumbent", "challenger")
        trials = {}
        for arm in arms:
            trials[arm] = await self.assess(record[arm], cases, split)
            pair[arm] = asdict(trials[arm])
        return trials["challenger"].mean > trials["incumbent"].mean

    @prompt(template="rsikit/prompts/prompt_search/revise.j2")
    async def revise(self, instruction: str, evidence: list[dict], *, generated: str) -> str:
        return generated
