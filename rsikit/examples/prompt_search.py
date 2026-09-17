"""Offline instruction search using the existing literal packing evaluator."""

import argparse
import asyncio
import json
from pathlib import Path

from slick import prompts
from slick.providers import Provider

from rsikit import (
    Draft,
    Evaluation,
    HillClimb,
    PromptProposer,
    PromptSearch,
    PromptTrial,
    ReflectionMemory,
    RepairingProposer,
)
from rsikit.examples.circle_packing.evaluate import evaluate, read_circles

PACKING = Path(__file__).resolve().parent / "circle_packing"
DEVELOPMENT = ("packing-dev/repeat-0",)
SELECTION = ("packing-selection/repeat-0",)
INITIAL_INSTRUCTION = "Improve the sum of radii."


class BudgetExhausted(RuntimeError):
    """No API call was dispatched because its allowance was exhausted."""


class BudgetedProvider:
    """Application-owned sequential call allowance, including failed API calls.

    Disable transport retries in the wrapped provider: this counts acall dispatches,
    not hidden transport retries. Trial scopes are sequential, never nested.
    """

    def __init__(self, provider: Provider, *, max_calls: int):
        self.provider, self.max_calls = provider, max_calls
        self.calls = 0
        self.trial_start: int | None = None
        self.trial_limit = 0

    def ensure_capacity(self, calls: int) -> None:
        if self.calls + calls > self.max_calls:
            raise BudgetExhausted("Root call allowance exhausted")
        if (
            self.trial_start is not None
            and self.calls - self.trial_start + calls > self.trial_limit
        ):
            raise BudgetExhausted("Trial call allowance exhausted")

    def start_trial(self, max_calls: int) -> None:
        self.trial_start, self.trial_limit = self.calls, max_calls

    def end_trial(self) -> None:
        self.trial_start = None

    async def acall(self, context, *, tools=None, tool_results=None):
        self.ensure_capacity(1)
        self.calls += 1
        return await self.provider.acall(context, tools=tools, tool_results=tool_results)


class PackingTrials:
    """Fresh one-attempt searches; utilities include failures as zero.

    Only literal packings are parsed; generated code is never executed. Each case
    is an independent repeat of the same fixed geometry task, not a new benchmark.
    """

    def __init__(self, provider: BudgetedProvider, *, max_trial_calls: int = 3):
        self.provider, self.max_trial_calls = provider, max_trial_calls
        self.initial = (PACKING / "initial.py").read_text()
        self.task = (Path(__file__).parent / "prompts" / "literal_packing.txt").read_text()
        self.records: list[dict] = []

    async def evaluate_instruction(self, instruction: str, cases: tuple[str, ...]) -> PromptTrial:
        utilities, feedback = [], []
        for case in cases:
            outcome = await self.run_case(instruction, case)
            utilities.append(outcome.metrics["sum_radii"] if outcome.valid else 0.0)
            feedback.append(outcome.feedback)
        return PromptTrial(cases, tuple(utilities), tuple(feedback))

    async def run_case(self, instruction: str, case: str) -> Evaluation:
        operations = PromptProposer(self.task, self.provider, instructions={"mutate": instruction})
        memory = ReflectionMemory(self.task, self.provider)
        record = {"case_id": case, "instruction": instruction, "checks": []}
        self.records.append(record)

        async def check(source):
            try:
                result = Evaluation(**evaluate(read_circles(source)))
            except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
                result = Evaluation(valid=False, feedback=f"{type(exc).__name__}: {exc}")
            record["checks"].append(
                {"source": source, "evaluation": result.model_dump(mode="json")}
            )
            return result

        async def propose(parent, history):
            return await operations(
                parent, history, context={**strategy.context, "guidance": memory.texts}
            )

        repair = RepairingProposer(propose, check, operations.repair, max_repairs=1)
        strategy = HillClimb(self.initial, await check(self.initial), repair, objective="sum_radii")
        self.provider.start_trial(self.max_trial_calls)
        start = self.provider.calls
        try:
            candidates = await strategy.generate()
            outcomes = [await check(candidate.source) for candidate in candidates]
            await strategy.update(candidates, outcomes)
            candidate = strategy.history[-1]
            await memory.observe(strategy.history[0], candidate, objective="sum_radii")
            record["outcome"] = candidate.evaluation.model_dump(mode="json")
            return candidate.evaluation
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.update(
                calls=self.provider.calls - start,
                history=[candidate.model_dump(mode="json") for candidate in strategy.history],
                proposals=operations.records,
                reflections=memory.attempts,
            )
            self.provider.end_trial()


async def run(
    provider: Provider,
    output: Path,
    *,
    max_calls: int = 32,
    model: str = "scripted-offline",
    model_settings: dict | None = None,
) -> dict:
    """Save completed and partial evidence even on failure; return best on budget stop."""
    budget = BudgetedProvider(provider, max_calls=max_calls)
    trials = PackingTrials(budget)
    search = PromptSearch(
        trials.task,
        budget,
        trials.evaluate_instruction,
        before_comparison=lambda: budget.ensure_capacity(
            2 * (len(DEVELOPMENT) + len(SELECTION)) * trials.max_trial_calls
        ),
    )
    status, error = "complete", None
    try:
        await search.run(
            INITIAL_INSTRUCTION, development=DEVELOPMENT, selection=SELECTION, revisions=2
        )
    except BudgetExhausted as exc:
        status, error = "budget", str(exc)
    except BaseException as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {
            "status": status,
            "error": error,
            "best": search.best,
            "initial_instruction": INITIAL_INSTRUCTION,
            "initial_source": trials.initial,
            "task": trials.task,
            "model": model,
            "model_settings": model_settings or {},
            "evaluator": "circle_packing.evaluate + read_circles; literal-only; tolerance=1e-9",
            "case_set": "packing-independent-repeats-v1",
            "development": DEVELOPMENT,
            "selection": SELECTION,
            "failure_utility": 0.0,
            "calls": budget.calls,
            "max_calls": max_calls,
            "max_trial_calls": trials.max_trial_calls,
            "tokens": None,
            "cost": None,
            "comparisons": search.history,
            "measurements": search.measurements,
            "trials": trials.records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def demo_responses() -> list[str]:
    """Script retain then promote; this demonstrates plumbing, not model quality."""
    initial = (PACKING / "initial.py").read_text()

    def trial(radius):
        source = initial.replace("0.10", str(radius))
        return [
            Draft(description="Change radii on the fixed grid", source=source).model_dump_json(),
            "The measured packing keeps the centers fixed and changes the radii.",
        ]

    return [
        *trial(0.11),
        "Increase radii aggressively.",
        *trial(0.11),
        *trial(0.12),  # Development: challenger wins.
        *trial(0.11),
        *trial(0.105),  # Selection: challenger loses.
        "Preserve spacing while increasing radii.",
        *trial(0.125),
        *trial(0.11),  # Order alternates on the second revision.
        *trial(0.125),
        *trial(0.11),
    ]


if __name__ == "__main__":
    from rsikit.tests.providers import ScriptedProvider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/prompt-search-offline.json"))
    args = parser.parse_args()
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
    result = asyncio.run(run(ScriptedProvider(demo_responses()), args.output))
    assert [record["decision"] for record in result["comparisons"]] == ["retain", "promote"]
    print(
        f"{result['status']}: {result['calls']} calls; best={result['best']!r}; saved {args.output}"
    )
