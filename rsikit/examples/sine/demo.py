"""Run offline optimization, repairing a supplied syntax error before evaluation."""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from rsikit import Evaluation, HillClimb, LocalEvaluator, RepairingProposer


async def run_experiment(
    directory: Path, propose, repair, *, iterations=3, max_repairs=2, timeout=180
):
    """Shared sine experiment; callers supply proposal and repair operations."""
    folder = Path(__file__).resolve().parent
    initial = folder / "initial.py"
    source = initial.read_text(encoding="utf-8")

    async def check(source):
        try:
            compile(source, "<candidate>", "exec")
        except SyntaxError as exc:
            return Evaluation(valid=False, feedback=f"{exc.msg} at line {exc.lineno}")
        return Evaluation(valid=True, feedback="Syntax check passed; fitness is not measured here.")

    proposer = RepairingProposer(propose, check, repair, max_repairs=max_repairs)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    evaluator = LocalEvaluator(folder / "evaluate.py")

    async def evaluate_source(identifier, source):
        attempt = directory / f"{identifier:04d}"
        attempt.mkdir()
        program = attempt / "candidate.py"
        program.write_text(source, encoding="utf-8")
        return await evaluator(program)

    baseline = await evaluate_source(0, source)
    strategy = HillClimb(source, baseline, proposer, objective="mse", maximize=False)
    print(f"Baseline MSE: {baseline.metrics['mse']:.8g}", flush=True)
    print(f"Output: {directory}", flush=True)
    try:
        for iteration in range(iterations):
            print(f"Generation {iteration + 1}/{iterations}...", flush=True)
            candidates = await asyncio.wait_for(strategy.generate(), timeout)
            evaluations = [await evaluate_source(c.id, c.source) for c in candidates]
            await strategy.update(candidates, evaluations)
            repairs = len(proposer.history[-1].attempts) - 1
            latest = strategy.history[-1].evaluation
            print(
                f"  repairs={repairs}, valid={latest.valid}, "
                f"mse={latest.metrics.get('mse')}, best={strategy.best.evaluation.metrics['mse']:.8g}",
                flush=True,
            )
            if not latest.valid:
                print(f"  {latest.feedback}", flush=True)
    finally:
        # Persistence is this experiment's choice, independent of the strategy.
        (directory / "best.py").write_text(strategy.best.source, encoding="utf-8")
        (directory / "history.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in strategy.history], indent=2),
            encoding="utf-8",
        )
        (directory / "repairs.json").write_text(
            json.dumps(
                [
                    [
                        {"source": a.source, "evaluation": a.evaluation.model_dump(mode="json")}
                        for a in result.attempts
                    ]
                    for result in proposer.history
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    best = strategy.best.evaluation.metrics["mse"]
    print(f"Mean squared error: {baseline.metrics['mse']:.8g} -> {best:.8g}")
    print(f"Best source: {directory / 'best.py'}")
    print(f"History: {directory / 'history.json'}")
    print(f"Repair traces: {directory / 'repairs.json'}")
    return strategy


async def run(directory: Path):
    source = Path(__file__).with_name("initial.py").read_text(encoding="utf-8")
    expressions = iter(
        [
            "x - x**3 /",  # Deliberate syntax error for the repair demonstration.
            "x - x**3 / 6 + x**5 / 120",
            "x - x**3 / 6 + x**5 / 120 - x**7 / 5040",
        ]
    )

    async def propose(parent, history):
        return source.replace("return x", f"return {next(expressions)}")

    async def repair(source, evaluation):
        # A supplied correction, not an LLM call or a general-purpose repair rule.
        return source.replace("x**3 /\n", "x**3 / 6\n")

    return await run_experiment(directory, propose, repair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / f"rsikit-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
    )
    asyncio.run(run(parser.parse_args().output))
