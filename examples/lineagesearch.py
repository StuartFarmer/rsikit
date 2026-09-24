"""Explore executable approach families with OpenRouter and Docker until stagnation."""

import argparse
import asyncio
import json
import logging
import math
import os
from contextlib import AsyncExitStack
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from slick import prompts
from slick.providers import OpenRouterAPI

import lineagesearch
from examples.alphaevolve import TASKS, _ProgressHandler, _show_scores, make_environment
from lineagesearch import Config, LineageSearch, Measurement
from rsikit import Executor, Run
from rsikit.episode import PolicyError


async def measure(run, policies, seeds):
    """Keep successful measurements when sibling candidates fail in the sandbox."""
    failures = {}
    try:
        await run.evaluate(policies, seeds=seeds)
    except PolicyError as exc:
        if not exc.failures or not set(exc.failures) <= {p.id for p in policies}:
            raise
        failures = exc.failures
    return {
        policy.id: Measurement(
            scores={}
            if policy.id in failures
            else {seed: run.scores(policy)[seed] for seed in seeds},
            failure=failures.get(policy.id),
        )
        for policy in policies
    }


async def run_search(agent, run, *, seeds, heldout_seeds, console=None):
    """Show generation, repair, evaluation and family progress; retain messages in run.log."""
    console = console or Console()
    loggers = (logging.getLogger("lineagesearch"), logging.getLogger("rsikit"))
    logger = loggers[0]
    settings = [(item.level, item.propagate) for item in loggers]
    previous_checkpoint = agent.on_checkpoint
    reported_batches = {}
    reported_plan = False
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        attempts = progress.add_task("Attempt budget", total=agent.config.max_attempts)
        families = progress.add_task("Families complete", total=agent.config.families)
        planning = progress.add_task("Planning families", total=agent.config.families)
        calls = progress.add_task("Model calls completed", total=None)
        display = _ProgressHandler(progress, overlap=True)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for item in loggers:
            item.addHandler(display)
            item.addHandler(log)
            item.setLevel(logging.INFO)
            item.propagate = False

        def checkpoint(current):
            nonlocal reported_batches, reported_plan
            run.save(*current.records())
            if previous_checkpoint is not None:
                previous_checkpoint(current)
            progress.update(attempts, completed=current.study.attempts)
            progress.update(
                calls, completed=sum("raw" in c or "error" in c for c in current.study.calls)
            )
            progress.update(families, completed=sum(f.status != "active" for f in current.families))
            if not reported_plan:
                progress.update(
                    planning,
                    description=current.study.phase.replace("_", " ").capitalize(),
                    completed=sum(
                        bool(f.initial_approaches) or f.status != "active" for f in current.families
                    ),
                )
            if current.study.phase == "search" and not reported_plan:
                reported_plan = True
                plan = dict(
                    task=current.task,
                    levels=["mechanism family", "experimental approach", "policy"],
                    families=[
                        f.model_dump(
                            include={"id", "name", "mechanism", "status", "initial_approaches"}
                        )
                        for f in current.families
                    ],
                    decompositions=current.study.decompositions,
                )
                (run.path / "decomposition.json").write_text(
                    json.dumps(plan, indent=2) + "\n", encoding="utf-8"
                )
                tree = Tree("Initial decomposition")
                for family in current.families:
                    branch = tree.add(Text(family.name + ": " + family.mechanism))
                    for approach in family.initial_approaches:
                        branch.add(Text(approach["hypothesis"] + " — " + approach["mechanism"]))
                    if not family.initial_approaches:
                        branch.add("Decomposition exhausted")
                console.print(tree)
                progress.update(planning, description="Initial decomposition complete")
            for family in current.families:
                previous = reported_batches.get(family.id, 0)
                if family.batches <= previous:
                    continue
                rows = [row for row in current.trials if row.family_id == family.id]
                batches = sorted({row.batch for row in rows})[previous : family.batches]
                for batch in batches:
                    ids = {
                        row.policy_id
                        for row in rows
                        if row.batch == batch and row.policy_id is not None
                    }
                    _show_scores(
                        [p for p in run.policies() if p.id in ids], run, console, seeds=seeds
                    )
                reported_batches[family.id] = family.batches

        agent.on_checkpoint = checkpoint
        try:
            logger.info("Run: %s", run.path)
            study = await agent.run()
            summary = dict(
                reason=study.reason,
                attempts=study.attempts,
                calls=len(study.calls),
                repairs=sum(row.repairs for row in agent.trials),
                families=[f.model_dump() for f in agent.families],
                best_id=None if agent.best is None else agent.best.id,
            )
            # Write search completion before any held-out infrastructure can fail.
            destination = run.path / "summary.json"
            destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            table = Table("Family", "Status", "Best score", "Stale batches")
            for family in agent.families:
                score = (
                    "—"
                    if family.best_id is None
                    else f"{agent.trials[family.best_id - 1].score:.3g}"
                )
                table.add_row(Text(family.name), family.status, score, str(family.stale_batches))
            console.print(table)
            if agent.best is not None:
                logger.info("Evaluating final incumbent on held-out seeds")
                heldout = (await measure(run, [agent.best], heldout_seeds))[agent.best.id]
                summary["heldout"] = dict(scores=heldout.scores, failure=heldout.failure)
                destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                _show_scores([agent.best], run, console, seeds=heldout_seeds)
            logger.info(
                "Stopped: %s; %s attempts, %s model calls, %s policy repairs",
                study.reason,
                study.attempts,
                len(study.calls),
                summary["repairs"],
            )
            return study
        except Exception:
            logger.exception("Search failed; saved evidence is in %s", run.path)
            raise
        finally:
            agent.on_checkpoint = previous_checkpoint
            for item, (level, propagate) in zip(loggers, settings):
                item.removeHandler(display)
                item.removeHandler(log)
                item.setLevel(level)
                item.propagate = propagate
            display.close()
            log.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16384,
        help="Output budget per model call, including large decomposition responses",
    )
    parser.add_argument("--families", type=int, default=10)
    parser.add_argument("--initial", type=int, default=10, help="Founders per family")
    parser.add_argument(
        "--decomposition-k",
        type=int,
        default=3,
        help="MAKER vote lead; samples 2k-1 partitions (1 disables competing partitions)",
    )
    parser.add_argument(
        "--decomposition-max-votes",
        type=int,
        default=40,
        help="Discriminator vote cap per partition election",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--frontier", type=int, help="Optional cap on survivors per family")
    parser.add_argument(
        "--cull-percent",
        type=float,
        default=90,
        help="Drop the lowest-scoring percentage globally after each round (default: 90)",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0)
    parser.add_argument(
        "--uncertainty",
        type=float,
        default=2,
        help="Paired standard-error multiplier; 0 for deterministic scores",
    )
    parser.add_argument("--exploration", type=float, default=0.2)
    parser.add_argument("--bonus-batches", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Repair calls per policy, shared across generation and execution",
    )
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=100,
        help="Global model-call limit for decomposition, voting, implementation and repair",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "families",
        "initial",
        "decomposition_k",
        "decomposition_max_votes",
        "batch_size",
        "patience",
        "max_attempts",
        "generation_concurrency",
        "max_output_tokens",
        "concurrency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bonus_batches < 0:
        parser.error("--bonus-batches must be nonnegative")
    if args.frontier is not None and args.frontier < 1:
        parser.error("--frontier must be positive")
    if not 0 <= args.cull_percent < 100:
        parser.error("--cull-percent must be between 0 (inclusive) and 100 (exclusive)")
    if args.max_repairs < 0:
        parser.error("--max-repairs must be nonnegative")
    if not 0 <= args.exploration <= 1:
        parser.error("--exploration must be between 0 and 1")
    if any(not math.isfinite(v) or v < 0 for v in (args.min_delta, args.uncertainty)):
        parser.error("--min-delta and --uncertainty must be finite and nonnegative")
    if set(args.seeds) & set(args.heldout_seeds):
        parser.error("Search and held-out seeds must be disjoint")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(lineagesearch.__file__).parent / "prompts"
    provider = OpenRouterAPI(
        model=args.model, max_output_tokens=args.max_output_tokens, timeout=120
    )
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(make_environment(args.env, max_steps=args.max_steps))
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-lineagesearch",
                environment=environment,
                executor=Executor(concurrency=args.concurrency),
                path=args.output,
            )
        )
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
        )

        async def evaluate(policies):
            return await measure(run, policies, args.seeds)

        agent = LineageSearch(
            "Maximize cumulative episode reward in the described environment.",
            provider,
            evaluate,
            context=environment.instructions,
            config=Config(
                families=args.families,
                initial_per_family=args.initial,
                decomposition_k=args.decomposition_k,
                decomposition_max_votes=args.decomposition_max_votes,
                batch_size=args.batch_size,
                frontier_per_family=args.frontier,
                cull_percent=args.cull_percent,
                patience=args.patience,
                min_delta=args.min_delta,
                uncertainty=args.uncertainty,
                exploration=args.exploration,
                bonus_batches=args.bonus_batches,
                max_attempts=args.max_attempts,
                max_repairs=args.max_repairs,
                generation_concurrency=args.generation_concurrency,
            ),
            seed=args.search_seed,
        )
        await run_search(agent, run, seeds=args.seeds, heldout_seeds=args.heldout_seeds)


if __name__ == "__main__":
    asyncio.run(main())
