"""Run a no-API demo, benchmark, or an LLM-driven EliteTable poker search."""

import argparse
import asyncio
import csv
import fcntl
import json
import logging
import math
import os
import secrets
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from rich.console import Console
from slick import prompts

from examples.elitelist_papers.run import LoggedOpenRouter, command_output
from research.elitesearch import Config

from .baselines import policies as references
from .display import SearchDisplay, make_progress, update_tables
from .game import derived_seed
from .pool import TablePool
from .search import PokerSearch, write_json
from .tournament import CONTRACT, Tournament, TournamentConfig

logger = logging.getLogger(__name__)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("demo", "benchmark", "search"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--population", type=int, default=48)
    result.add_argument("--elites", type=int, default=12)
    result.add_argument("--generations", type=int, default=5)
    result.add_argument("--rounds", type=int, default=4)
    result.add_argument(
        "--deals", type=int, default=100, help="Deals per table before six seat rotations"
    )
    result.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent tables inside one evaluator container (also its CPU limit)",
    )
    result.add_argument(
        "--memory-gb",
        type=int,
        help="Shared evaluator container RAM limit in GiB (default: 32); may override on resume",
    )
    result.add_argument("--stack", type=int, default=200, help="Chips per player; BB is 2")
    result.add_argument("--seed", type=int, help="Reproducibility seed; randomly chosen if omitted")
    result.add_argument("--call-timeout", type=float, default=2)
    result.add_argument("--block-timeout", type=float, default=180)
    result.add_argument(
        "--policy-ms",
        type=float,
        default=5,
        help="Mean policy call budget in milliseconds, excluding warm-up",
    )
    result.add_argument(
        "--warmup-timeout",
        type=float,
        default=15,
        help="First reset/act deadline per player rotation, including JIT compilation",
    )
    result.add_argument("--image", default="elitetable-poker:local")
    result.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    result.add_argument("--model-workers", type=int, default=4)
    result.add_argument("--max-repairs", type=int, default=2)
    result.add_argument("--heldout-rounds", type=int, default=4)
    return result


async def heldout(agent, args, config, *, console=None, display=None, pool=None):
    """Fixed references and fresh deals; called only after breeding finishes."""
    path = args.output / "heldout.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    opponents = references(5)
    for index, generation in enumerate(agent.history):
        winner = agent.organisms[generation["elite_ids"][0] - 1]
        if display:
            display.heldout(index, len(agent.history), "")
        if results.get(winner.policy_id, {}).get("scores"):
            logger.info("Reusing held-out results: %s", winner.name)
            continue
        tournament = Tournament(
            replace(config, rounds=args.heldout_rounds),
            seed=derived_seed(args.seed, "heldout"),
            runner=pool,
        )
        policies = [agent._policies[winner.id], *opponents]
        if display:
            display.heldout(index, len(agent.history), winner.name)
            tournament.progress = display.tables
            measured = await tournament(policies)
        else:
            measured = await evaluate(
                tournament, policies, description=f"Held-out tables: {winner.name}", console=console
            )
        result = measured[winner.policy_id]
        failure = result.failure
        if tournament.report["failures"] and not failure:
            failure = f"Reference opponents failed: {tournament.report['failures']}"
        results[winner.policy_id] = dict(
            scores=result.scores, failure=failure, report=tournament.report
        )
        write_json(path, results)
        if failure:
            logger.warning("Held-out %s failed: %s", winner.name, failure)
        else:
            logger.info(
                "Held-out %s: %.3f BB/100",
                winner.name,
                sum(result.scores.values()) / len(result.scores),
            )
    if display:
        display.heldout(len(agent.history), len(agent.history), "")
    with (args.output / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "generation",
                "policy_id",
                "search_bb_per_100",
                "heldout_bb_per_100",
                "heldout_failure",
            ),
        )
        writer.writeheader()
        for generation in agent.history:
            winner = agent.organisms[generation["elite_ids"][0] - 1]
            measured = results[winner.policy_id]
            scores = list(measured["scores"].values())
            writer.writerow(
                dict(
                    generation=generation["generation"],
                    policy_id=winner.policy_id,
                    search_bb_per_100=generation["scores"][str(winner.id)],
                    heldout_bb_per_100=sum(scores) / len(scores) if scores else None,
                    heldout_failure=measured["failure"],
                )
            )


async def evaluate(tournament, policies, *, description="Tables", console=None):
    with make_progress(console) as display:
        task = display.add_task(description, total=None)
        tournament.progress = lambda done, total: update_tables(display, task, done, total)
        return await tournament(policies)


async def run_search(args, config, *, console=None, pool=None):
    if pool is None:
        async with TablePool(config) as pool:
            return await run_search(args, config, console=console, pool=pool)
    console = console or Console()
    tournament = Tournament(config, seed=args.seed, runner=pool)
    provider = LoggedOpenRouter(
        output=args.output / "llm_calls.jsonl",
        model=args.model,
        max_output_tokens=12000,
        timeout=120,
    )
    search_config = Config(
        population_size=args.population,
        elite_size=args.elites,
        generations=args.generations,
        generation_concurrency=args.model_workers,
        max_repairs=args.max_repairs,
    )
    agent = PokerSearch(
        "Evolve profitable Texas Hold'em policies. Maximize current-field BB/100.",
        provider,
        tournament,
        config=search_config,
        seed=args.seed,
        context=CONTRACT
        + f"\nStarting stack: {args.stack} chips. Policy call timeout: {args.call_timeout}s."
        + f" Mean policy call budget: {args.policy_ms} ms; first reset/act warm-up: {args.warmup_timeout}s.",
    )
    # Only libraries actually available in this dedicated worker are advertised.
    agent.libraries = (
        "Available: Python standard library, NumPy and Numba 0.64.0 (serial njit kernels)."
    )
    checkpoint = args.output / "checkpoint.json"
    if args.resume:
        agent.load(checkpoint)
        agent.context = CONTRACT + (
            f"\nStarting stack: {args.stack} chips. Policy call timeout: {args.call_timeout}s."
            f" Mean policy call budget: {args.policy_ms} ms; first reset/act warm-up: {args.warmup_timeout}s."
        )

    def save(current):
        current.save(checkpoint)
        display.checkpoint()
        if current.generations:
            generation = current.generations[-1]
            tournament.seed = derived_seed(args.seed, "search", generation.number)
        for generation in current.history:
            path = args.output / f"generation-{generation['generation']:03}.json"
            if not path.exists():
                write_json(path, generation)
                logger.info(
                    "Saved generation %s; current leaders: %s",
                    generation["generation"],
                    ", ".join(current.organisms[i - 1].name for i in generation["elite_ids"]),
                )

    agent.on_checkpoint = save
    previous = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    try:
        with SearchDisplay(agent, console).run(args.output / "run.log") as display:
            tournament.progress = display.tables
            save(agent)
            logger.info(
                "%s poker search: %s", "Resuming" if args.resume else "Starting", args.output
            )
            logger.info(
                "Concurrency: %s model calls, %s table workers", args.model_workers, config.workers
            )
            logger.info(
                "Policy budget: %.2f ms/call after warm-up; %.1fs first reset/act allowance; "
                "Numba enabled; worker image %s",
                config.policy_ms,
                config.warmup_timeout,
                command_output("docker", "image", "inspect", config.image, "--format", "{{.Id}}"),
            )
            try:
                await agent.run()
                agent.best.to_file(args.output / "best.py")
                write_json(
                    args.output / "leaderboard.json",
                    agent.history[-1]["attempts"][-1].get("leaderboard", []),
                )
                await heldout(agent, args, config, console=console, display=display, pool=pool)
                logger.info("Search and held-out evaluation completed")
            except asyncio.CancelledError:
                logger.warning("Search interrupted; checkpoint saved for --resume")
                raise
            except Exception:
                logger.exception("Poker search failed")
                raise
    finally:
        prompts.TEMPLATE_ROOT = previous


async def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    memory_override = args.memory_gb
    if memory_override is not None and memory_override < 1:
        cli.error("--memory-gb must be positive")
    for key in (
        "population",
        "elites",
        "generations",
        "rounds",
        "deals",
        "workers",
        "model_workers",
        "heldout_rounds",
    ):
        if getattr(args, key) < 1:
            cli.error(f"--{key.replace('_', '-')} must be positive")
    if args.population < 2 or args.stack < 2 or args.max_repairs < 0:
        cli.error("population and stack must be >=2; max-repairs must be nonnegative")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.call_timeout, args.block_timeout, args.policy_ms, args.warmup_timeout)
    ):
        cli.error("Timeouts must be positive and finite")
    if args.mode == "search" and not os.environ.get("OPENROUTER_API_KEY"):
        cli.error("Set OPENROUTER_API_KEY for search; demo and benchmark make no model calls")
    if args.resume:
        if args.mode != "search":
            cli.error("Only search runs can resume")
        saved = json.loads((args.output / "experiment.json").read_text())
        budget = args.generations
        for key, value in saved["arguments"].items():
            if key not in ("output", "resume", "generations"):
                setattr(args, key, value)
        args.generations = max(budget, saved["arguments"]["generations"])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        args.seed = args.seed if args.seed is not None else secrets.randbits(128)
    if memory_override is not None:
        args.memory_gb = memory_override
    if args.memory_gb is None:
        args.memory_gb = 32
    if args.memory_gb < 1:
        cli.error("--memory-gb must be positive")
    config = TournamentConfig(
        rounds=args.rounds,
        deals=args.deals,
        stack=args.stack,
        workers=args.workers,
        memory_gb=args.memory_gb,
        call_timeout=args.call_timeout,
        block_timeout=args.block_timeout,
        policy_ms=args.policy_ms,
        warmup_timeout=args.warmup_timeout,
        image=args.image,
    )
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:
            write_json(
                args.output / "experiment.json",
                dict(
                    arguments={**vars(args), "output": str(args.output)},
                    tournament=asdict(config),
                    image_id=command_output(
                        "docker", "image", "inspect", args.image, "--format", "{{.Id}}"
                    ),
                    git_revision=command_output("git", "rev-parse", "HEAD"),
                ),
            )
            (args.output / "context.txt").write_text(CONTRACT)
        started = perf_counter()
        write_json(args.output / "status.json", {"status": "running"})
        try:
            async with TablePool(config) as pool:
                if args.mode == "search":
                    await run_search(args, config, pool=pool)
                elif args.mode == "demo":
                    tournament = Tournament(config, seed=args.seed, runner=pool)
                    await evaluate(tournament, references(12))
                    write_json(args.output / "results.json", tournament.report)
                    if tournament.report["failures"]:
                        raise RuntimeError(
                            f"Reference policy failures: {tournament.report['failures']}"
                        )
                    print(f"{tournament.report['hands_per_second']:.1f} table hands/s", flush=True)
                else:
                    reports = []
                    # Equal total work in each measurement; only parallelism changes.
                    for workers in sorted({1, args.workers}):
                        tournament = Tournament(
                            replace(config, workers=workers), seed=args.seed, runner=pool
                        )
                        await evaluate(
                            tournament, references(12), description=f"Tables ({workers} workers)"
                        )
                        if tournament.report["failures"]:
                            raise RuntimeError(str(tournament.report["failures"]))
                        reports.append(tournament.report)
                        print(
                            f"workers={workers}: {tournament.report['hands_per_second']:.1f} table hands/s",
                            flush=True,
                        )
                    write_json(args.output / "benchmark.json", reports)
        except BaseException as exc:
            write_json(
                args.output / "status.json",
                dict(status="failed", error=str(exc), elapsed_seconds=perf_counter() - started),
            )
            raise
        write_json(
            args.output / "status.json",
            dict(status="completed", elapsed_seconds=perf_counter() - started),
        )


if __name__ == "__main__":
    asyncio.run(main())
