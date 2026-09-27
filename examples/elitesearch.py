"""Breed new ideas, elite edits and multi-elite remixes with OpenRouter and Docker."""

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
from contextlib import AsyncExitStack, suppress
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
from slick import prompts
from slick.providers import OpenRouterAPI

from research import elitesearch
from research.elitesearch import Config, EliteSearch, Measurement
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment
from rsikit.episode import PolicyError
from rsikit.progress import ProgressHandler
from rsikit.sandbox.docker import DockerSandbox


async def measure(run, policies, seeds):
    failures = {}
    try:
        await run.evaluate(policies, seeds=seeds)
    except PolicyError as exc:
        if not exc.failures or not set(exc.failures) <= {p.id for p in policies}:
            raise
        failures = exc.failures
    return {
        p.id: Measurement(
            {} if p.id in failures else {seed: run.scores(p)[seed] for seed in seeds},
            failures.get(p.id),
        )
        for p in policies
    }


async def generation_videos(queue, path, top, workers):
    output = path / "videos"
    output.mkdir(exist_ok=True)
    while (generation := await queue.get()) is not None:
        log_path = output / f"generation-{generation:02}.log"
        with log_path.open("w") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "examples.blackjack_videos",
                str(path),
                "--output",
                str(output),
                "--top",
                str(top),
                "--workers",
                str(workers),
                "--through-generation",
                str(generation),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                code = await process.wait()
            except asyncio.CancelledError:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                await process.wait()
                raise
        if code:
            raise RuntimeError(f"Generation {generation} video export failed; see {log_path}")
        logging.getLogger("research.elitesearch").info(
            "Generation %s videos: %s", generation, output / "index.html"
        )


async def run_search(
    agent, run, *, seeds, heldout_seeds, console=None, video_top=0, video_workers=2
):
    console = console or Console()
    loggers = [logging.getLogger("research.elitesearch"), logging.getLogger("rsikit")]
    settings = [(item.level, item.propagate) for item in loggers]
    previous = agent.on_checkpoint
    reported = 0
    video_queue = asyncio.Queue()
    video_task = (
        asyncio.create_task(generation_videos(video_queue, run.path, video_top, video_workers))
        if video_top
        else None
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        generations = progress.add_task("Generations", total=agent.config.generations)
        population = progress.add_task(
            "Population evaluated/discarded", total=agent.config.population_size
        )
        elites = progress.add_task("Elite slots filled", total=agent.config.elite_size)
        display = ProgressHandler(progress, overlap=True)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for item in loggers:
            item.addHandler(display)
            item.addHandler(log)
            item.setLevel(logging.INFO)
            item.propagate = False

        def checkpoint(current):
            nonlocal reported
            if video_task is not None and video_task.done():
                video_task.result()
            run.save(*current.records())
            if previous is not None:
                previous(current)
            if current.generations:
                generation = current.generations[-1]
                progress.update(
                    population,
                    description=f"Generation {generation.number} population evaluated/discarded",
                    completed=sum(
                        row.generation == generation.number
                        and row.status in ("evaluated", "discarded")
                        for row in current.organisms
                    ),
                )
                completed = sum(g.status == "completed" for g in current.generations)
                progress.update(generations, completed=completed)
                progress.update(elites, completed=len(current.elites))
                if completed > reported:
                    table = Table(
                        "Rank",
                        "Elite",
                        "Score",
                        "Origin",
                        "Parents",
                        title=f"Elite leaderboard — generation {generation.number}",
                    )
                    for rank, row in enumerate(current.elites, 1):
                        table.add_row(
                            str(rank),
                            Text(row.name),
                            f"{row.score:.6g}",
                            row.kind,
                            ", ".join(map(str, row.parent_ids)) or "—",
                        )
                    console.print(table)
                    reported = completed
                    (run.path / "leaderboard.json").write_text(
                        json.dumps(
                            [
                                row.model_dump(exclude={"implementation", "calls", "revisions"})
                                for row in current.elites
                            ],
                            indent=2,
                        )
                        + "\n"
                    )
                    if video_task is not None:
                        video_queue.put_nowait(generation.number)
                        loggers[0].info("Queued leader videos for generation %s", generation.number)

        agent.on_checkpoint = checkpoint
        summary = {}
        try:
            loggers[0].info("Run: %s", run.path)
            await agent.run()
            if agent.best is not None:
                (run.path / "best.py").write_text(agent.best._implementation, encoding="utf-8")
                loggers[0].info("Evaluating best elite on held-out seeds")
                result = (await measure(run, [agent.best], heldout_seeds))[agent.best.id]
                summary["heldout"] = dict(scores=result.scores, failure=result.failure)
            if video_task is not None:
                video_queue.put_nowait(None)
                loggers[0].info("Waiting for queued generation videos to finish")
                await video_task
                summary["videos"] = str(run.path / "videos/index.html")
        finally:
            if video_task is not None:
                if not video_task.done():
                    video_task.cancel()
                await asyncio.gather(video_task, return_exceptions=True)
            summary.update(
                reason=agent.reason,
                generations=sum(g.status == "completed" for g in agent.generations),
                organisms=len(agent.organisms),
                elite_ids=[row.id for row in agent.elites],
                best_id=None if agent.best is None else agent.best.id,
            )
            (run.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            agent.on_checkpoint = previous
            for item, (level, propagate) in zip(loggers, settings):
                item.removeHandler(display)
                item.removeHandler(log)
                item.setLevel(level)
                item.propagate = propagate
            display.close()
            log.close()
        return summary


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TASKS, default="BipedalWalker-v3")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--new-fraction", type=float, default=0.2)
    parser.add_argument(
        "--remix-fraction",
        type=float,
        default=0.4,
        help="Remaining population slots edit one elite",
    )
    parser.add_argument("--remix-parents", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--generation-concurrency", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8, help="Docker episode workers")
    parser.add_argument(
        "--episode-timeout", type=float, default=60.0, help="Wall-clock seconds per seed"
    )
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--shoes-per-seed", type=int, default=24, help="Blackjack shoes per episode"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--video-top",
        type=int,
        default=0,
        help="Blackjack leaders to video after each generation; 0 disables",
    )
    parser.add_argument("--video-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("--episode-timeout must be positive and finite")
    for name in (
        "elites",
        "population",
        "generations",
        "generation_concurrency",
        "concurrency",
        "max_output_tokens",
        "shoes_per_seed",
        "video_workers",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.video_top < 0 or (
        args.video_top and (args.env != "Blackjack" or args.max_steps is not None)
    ):
        parser.error("--video-top requires full-episode Blackjack and must be nonnegative")
    if args.remix_parents < 2 or args.max_repairs < 0:
        parser.error("--remix-parents must be at least 2 and --max-repairs nonnegative")
    if not (
        0 <= args.new_fraction <= 1
        and 0 <= args.remix_fraction <= 1
        and args.new_fraction + args.remix_fraction <= 1
    ):
        parser.error("New/remix fractions must be between 0 and 1 and sum to at most 1")
    if set(args.seeds) & set(args.heldout_seeds):
        parser.error("Search and held-out seeds must be disjoint")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(elitesearch.__file__).parent / "prompts"
    provider = OpenRouterAPI(
        model=args.model, max_output_tokens=args.max_output_tokens, timeout=120
    )
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(
            make_environment(
                args.env, max_steps=args.max_steps, shoes_per_episode=args.shoes_per_seed
            )
        )
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-elitesearch",
                environment=environment,
                executor=Executor(
                    concurrency=args.concurrency,
                    sandbox=DockerSandbox(episode_timeout=args.episode_timeout),
                ),
                path=args.output,
            )
        )
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n"
        )

        async def evaluate(policies):
            return await measure(run, policies, args.seeds)

        agent = EliteSearch(
            "Maximize cumulative episode reward in the described environment.",
            provider,
            evaluate,
            context=environment.instructions,
            config=Config(
                elite_size=args.elites,
                population_size=args.population,
                generations=args.generations,
                new_fraction=args.new_fraction,
                remix_fraction=args.remix_fraction,
                remix_parents=args.remix_parents,
                generation_concurrency=args.generation_concurrency,
                max_repairs=args.max_repairs,
            ),
            seed=args.search_seed,
        )
        await run_search(
            agent,
            run,
            seeds=args.seeds,
            heldout_seeds=args.heldout_seeds,
            video_top=args.video_top,
            video_workers=args.video_workers,
        )


if __name__ == "__main__":
    asyncio.run(main())
