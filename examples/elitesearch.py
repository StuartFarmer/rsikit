"""Breed new ideas, elite edits and multi-elite remixes with OpenRouter and Docker."""

import argparse
import asyncio
import json
import logging
import math
import os
from contextlib import AsyncExitStack
from pathlib import Path

from slick import prompts
from slick.providers import OpenRouterAPI

from research import elitesearch
from research.elitesearch import Config, EliteSearch
from research.elitesearch.videos import generation_videos
from research.rewards import episode_error, episode_scores
from research.rewards import measure_rewards as measure
from research.rollouts import Rollouts
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment


async def run_search(agent, run, rollouts, *, seeds, heldout_seeds, video_top=0, video_workers=2):
    logger = logging.getLogger("research.elitesearch")
    previous = agent.on_checkpoint
    reported = 0
    video_queue = asyncio.Queue()
    video_task = (
        asyncio.create_task(generation_videos(video_queue, run.path, video_top, video_workers))
        if video_top
        else None
    )

    def checkpoint(current):
        nonlocal reported
        if video_task is not None and video_task.done():
            video_task.result()
        run.save(*current.records())
        if previous is not None:
            previous(current)
        if current.generations:
            generation = current.generations[-1]
            completed = sum((g.status == "completed" for g in current.generations))
            if completed > reported:
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
                    logger.info("Queued leader videos for generation %s", generation.number)

    agent.on_checkpoint = checkpoint
    summary = {}
    try:
        logger.info("Run: %s", run.path)
        await agent.run()
        if agent.best is not None:
            agent.best.to_file(run.path / "best.py")
            logger.info("Evaluating best elite on held-out seeds")
            result = (await measure(rollouts, [agent.best], heldout_seeds))[agent.best.id]
            summary["heldout"] = dict(scores=episode_scores(result), failure=episode_error(result))
        if video_task is not None:
            video_queue.put_nowait(None)
            logger.info("Waiting for queued generation videos to finish")
            await video_task
            summary["videos"] = str(run.path / "videos/index.html")
    finally:
        if video_task is not None:
            if not video_task.done():
                video_task.cancel()
            await asyncio.gather(video_task, return_exceptions=True)
        summary.update(
            reason=agent.reason,
            generations=sum((g.status == "completed" for g in agent.generations)),
            organisms=len(agent.organisms),
            elite_ids=[row.id for row in agent.elites],
            best_id=None if agent.best is None else agent.best.id,
        )
        (run.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        agent.on_checkpoint = previous
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
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=120.0,
        help="Seconds per model call, including policy generation and repairs",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent episode processes")
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
        help="Bitcoin/Blackjack leaders to video after each generation; 0 disables",
    )
    parser.add_argument("--video-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("--episode-timeout must be positive and finite")
    if not math.isfinite(args.generation_timeout) or args.generation_timeout <= 0:
        parser.error("--generation-timeout must be positive and finite")
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
        args.video_top and (args.env not in ("Blackjack", "Bitcoin") or args.max_steps is not None)
    ):
        parser.error("--video-top requires full-episode Bitcoin/Blackjack and must be nonnegative")
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
        model=args.model, max_output_tokens=args.max_output_tokens, timeout=args.generation_timeout
    )
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(
            make_environment(
                args.env, max_steps=args.max_steps, shoes_per_episode=args.shoes_per_seed
            )
        )
        executor = await stack.enter_async_context(
            Executor(
                concurrency=args.concurrency,
                episode_timeout=args.episode_timeout,
            )
        )
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-elitesearch",
                path=args.output,
            )
        )
        rollouts = Rollouts(environment, executor, run)
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n"
        )

        async def evaluate(policies):
            return await measure(rollouts, policies, args.seeds)

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
                generation_timeout=args.generation_timeout,
                max_repairs=args.max_repairs,
            ),
            seed=args.search_seed,
        )
        await run_search(
            agent,
            run,
            rollouts,
            seeds=args.seeds,
            heldout_seeds=args.heldout_seeds,
            video_top=args.video_top,
            video_workers=args.video_workers,
        )


if __name__ == "__main__":
    asyncio.run(main())
