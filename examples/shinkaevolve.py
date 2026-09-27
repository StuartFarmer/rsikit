"""Evolve Gymnasium policies with ShinkaEvolve using OpenRouter and Docker."""

import argparse
import asyncio
import json
import logging
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
from slick import prompts
from slick.providers import OpenRouterAPI

from research import shinkaevolve
from research.shinkaevolve import Config, ShinkaEvolve
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment
from rsikit.episode import PolicyError
from rsikit.progress import ProgressHandler, show_scores


async def run_search(
    generator, run, *, generations, batch_size, generation_concurrency=4, seeds=(0,), console=None
):
    seeds = tuple(seeds)
    console = console or Console()
    logger = logging.getLogger("research.shinkaevolve")
    loggers = (logger, logging.getLogger("rsikit"))
    old_settings = [(item.level, item.propagate) for item in loggers]
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Generations", total=generations)
        display = ProgressHandler(progress)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for item in loggers:
            item.addHandler(display)
            item.addHandler(log)
            item.setLevel(logging.INFO)
            item.propagate = False
        try:
            logger.info("Run: %s", run.path)
            logger.info("Optimizer: %s", type(generator).__module__)
            for generation in range(generations):
                logger.info("Generation %s/%s", generation + 1, generations)
                complete = False
                try:
                    policies = await generator.generate(
                        n=batch_size, concurrency=generation_concurrency
                    )
                    scores = {}
                    while policies:
                        run.save(*generator.records(seeds=seeds))
                        try:
                            scores = await run.evaluate(policies, seeds=seeds)
                            break
                        except PolicyError as exc:
                            if not exc.failures:
                                raise
                            generator.evaluation_failed(exc.failures)
                            run.save(*generator.records(seeds=seeds))
                            replacements = {}
                            for policy in policies:
                                if policy.id in exc.failures and policy.id not in replacements:
                                    replacements[policy.id] = await generator.repair(
                                        policy, exc.failures[policy.id]
                                    )
                            policies = [
                                replacement
                                for policy in policies
                                if (replacement := replacements.get(policy.id, policy)) is not None
                            ]
                    generator.update(scores)
                    complete = True
                finally:
                    run.save(*generator.records(seeds=seeds, complete=complete))
                show_scores(policies, run, console)
                if not policies:
                    logger.warning("No surviving policies in this generation; continuing")
                if generator.best is not None:
                    logger.info("Best so far: %s", generator.best.name)
                progress.advance(overall)
        except Exception:
            logger.exception("Run failed; saved results and details are in %s", run.path)
            show_scores(run.policies(), run, console)
            raise
        finally:
            for item, (level, propagate) in zip(loggers, old_settings):
                item.removeHandler(display)
                item.removeHandler(log)
                item.setLevel(level)
                item.propagate = propagate
            display.close()
            log.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", action="append", help="Repeat for adaptive model selection")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--archive-size", type=int, default=40)
    parser.add_argument(
        "--parent-selection", choices=("weighted", "uniform", "best", "power"), default="weighted"
    )
    parser.add_argument("--meta-interval", type=int, default=10)
    parser.add_argument("--migration-interval", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    args.model = args.model or ["openai/gpt-oss-120b:nitro"]
    prompts.TEMPLATE_ROOT = Path(shinkaevolve.__file__).parent / "prompts"
    models = [OpenRouterAPI(model=name, max_output_tokens=8192, timeout=120) for name in args.model]
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(make_environment(args.env, max_steps=args.max_steps))
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-shinkaevolve",
                environment=environment,
                executor=Executor(concurrency=args.concurrency),
                path=args.output,
            )
        )
        generator = ShinkaEvolve(
            task="Maximize cumulative episode reward in the described environment.",
            context=environment.instructions,
            provider=models[0],
            ensemble=models,
            config=Config(
                islands=args.islands,
                archive_size=args.archive_size,
                parent_selection=args.parent_selection,
                max_repairs=args.max_repairs,
                max_proposals=args.max_proposals,
                meta_interval=args.meta_interval,
                migration_interval=args.migration_interval,
            ),
            seed=args.search_seed,
        )
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
        )
        try:
            await run_search(
                generator,
                run,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
                seeds=args.seeds,
            )
        except Exception:
            raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(main())
