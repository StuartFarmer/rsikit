"""Evolve Gymnasium policies with ShinkaEvolve using OpenRouter and Docker."""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path

from slick import prompts
from slick.providers import OpenRouterAPI

from research import shinkaevolve
from research.shinkaevolve import Config, ShinkaEvolve
from research.shinkaevolve.search import run_search as run_search
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment


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
        executor = await stack.enter_async_context(Executor(concurrency=args.concurrency))
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-shinkaevolve",
                path=args.output,
            )
        )
        evaluator = partial(run.evaluate, environment=environment, executor=executor)
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
                evaluator,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
                seeds=args.seeds,
            )
        except Exception:
            raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(main())
