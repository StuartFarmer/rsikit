"""Explore executable approach families with OpenRouter and Docker until stagnation."""

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

from research import lineagesearch
from research.lineagesearch import Config, LineageSearch
from research.rewards import measure_rewards as measure
from research.rollouts import Rollouts
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment


async def run_search(agent, run, rollouts, *, seeds, heldout_seeds):
    """Show generation, repair, evaluation and family progress; retain messages in run.log."""
    logger = logging.getLogger("research")
    previous_checkpoint = agent.on_checkpoint
    reported_plan = False

    def checkpoint(current):
        nonlocal reported_plan
        run.save(*current.records())
        if previous_checkpoint is not None:
            previous_checkpoint(current)
        if current.study.phase == "search" and (not reported_plan):
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

    agent.on_checkpoint = checkpoint
    try:
        logger.info("Run: %s", run.path)
        study = await agent.run()
        summary = dict(
            reason=study.reason,
            attempts=study.attempts,
            calls=len(study.calls),
            repairs=sum((row.repairs for row in agent.trials)),
            families=[f.model_dump() for f in agent.families],
            best_id=None if agent.best is None else agent.best.id,
        )
        destination = run.path / "summary.json"
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if agent.best is not None:
            logger.info("Evaluating final incumbent on held-out seeds")
            heldout = (await measure(rollouts, [agent.best], heldout_seeds))[agent.best.id]
            summary["heldout"] = dict(scores=heldout.scores, failure=heldout.failure)
            destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
        executor = await stack.enter_async_context(Executor(concurrency=args.concurrency))
        run = await stack.enter_async_context(
            Run.create(
                name=f"{args.env.lower()}-lineagesearch",
                path=args.output,
            )
        )
        rollouts = Rollouts(environment, executor, run)
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
        )

        async def evaluate(policies):
            return await measure(rollouts, policies, args.seeds)

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
        await run_search(agent, run, rollouts, seeds=args.seeds, heldout_seeds=args.heldout_seeds)


if __name__ == "__main__":
    asyncio.run(main())
