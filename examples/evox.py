"""Run EvoX policy search through Run evaluation; launch inside Docker."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from slick import prompts
from slick.providers import OpenRouterAPI

from research import evox
from research.evox import Config, EvoX
from research.evox.records import EvoXEvent
from research.evox.runtime import run_python_strategy
from rsikit import Executor, PolicyDefinition, Run
from rsikit.envs.tasks import TASKS, make_environment
from rsikit.evaluation import episode_error, episode_scores


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--window", type=int)
    args = parser.parse_args()
    args.seeds = list(dict.fromkeys(args.seeds))
    args.test_seeds = list(dict.fromkeys(args.test_seeds))
    if set(args.seeds) & set(args.test_seeds):
        parser.error("Training and final test seeds must be disjoint")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(evox.__file__).parent / "prompts"
    provider = OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120)
    environment = make_environment(args.env, max_steps=args.max_steps)
    with environment:
        async with (
            Executor(concurrency=args.concurrency) as executor,
            Run.create(name=f"{args.env}-evox", path=args.output) as run,
        ):
            (run.path / "experiment.json").write_text(
                json.dumps(vars(args), default=str, indent=2) + "\n"
            )
            logger = logging.getLogger(__name__)
            logger.info(
                "Starting EvoX",
                extra={
                    "progress": dict(
                        kind="search_started",
                        optimizer="EvoX",
                        total_candidates=args.iterations,
                        total_generations=None,
                        columns={},
                    )
                },
            )
            logger.info(
                "Search candidates",
                extra={
                    "progress": dict(
                        kind="batch_started",
                        batch_id="search",
                        label="Search",
                        total_candidates=args.iterations,
                    )
                },
            )
            executed_episodes = 0

            async def evaluate(policies, *, seeds=None):
                nonlocal executed_episodes
                seeds = tuple(args.seeds if seeds is None else seeds)
                executed_episodes += sum(
                    (cached := run.load_episode(policy, seed)) is None or cached.error is not None
                    for policy in policies
                    for seed in seeds
                )
                return await run.evaluate(
                    policies, environment=environment, executor=executor, seeds=seeds
                )

            def record(kind, data):
                run.save(EvoXEvent(kind=kind, data=data))
                source = data.get("policy") or data.get("text")
                if source:
                    policy = PolicyDefinition.from_text(source)
                    run.save_policy(policy)
                    logger.info(
                        "%s: %s",
                        policy.name,
                        kind,
                        extra={
                            "progress": dict(
                                kind="candidate",
                                batch_id="search",
                                attempt_id=str(data.get("id", policy.id)),
                                status="evaluated"
                                if data.get("score") is not None
                                else "generated",
                                proposal_done=True,
                                policy_id=policy.id,
                                name=policy.name,
                                description=policy.description,
                                score=data.get("score"),
                            )
                        },
                    )
                elif kind in ("rejected", "strategy_rejected"):
                    logger.warning("%s: %s", kind, data.get("error", ""))

            task = environment.instructions
            agent = EvoX(
                task,
                provider,
                evaluate,
                run_strategy=run_python_strategy,
                config=Config(
                    iterations=args.iterations, window=args.window, seed=args.search_seed
                ),
                on_event=record,
            )
            await agent.run()
            summary = dict(
                executed_search_episodes=executed_episodes,
                best_policy=agent.best.id if agent.best else None,
            )
            summary["model_calls"] = len(agent.result.generations)
            summary["model_calls_by_channel"] = {
                channel: sum(row.channel == channel for row in agent.result.generations)
                for channel in ("solution", "strategy", "operators")
            }
            if agent.best is not None:
                measured = (
                    await run.evaluate(
                        [agent.best],
                        environment=environment,
                        executor=executor,
                        seeds=args.test_seeds,
                    )
                )[agent.best.id]
                summary["final_test_scores"] = episode_scores(measured)
                summary["final_test_accepted"] = episode_error(measured) is None
                summary["final_test_failure"] = episode_error(measured)
                agent.best.to_file(run.path / "best.py")
            (run.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            logger.info(
                "Search complete",
                extra={"progress": dict(kind="search_finished", status="completed")},
            )
            print(run.path)


if __name__ == "__main__":
    asyncio.run(main())
