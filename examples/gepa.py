"""Run GEPA policy search through Run evaluation; launch inside Docker."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from statistics import fmean

from slick import prompts
from slick.providers import OpenRouterAPI

from research import gepa
from research.gepa import GEPA
from research.gepa.records import GEPAEvent
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
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--initial", type=Path, required=True)
    args = parser.parse_args()
    args.seeds = list(dict.fromkeys(args.seeds))
    args.test_seeds = list(dict.fromkeys(args.test_seeds))
    args.selection_seeds = list(dict.fromkeys(args.selection_seeds))
    if set(args.seeds) & set(args.test_seeds):
        parser.error("Training and final test seeds must be disjoint")
    if set(args.seeds) & set(args.selection_seeds) or set(args.selection_seeds) & set(
        args.test_seeds
    ):
        parser.error("Training, selection, and test seeds must be disjoint")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(gepa.__file__).parent / "prompts"
    provider = OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120)
    environment = make_environment(args.env, max_steps=args.max_steps)
    with environment:
        async with (
            Executor(concurrency=args.concurrency) as executor,
            Run.create(name=f"{args.env}-gepa", path=args.output) as run,
        ):
            (run.path / "experiment.json").write_text(
                json.dumps(vars(args), default=str, indent=2) + "\n"
            )
            logger = logging.getLogger(__name__)
            logger.info(
                "Starting GEPA",
                extra={
                    "progress": dict(
                        kind="search_started",
                        optimizer="GEPA",
                        total_candidates=None,
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
                        total_candidates=None,
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
                run.save(GEPAEvent(kind=kind, data=data))
                source = data.get("policy") or data.get("text")
                if source:
                    policy = PolicyDefinition.from_text(source)
                    run.save_policy(policy)
                    scores = data.get("scores")
                    score = (
                        fmean(scores.values() if isinstance(scores, dict) else scores)
                        if scores and data.get("accepted", True)
                        else None
                    )
                    logger.info(
                        "%s: %s",
                        policy.name,
                        kind,
                        extra={
                            "progress": dict(
                                kind="candidate",
                                batch_id="search",
                                attempt_id=str(data.get("id", policy.id)),
                                status="evaluated" if score is not None else "generated",
                                proposal_done=True,
                                policy_id=policy.id,
                                name=policy.name,
                                description=policy.description,
                                score=score,
                            )
                        },
                    )
                elif kind in ("rejected", "strategy_rejected"):
                    logger.warning("%s: %s", kind, data.get("error", ""))

            task = environment.instructions
            agent = GEPA(task, provider, evaluate, run.load_episode, on_event=record)
            await agent.run(
                PolicyDefinition.from_file(args.initial),
                train_seeds=args.seeds,
                selection_seeds=args.selection_seeds,
                budget=args.budget,
                minibatch_size=args.minibatch,
                seed=args.search_seed,
            )
            summary = dict(
                executed_search_episodes=executed_episodes,
                best_policy=agent.best.id if agent.best else None,
            )
            summary["logical_search_episodes"] = agent.result.rollouts
            summary["reflection_calls"] = agent.result.reflection_calls
            if agent.best is not None:
                summary["selection_scores"] = {
                    seed: run.scores(agent.best)[seed] for seed in args.selection_seeds
                }
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
