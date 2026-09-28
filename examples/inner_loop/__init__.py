"""Generate five named policies and pass them directly to a persistent Run."""

import argparse
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import gymnasium as gym
from jinja2 import Environment, StrictUndefined
from slick import prompts
from slick.providers import OpenRouterAPI

import rsikit.generation as generation
from research.rewards import mean_rewards
from research.rollouts import Rollouts
from rsikit import Executor, Run, generate
from rsikit.evaluation import PolicyError

MODEL = "openai/gpt-oss-120b:nitro"
ENVIRONMENT = "CartPole-v1"
APPROACHES = (
    "random",
    "angle_only",
    "proportional_derivative",
    "full_state_feedback",
    "predictive_rollout",
)


async def run_demo(
    provider, output=None, *, seeds=(0, 1, 2, 3, 4), max_steps=500, video=False, concurrency=2
):
    task = Environment(undefined=StrictUndefined).from_string(
        (Path(__file__).parent / "prompts/task.j2").read_text()
    )
    async with AsyncExitStack() as stack:
        video_folder = stack.enter_context(TemporaryDirectory())
        environment = stack.enter_context(make_environment(max_steps, video, video_folder))
        executor = await stack.enter_async_context(Executor(concurrency=concurrency))
        run = await stack.enter_async_context(
            Run.create(
                name="cartpole-comparison",
                path=output,
            )
        )
        rollouts = Rollouts(environment, executor, run)
        (run.path / "experiment.json").write_text(
            json.dumps(dict(seeds=list(seeds), max_steps=max_steps))
        )
        logger = logging.getLogger("research.inner_loop")
        logger.info(
            "Starting policy comparison",
            extra={
                "progress": dict(
                    kind="search_started",
                    optimizer="Policy comparison",
                    total_candidates=len(APPROACHES),
                    total_generations=1,
                    columns={},
                    resumed=False,
                )
            },
        )
        logger.info(
            "Generating five approaches",
            extra={
                "progress": dict(
                    kind="batch_started",
                    batch_id="generic",
                    label="Generation 1",
                    total_candidates=len(APPROACHES),
                    total_generations=1,
                )
            },
        )
        policies = []
        for index, approach in enumerate(APPROACHES, 1):
            logger.info("[%s/5] Generating %s", index, approach)
            policy = await generate(
                task.render(approach=approach, max_steps=max_steps), provider=provider
            )
            policies.append(policy)
            run.save_policy(policy)
        await evaluate_policies(rollouts, policies, seeds)
        return write_report(run)


async def evaluate_policies(rollouts, policies, seeds, *, resumed=False):
    """Complete this comparison's full seed panel and report its final decisions."""
    logger = logging.getLogger("research.inner_loop")
    policies, seeds = list(policies), tuple(seeds)
    if resumed:
        logger.info(
            "Resuming policy comparison",
            extra={
                "progress": dict(
                    kind="search_started",
                    optimizer="Policy comparison",
                    total_candidates=len(policies),
                    total_generations=1,
                    columns={},
                    resumed=True,
                )
            },
        )
    logger.info(
        "Evaluating %s policies",
        len(policies),
        extra={
            "progress": dict(
                kind="batch_started",
                batch_id="generic",
                label="Generation 1",
                total_candidates=len(policies),
            )
        },
    )
    records = [
        dict(
            kind="candidate",
            batch_id="generic",
            attempt_id=getattr(p, "_progress_attempt", p.id),
            revision=0,
            proposal_done=True,
            policy_id=p.id,
            name=p.name,
            description=p.description,
        )
        for p in policies
    ]
    for row in records:
        logger.info(
            "Evaluating %s", row["name"], extra={"progress": dict(row, status="evaluating")}
        )
    failures = {}
    try:
        await mean_rewards(rollouts, policies, seeds=seeds)
    except PolicyError as exc:
        failures = exc.failures
        logger.exception("Policy evaluation failed")
    leaders = []
    for policy, row in zip(policies, records):
        values = [rollouts.run.scores(policy).get(seed) for seed in seeds]
        complete = (
            bool(values)
            and all(value is not None for value in values)
            and policy.id not in failures
        )
        score = fmean(values) if complete else None
        logger.info(
            "%s: score=%s",
            policy.name,
            score,
            extra={
                "progress": dict(
                    row,
                    status="evaluated" if complete else "failed",
                    score=score,
                    error=failures.get(policy.id),
                )
            },
        )
        if complete:
            leaders.append(
                dict(
                    id=policy.id,
                    name=policy.name,
                    description=policy.description,
                    score=score,
                    generation=1,
                    extras={},
                )
            )
    logger.info(
        "Comparison leaderboard",
        extra={
            "progress": dict(
                kind="leaderboard", rows=sorted(leaders, key=lambda row: -row["score"])
            )
        },
    )
    logger.info(
        "Comparison complete",
        extra={"progress": dict(kind="batch_finished", batch_id="generic", status="completed")},
    )
    logger.info(
        "Search complete",
        extra={
            "progress": dict(
                kind="search_finished",
                status="completed",
                reason="completed with failures" if failures else "completed",
            )
        },
    )


def make_environment(max_steps, video, video_folder):
    env = gym.make(
        ENVIRONMENT, max_episode_steps=max_steps, render_mode="rgb_array" if video else None
    )
    if video:
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda _: True)
    return env


def write_report(run):
    rows = []
    for policy in run.policies():
        scores = run.scores(policy)
        complete = bool(scores) and all(score is not None for score in scores.values())
        rows.append(
            {
                "name": policy.name,
                "scores": {str(seed): score for seed, score in scores.items()},
                "mean_score": fmean(scores.values()) if complete else None,
            }
        )
    (run.path / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| Policy | Mean score | Per-seed scores |", "| --- | ---: | --- |"]
    for row in rows:
        mean = "—" if row["mean_score"] is None else f"{row['mean_score']:.1f}"
        values = ", ".join(
            "unfinished" if score is None else str(score) for score in row["scores"].values()
        )
        lines.append(f"| {row['name'].replace('|', '/')} | {mean} | {values} |")
    (run.path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return rows


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Finish stored episodes; makes no model calls")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.resume:
        async with AsyncExitStack() as stack:
            video_folder = stack.enter_context(TemporaryDirectory())
            run = await stack.enter_async_context(Run.open(args.resume))
            saved = json.loads((run.path / "experiment.json").read_text())
            environment = stack.enter_context(
                make_environment(saved["max_steps"], args.video, video_folder)
            )
            executor = await stack.enter_async_context(Executor(concurrency=args.concurrency))
            await evaluate_policies(
                Rollouts(environment, executor, run), run.policies(), saved["seeds"], resumed=True
            )
            write_report(run)
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(generation.__file__).parent / "prompts"
    provider = OpenRouterAPI(model=MODEL, max_output_tokens=8192, timeout=120)
    await run_demo(
        provider,
        args.output,
        seeds=args.seeds,
        max_steps=args.max_steps,
        video=args.video,
        concurrency=args.concurrency,
    )
