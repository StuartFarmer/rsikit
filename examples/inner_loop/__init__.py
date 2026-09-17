"""Generate five named policies and pass them directly to a persistent Run."""

import argparse
import json
import os
from pathlib import Path
from statistics import fmean

from jinja2 import Environment, StrictUndefined
from slick import prompts
from slick.providers import OpenRouterAPI

import rsikit.generation as generation
from rsikit import DockerExecutor, Run, generate
from rsikit.episode import PolicyError

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
    with Run.create(
        name="cartpole-comparison",
        environment=ENVIRONMENT,
        path=output,
        max_steps=max_steps,
        record_video=video,
        executor=DockerExecutor(),
        concurrency=concurrency,
    ) as run:
        print(f"Run: {run.path}", flush=True)
        policies = []
        for index, approach in enumerate(APPROACHES, 1):
            print(f"[{index}/5] Generating {approach}...", flush=True)
            policy = await generate(
                task.render(approach=approach, max_steps=max_steps), provider=provider
            )
            policies.append(policy)
        print(f"Evaluating {len(policies)} policies, concurrency={concurrency}...", flush=True)
        try:
            await run.evaluate(*policies, seeds=seeds)
        except PolicyError as exc:
            print(f"Policy evaluation failed: {exc}", flush=True)
        return write_report(run)


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
        with Run.open(args.resume, executor=DockerExecutor(), concurrency=args.concurrency) as run:
            await run.resume()
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
