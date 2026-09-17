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
from rsikit import Run, generate

MODEL = "openai/gpt-oss-120b:nitro"
ENVIRONMENT = "CartPole-v1"
APPROACHES = (
    "random",
    "angle_only",
    "proportional_derivative",
    "full_state_feedback",
    "predictive_rollout",
)


async def run_demo(provider, output=None, *, seeds=(0, 1, 2, 3, 4), max_steps=500, video=False):
    task = Environment(undefined=StrictUndefined).from_string(
        (Path(__file__).parent / "prompts/task.j2").read_text()
    )
    with Run.create(
        name="cartpole-comparison",
        environment=ENVIRONMENT,
        path=output,
        max_steps=max_steps,
        record_video=video,
    ) as run:
        print(f"Run: {run.path}", flush=True)
        for index, approach in enumerate(APPROACHES, 1):
            print(f"[{index}/5] Generating {approach}...", flush=True)
            policy = await generate(
                task.render(approach=approach, max_steps=max_steps), provider=provider
            )
            print(f"[{index}/5] Evaluating {policy.name}...", flush=True)
            await run.evaluate(policy, seeds=seeds)
        return write_report(run)


def write_report(run):
    rows = []
    for policy in run.policies():
        episodes = run.executions(policy)
        complete = bool(episodes) and all(ep.status == "completed" for ep in episodes)
        rows.append(
            {
                "name": policy.name,
                "id": policy.id,
                "summary": policy.summary,
                "episodes": [ep.model_dump(mode="json") for ep in episodes],
                "mean_return": fmean(ep.reward for ep in episodes) if complete else None,
            }
        )
    (run.path / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| Policy | Mean return | Per-seed returns |", "| --- | ---: | --- |"]
    for row in rows:
        mean = "—" if row["mean_return"] is None else f"{row['mean_return']:.1f}"
        values = ", ".join(
            str(ep["reward"]) if ep["status"] == "completed" else ep["status"]
            for ep in row["episodes"]
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
    args = parser.parse_args()
    if args.resume:
        with Run.open(args.resume) as run:
            await run.resume()
            write_report(run)
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(generation.__file__).parent / "prompts"
    provider = OpenRouterAPI(model=MODEL, max_output_tokens=8192, timeout=120)
    await run_demo(
        provider, args.output, seeds=args.seeds, max_steps=args.max_steps, video=args.video
    )
