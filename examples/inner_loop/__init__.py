"""Generate five policies with OpenRouter, then compare isolated Gymnasium episodes."""

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, Field, ValidationError
from slick import prompt, prompts
from slick.providers import OpenRouterAPI, ProviderError

from rsikit import run_program
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


class PolicyProposal(BaseModel, extra="forbid"):
    summary: str = Field(min_length=1)
    source: str = Field(min_length=1)


@prompt(template="generate.j2", output_type=PolicyProposal)
async def generate_policy(
    approach: str, max_steps: int, *, generated: PolicyProposal
) -> PolicyProposal:
    return generated


async def run_demo(provider, output: Path, *, seeds=(0, 1, 2, 3, 4), max_steps=500):
    """Make five generation calls; evaluate every valid policy on the same seeds.

    The caller configures Slick's template root once. No candidate is imported on
    the host. Invalid generation/policy results remain visible in the report;
    environment or Docker failures propagate. There are no generation retries.
    """
    output.mkdir(parents=True, exist_ok=False)
    rows, signatures = [], set()
    for index, approach in enumerate(APPROACHES, 1):
        print(f"[{index}/5] Generating {approach}...", flush=True)
        path = output / f"{index:02d}_{approach}.py"
        row = {
            "approach": approach,
            "file": None,
            "summary": "",
            "episodes": [],
            "mean_return": None,
            "error": "",
        }
        rows.append(row)
        try:
            proposal = await generate_policy(approach, max_steps, provider=provider)
            path.write_text(proposal.source, encoding="utf-8")
            row.update(file=path.name, summary=proposal.summary)
            signature = ast.dump(ast.parse(proposal.source), include_attributes=False)
        except (ValidationError, SyntaxError, ProviderError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            continue
        if signature in signatures:
            row["error"] = "Duplicate policy: its Python syntax matches an earlier proposal"
            continue
        signatures.add(signature)
        print(f"[{index}/5] Evaluating {approach} on {len(seeds)} seeds...", flush=True)
        try:
            for seed in seeds:
                _, _, terminated, truncated, info = await run_program(
                    path,
                    ENVIRONMENT,
                    env_seed=seed,
                    policy_seed=seed,
                    max_steps=max_steps,
                )
                row["episodes"].append(
                    {
                        "seed": seed,
                        "return": info["episode"]["r"],
                        "length": info["episode"]["l"],
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
        except PolicyError as exc:
            row["error"] = f"Seed {seed}: {type(exc).__name__}: {exc}"
            continue
        row["mean_return"] = fmean(episode["return"] for episode in row["episodes"])

    result = {
        "model": provider.model,
        "environment": ENVIRONMENT,
        "seeds": list(seeds),
        "max_steps": max_steps,
        "policies": rows,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        f"# Five generated policies on {ENVIRONMENT}",
        "",
        f"Model: `{provider.model}`. Seeds: {list(seeds)}. Step cap: {max_steps}.",
        "",
        "Reward is the number of surviving steps; higher is better. Failed policies have no mean.",
        "",
        "| Policy | Mean return | Per-seed returns | Status |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        mean = "—" if row["mean_return"] is None else f"{row['mean_return']:.1f}"
        rewards = ", ".join(f"{ep['return']:g}" for ep in row["episodes"]) or "—"
        status = "failed" if row["error"] else "ok"
        report.append(f"| {row['approach']} | {mean} | {rewards} | {status} |")
    for row in rows:
        report.extend(["", f"## {row['approach']}", "", row["summary"]])
        if row["file"]:
            report.extend(["", f"[Generated source]({row['file']})"])
        if row["error"]:
            report.extend(["", "```text", row["error"], "```"])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[6 : 8 + len(rows)]))
    print(f"\nSources and full results: {output}")
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs")
        / ("inner-loop-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    prompts.TEMPLATE_ROOT = Path(__file__).parent / "prompts"
    provider = OpenRouterAPI(model=MODEL, max_output_tokens=8192, timeout=120)
    await run_demo(provider, args.output, seeds=args.seeds, max_steps=args.max_steps)
