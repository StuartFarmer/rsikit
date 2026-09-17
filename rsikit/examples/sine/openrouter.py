"""Optimize sine approximation with OpenRouter GPT-OSS 120B Nitro."""

import argparse
import asyncio
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path

from slick import prompt, prompts
from slick.providers import OpenRouterAPI

from rsikit import Evaluation, SlickProposer

from .demo import run_experiment

MODEL = "openai/gpt-oss-120b:nitro"
TASK = (
    "Approximate sin(x) on [-1, 1], minimizing mean squared error. "
    "Define approximate(x) with exactly one return expression using only x, numeric "
    "constants and arithmetic (+, -, *, /, **). No imports, calls, assignments, "
    "decorators or extra functions. Preserve all text outside the evolution block "
    "and the marker lines exactly. Return finite numbers at every point on [-1, 1]."
)


@prompt(template="sine_repair.j2")
async def repair_source(
    source: str, evaluation: Evaluation, initial: str, *, generated: str
) -> str:
    """Return corrected source from the failed check's diagnostics."""
    return generated


async def run(
    directory: Path, provider, *, iterations=3, max_repairs=2, timeout=180, repair_demo=False
):
    initial = Path(__file__).with_name("initial.py").read_text(encoding="utf-8")
    proposer = SlickProposer(TASK, provider)
    first = True

    async def propose(parent, history):
        nonlocal first
        if first and repair_demo:
            first = False
            print(
                "Using a deliberately broken first proposal to exercise model repair.", flush=True
            )
            return initial.replace("return x", "return x - x**3 /")
        first = False
        return await proposer(parent, history)

    async def repair(source, evaluation):
        print(f"  Repairing: {evaluation.feedback}", flush=True)
        return await repair_source(source, evaluation, initial, provider=provider)

    return await run_experiment(
        directory,
        propose,
        repair,
        iterations=iterations,
        max_repairs=max_repairs,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="seconds for each generation, including its repairs",
    )
    parser.add_argument(
        "--repair-demo",
        action="store_true",
        help="supply a broken first proposal for the model to repair",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / f"rsikit-openrouter-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
    )
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example.")
    if importlib.util.find_spec("openai") is None:
        parser.error("Install the SDK: uv pip install --python .venv/bin/python 'openai>=2,<3'")
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "prompts"
    provider = OpenRouterAPI(
        args.model, timeout=args.timeout, max_output_tokens=args.max_tokens, max_retries=0
    )
    print(f"Model: {args.model}", flush=True)
    asyncio.run(
        run(
            args.output,
            provider,
            iterations=args.iterations,
            max_repairs=args.max_repairs,
            timeout=args.timeout,
            repair_demo=args.repair_demo,
        )
    )


if __name__ == "__main__":
    main()
