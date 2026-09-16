"""Run the complete circle-packing optimizer against OpenRouter GPT-OSS 120B Nitro."""

import argparse
import asyncio
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path

from slick import prompts
from slick.providers import OpenRouterAPI

from .experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=("hillclimb", "alphaevolve", "eoh", "dgm-archive"),
        default="hillclimb",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=4, help="EoH population size")
    parser.add_argument("--islands", type=int, default=4, help="AlphaEvolve island count")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--evaluation-timeout", type=float, default=10, help="Seconds per function execution"
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / f"circle-packing-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
    )
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running.")
    if importlib.util.find_spec("openai") is None:
        parser.error(
            "Install: uv pip install --python optimizer/.venv/bin/python "
            "-r rsikit/examples/circle_packing/requirements.txt"
        )
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    provider = OpenRouterAPI(
        args.model, timeout=args.timeout, max_output_tokens=args.max_tokens, max_retries=0
    )
    print(f"Model: {args.model}\nStrategy: {args.strategy}", flush=True)
    strategy = asyncio.run(
        run(
            args.output,
            provider,
            iterations=args.iterations,
            max_repairs=args.max_repairs,
            evaluation_timeout=args.evaluation_timeout,
            timeout=args.timeout,
            strategy_name=args.strategy,
            seed=args.seed,
            population_size=args.population_size,
            islands=args.islands,
        )
    )
    improved = (
        strategy.best.evaluation.metrics["sum_radii"]
        > strategy.history[0].evaluation.metrics["sum_radii"]
    )
    print(
        "PASS: improved a valid packing."
        if improved
        else "NO IMPROVEMENT: best remains the baseline."
    )
    return 0 if improved else 2


if __name__ == "__main__":
    raise SystemExit(main())
