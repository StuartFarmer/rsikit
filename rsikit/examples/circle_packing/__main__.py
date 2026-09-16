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
        choices=("hillclimb", "alphaevolve", "eoh", "dgm-archive", "shinkaevolve"),
        default="hillclimb",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=("legacy", "modular"), default="legacy")
    parser.add_argument(
        "--instruction-file", type=Path, help="Plain-text mutate instruction (modular mode)"
    )
    parser.add_argument(
        "--reflect", action="store_true", help="Use measured reflection (modular mode)"
    )
    parser.add_argument("--population-size", type=int, default=4, help="EoH population size")
    parser.add_argument(
        "--islands", type=int, default=4, help="AlphaEvolve/ShinkaEvolve island count"
    )
    parser.add_argument(
        "--ensemble-model",
        action="append",
        default=[],
        help="Additional ShinkaEvolve mutation model; repeat for an ensemble",
    )
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
    if args.strategy == "shinkaevolve":
        args.prompt_mode = "modular"
    elif args.ensemble_model:
        parser.error("--ensemble-model requires --strategy shinkaevolve")
    if (args.reflect or args.instruction_file) and args.prompt_mode != "modular":
        parser.error("--reflect and --instruction-file require --prompt-mode modular")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running.")
    if importlib.util.find_spec("openai") is None:
        parser.error(
            "Install: uv pip install --python optimizer/.venv/bin/python "
            "-r rsikit/examples/circle_packing/requirements.txt"
        )
    prompts.TEMPLATE_ROOT = (
        Path(__file__).resolve().parents[3]
        if args.prompt_mode == "modular"
        else Path(__file__).resolve().parent / "prompts"
    )
    provider = OpenRouterAPI(
        args.model, timeout=args.timeout, max_output_tokens=args.max_tokens, max_retries=0
    )
    ensemble = (
        (
            provider,
            *(
                OpenRouterAPI(
                    model, timeout=args.timeout, max_output_tokens=args.max_tokens, max_retries=0
                )
                for model in args.ensemble_model
            ),
        )
        if args.ensemble_model
        else ()
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
            prompt_mode=args.prompt_mode,
            instructions={
                operation: args.instruction_file.read_text(encoding="utf-8")
                for operation in (
                    ("diff", "full", "cross") if args.strategy == "shinkaevolve" else ("mutate",)
                )
            }
            if args.instruction_file
            else None,
            reflect=args.reflect,
            shinka_providers=ensemble,
            model_settings={
                "model": args.model,
                "max_output_tokens": args.max_tokens,
                "temperature": None,
                "max_retries": 0,
            },
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
