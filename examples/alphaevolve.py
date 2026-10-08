"""Generate, evaluate, and update Gymnasium policies through OpenRouter."""

import argparse
import asyncio
import json
import math
import os
import sqlite3
from contextlib import AsyncExitStack, closing
from dataclasses import asdict
from functools import partial
from pathlib import Path

from slick import prompts
from slick.providers import OpenRouterAPI

from research import alphaevolve
from research.alphaevolve import improved, original, paper
from research.alphaevolve.search import _check_gym_evaluation as _check_gym_evaluation
from research.alphaevolve.search import run_paper_search as run_paper_search
from research.alphaevolve.search import run_search as run_search
from rsikit import Executor, Run
from rsikit.envs.tasks import TASKS, make_environment

FEATURE_BOUNDS = {
    "Bitcoin": {"mean_reward": (-10000, 1000000, 50), "reward_std": (0, 1, 1)},
    "Blackjack": {"mean_reward": (-1000, 1000, 40), "reward_std": (0, 500, 20)},
    "CartPole-v1": {"mean_reward": (0, 500, 20), "reward_std": (0, 250, 10)},
    "LunarLander-v3": {"mean_reward": (-500, 350, 34), "reward_std": (0, 500, 10)},
    "BipedalWalker-v3": {"mean_reward": (-200, 350, 22), "reward_std": (0, 200, 10)},
}


def _resume_settings(parser, args):
    """Restore resolved settings, allowing only budget and worker overrides."""
    if args.output or args.initial_policy or args.search_config:
        parser.error(
            "--resume cannot be combined with --output, --initial-policy or --search-config"
        )
    try:
        saved = json.loads((args.resume / "experiment.json").read_text(encoding="utf-8"))
        if saved["variant"] != "paper":
            raise ValueError("Only paper runs have resumable optimizer checkpoints")
        population = args.resume / "population.sqlite"
        if not population.is_file():
            raise ValueError("Run has no population checkpoint (population.sqlite)")
        with closing(sqlite3.connect(population.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT value FROM state WHERE key = 'optimizer'").fetchone()
            state = json.loads(row[0]) if row is not None else None
            if (
                not isinstance(state, dict)
                or not {"task", "context", "completed", "rng"} <= state.keys()
            ):
                raise ValueError("Population checkpoint has no saved optimizer state")
        locked = (
            "variant",
            "search_seed",
            "env",
            "model",
            "ensemble",
            "max_steps",
            "seeds",
            "screening_seeds",
            "screening_min_reward",
        )
        defaults = {
            name: saved[name]
            for name in (
                *locked,
                "generations",
                "batch_size",
                "generation_concurrency",
                "concurrency",
            )
        }
        defaults.update(mode=saved["config"]["mode"], max_repairs=saved["config"]["max_repairs"])
        parser.set_defaults(**defaults)
        args = parser.parse_args()
        for name in (*locked, "mode", "max_repairs"):
            if getattr(args, name) != defaults[name]:
                raise ValueError(f"--{name.replace('_', '-')} cannot change when resuming a run")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        parser.error(f"Cannot resume run: {exc}")
    return args, saved


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("paper", "original", "improved"), default="paper")
    parser.add_argument("--search-seed", type=int, default=0, help="Optimizer sampling seed")
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument(
        "--generations",
        type=int,
        default=25,
        help="Proposal batches to attempt; additional batches when resuming (default: 25)",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-repairs", type=int, help="Override Config.max_repairs (default: 2)")
    parser.add_argument("--mode", choices=("diff", "rewrite"))
    parser.add_argument("--search-config", type=Path, help="JSON Config fields; CLI overrides win")
    parser.add_argument(
        "--ensemble", nargs=2, action="append", default=[], metavar=("MODEL", "WEIGHT")
    )
    parser.add_argument(
        "--initial-policy", type=Path, help="Paper variant: evaluate and seed a policy"
    )
    parser.add_argument("--screening-seeds", type=int, nargs="+")
    parser.add_argument("--screening-min-reward", type=float)
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=4,
        help="Concurrent model proposals and runtime repairs (default: 4)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent episode evaluations (default: 4)"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override the environment's native time limit"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue a paper run using its saved settings")
    args = parser.parse_args()
    saved = None
    if args.resume is not None:
        args, saved = _resume_settings(parser, args)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    if (
        args.generations < 0
        or min(args.batch_size, args.generation_concurrency, args.concurrency) < 1
    ):
        parser.error(
            "Generations must be nonnegative; batch size and worker counts must be positive"
        )

    if bool(args.screening_seeds) != (args.screening_min_reward is not None):
        parser.error("--screening-seeds and --screening-min-reward must be supplied together")
    if args.screening_min_reward is not None and not math.isfinite(args.screening_min_reward):
        parser.error("--screening-min-reward must be finite")
    if args.variant != "paper" and (args.initial_policy or args.screening_seeds):
        parser.error("Initial policies and screening require --variant paper")
    variant = {"paper": paper, "original": original, "improved": improved}[args.variant]
    try:
        options = (
            dict(saved["config"])
            if saved is not None
            else {}
            if args.search_config is None
            else json.loads(args.search_config.read_text())
        )
        if not isinstance(options, dict):
            raise ValueError("Search config must be a JSON object")
        options["mode"] = args.mode or options.get(
            "mode", "rewrite" if args.env == "BipedalWalker-v3" else "diff"
        )
        if args.max_repairs is not None:
            options["max_repairs"] = args.max_repairs
        if args.variant == "paper":
            options.setdefault("features", FEATURE_BOUNDS[args.env])
        config = variant.Config(**options)
        if args.variant == "paper":
            _check_gym_evaluation(
                config.features,
                args.seeds,
                args.screening_seeds,
                args.screening_min_reward,
                config.objective,
            )
        models = [(args.model, 1.0)] + [(name, float(weight)) for name, weight in args.ensemble]
        if any(not math.isfinite(weight) or weight <= 0 for _, weight in models):
            raise ValueError("Ensemble weights must be positive and finite")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.mode = config.mode
    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent
    executor = Executor(concurrency=args.concurrency)
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(make_environment(args.env, max_steps=args.max_steps))
        await stack.enter_async_context(executor)
        run = await stack.enter_async_context(
            Run.open(args.resume)
            if args.resume
            else Run.create(
                name=f"{args.env.lower()}-{args.variant}-evolution",
                path=args.output,
            )
        )
        evaluator = partial(run.evaluate, environment=environment, executor=executor)
        providers = [
            (OpenRouterAPI(model=name, max_output_tokens=8192, timeout=120), weight)
            for name, weight in models
        ]
        generator = variant.AlphaEvolve(
            task="Maximize cumulative episode reward in the described environment.",
            context=environment.instructions,
            provider=providers[0][0],
            ensemble=providers,
            config=config,
            seed=args.search_seed,
            **(
                {"database_path": run.path / "population.sqlite"} if args.variant == "paper" else {}
            ),
        )
        if saved is None:
            (run.path / "experiment.json").write_text(
                json.dumps(
                    {**vars(args), "config": asdict(config), "optimization_schedule": "round-v1"},
                    default=str,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        elif saved.get("optimization_schedule") != "round-v1":
            with (run.path / "schedule_changes.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"from": saved.get("optimization_schedule"), "to": "round-v1"})
                    + "\n"
                )
        try:
            await run_search(
                generator,
                run,
                evaluator,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
                seeds=args.seeds,
                screening_seeds=args.screening_seeds or (),
                screening_min_reward=args.screening_min_reward,
                initial_policy=args.initial_policy,
            )
        except Exception:
            # run_search has already displayed and saved the traceback.
            raise SystemExit(1) from None
        finally:
            if args.variant == "paper":
                generator.close()


if __name__ == "__main__":
    asyncio.run(main())
