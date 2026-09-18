"""Generate, evaluate, and update Gymnasium policies through OpenRouter."""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from statistics import fmean

import gymnasium as gym
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from slick import prompts
from slick.providers import OpenRouterAPI

import rsikit.alphaevolve as alphaevolve
from rsikit import Executor, Run
from rsikit.alphaevolve import AlphaEvolve, Config
from rsikit.episode import PolicyError

TASKS = {
    "CartPole-v1": (
        "Balance the pole for as many steps as possible. Observation: [cart position, "
        "cart velocity, pole angle, pole angular velocity]. Action 0 pushes left; "
        "1 pushes right. Each surviving step earns 1 reward. The episode terminates "
        "when the pole tilts too far or the cart leaves the allowed range."
    ),
    "LunarLander-v3": (
        "Land safely near the pad at (0, 0), maximizing cumulative reward. Wind and "
        "turbulence are enabled. Observation indices: 0 horizontal position, 1 vertical "
        "position, 2 horizontal velocity, 3 vertical velocity, 4 angle, 5 angular "
        "velocity, 6 left leg contact, 7 right leg contact. Positions and velocities "
        "are normalized simulator values. Return a float32 array [main, lateral], "
        "each in [-1, 1]. Main < 0 turns the main engine off; main in [0, 1] maps to "
        "50-100% power. Lateral in (-0.5, 0.5) turns side engines off; below -0.5 "
        "fires the left booster, above 0.5 fires the right, with magnitude mapping "
        "to 50-100% power. Reward favors approaching the pad, slowing down, staying "
        "upright and leg contact; firing engines costs reward. A crash costs 100; "
        "a safe landing earns 100. Termination: crash, leaving the horizontal bounds, "
        "or coming to rest. Initial force and wind vary with the episode seed."
    ),
    "BipedalWalker-v3": (
        "Walk to the right over uneven terrain without falling, maximizing cumulative "
        "reward. Normal terrain, not hardcore. Observation indices: 0 hull angle, "
        "1 scaled hull angular velocity, 2-3 scaled horizontal/vertical velocity; "
        "4 hip angle, 5 scaled hip speed, 6 knee angle plus 1, 7 scaled knee speed, "
        "8 foot contact for the first leg; 9-13 the same five values for the second "
        "leg; 14-23 ten lidar fractions (0 near, 1 far). Return a float32 array of "
        "four motor commands in [-1, 1]: first hip, first knee, second hip, second "
        "knee. Sign sets motor direction; magnitude limits torque. Reward favors "
        "forward progress and an upright hull, with a motor-effort penalty. Falling "
        "costs 100. Termination: hull contacts ground or reaching the terrain end. "
        "Terrain and initial push vary with the episode seed."
    ),
}


def make_environment(name, *, max_steps=None, render_mode=None):
    options = {"continuous": True, "enable_wind": True} if name == "LunarLander-v3" else {}
    if max_steps is not None:
        options["max_episode_steps"] = max_steps
    # Keep instructions on a wrapper: Box2D's EzPickle reconstructs the base env.
    env = gym.Wrapper(gym.make(name, render_mode=render_mode, **options))
    env.instructions = (
        f"{name}: {TASKS[name]}\n"
        f"Observation space: {env.observation_space}\nAction space: {env.action_space}\n"
        f"The episode is truncated after {env.spec.max_episode_steps} steps."
    )
    return env


class _ProgressHandler(RichHandler):
    def __init__(self, progress):
        super().__init__(
            console=progress.console,
            show_path=False,
            markup=False,
            highlighter=None,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        self.progress = progress
        self.generation = progress.add_task("Generating policies", total=0, visible=False)
        self.evaluation = progress.add_task("Evaluating policies", total=0, visible=False)

    def emit(self, record):
        event = getattr(record, "event", None)
        if event == "generation_started":
            self.progress.update(self.evaluation, visible=False)
            self.progress.reset(self.generation, total=record.total, visible=True)
        elif event in ("policy_generated", "proposal_discarded"):
            self.progress.advance(self.generation)
        elif event == "evaluation_started":
            self.progress.update(self.generation, visible=False)
            self.progress.reset(self.evaluation, total=record.total, visible=True)
        elif event in ("policy_evaluated", "evaluation_failed"):
            self.progress.advance(self.evaluation)
        super().emit(record)


def _show_scores(policies, run, console):
    table = Table("Policy", "Description", "Score")
    for policy in policies:
        values = list(run.scores(policy).values())
        score = (
            f"{fmean(values):.1f}"
            if values and all(v is not None for v in values)
            else "unfinished"
        )
        table.add_row(Text(policy.name), Text(policy.description), score)
    console.print(table)


async def run_search(
    generator, run, *, generations, batch_size, generation_concurrency=4, seeds=(0,), console=None
):
    """Display completed policies immediately and keep the same messages in run.log."""
    seeds = tuple(seeds)
    console = console or Console()
    logger = logging.getLogger("rsikit")
    old_level, old_propagate = logger.level, logger.propagate
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Generations", total=generations)
        display = _ProgressHandler(progress)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(display)
        logger.addHandler(log)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info("Run: %s", run.path)
            for generation in range(generations):
                logger.info("Generation %s/%s", generation + 1, generations)
                policies = await generator.generate(
                    n=batch_size, concurrency=generation_concurrency
                )
                scores = {}
                while policies:
                    try:
                        scores = await run.evaluate(policies, seeds=seeds)
                        break
                    except PolicyError as exc:
                        if not exc.failures:
                            raise
                        replacements = {}
                        for policy in policies:
                            if policy.id in exc.failures and policy.id not in replacements:
                                replacements[policy.id] = await generator.repair(
                                    policy, exc.failures[policy.id]
                                )
                        policies = [
                            replacement
                            for policy in policies
                            if (replacement := replacements.get(policy.id, policy)) is not None
                        ]
                generator.update(
                    scores,
                    seed_scores={
                        policy.id: {
                            seed: score
                            for seed, score in run.scores(policy).items()
                            if seed in seeds
                        }
                        for policy in policies
                    },
                )
                _show_scores(policies, run, console)
                if not policies:
                    logger.warning("No surviving policies in this generation; continuing")
                if generator.best is not None:
                    logger.info("Best so far: %s", generator.best.name)
                progress.advance(overall)
        except Exception:
            logger.exception("Run failed; saved results and details are in %s", run.path)
            _show_scores(run.policies(), run, console)
            raise
        finally:
            logger.removeHandler(display)
            logger.removeHandler(log)
            display.close()
            log.close()
            logger.setLevel(old_level)
            logger.propagate = old_propagate


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=4,
        help="Concurrent policy proposals (default: 4)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent sandbox evaluations (default: 4)"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override the environment's native time limit"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")

    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"
    executor = Executor(concurrency=args.concurrency)
    with (
        make_environment(args.env, max_steps=args.max_steps) as environment,
        Run.create(
            name=f"{args.env.lower()}-evolution",
            environment=environment,
            executor=executor,
            path=args.output,
        ) as run,
    ):
        generator = AlphaEvolve(
            task="Maximize cumulative episode reward in the described environment.",
            context=environment.instructions,
            provider=OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120),
            config=Config(max_repairs=args.max_repairs),
        )
        try:
            await run_search(
                generator,
                run,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
                seeds=args.seeds,
            )
        except Exception:
            # run_search has already displayed and saved the traceback.
            raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(main())
