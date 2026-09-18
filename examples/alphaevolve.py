"""Generate, evaluate, and update CartPole policies through OpenRouter."""

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
        elif event == "policy_generated":
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
    generator, run, *, generations, batch_size, generation_concurrency=4, console=None
):
    """Display completed policies immediately and keep the same messages in run.log."""
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
                while True:
                    try:
                        scores = await run.evaluate(policies)
                        break
                    except PolicyError as exc:
                        if not exc.failures or generator.config.max_repairs == 0:
                            raise
                        replacements = {}
                        for policy in policies:
                            if policy.id in exc.failures and policy.id not in replacements:
                                replacements[policy.id] = await generator.repair(
                                    policy, exc.failures[policy.id]
                                )
                        policies = [replacements.get(policy.id, policy) for policy in policies]
                generator.update(scores)
                _show_scores(policies, run, console)
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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")

    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"
    generator = AlphaEvolve(
        task="Balance the CartPole pole for as many steps as possible.",
        context=(
            "CartPole-v1: observation is [cart position, cart velocity, pole angle, "
            "pole angular velocity]. Action 0 pushes left; 1 pushes right. "
            f"Each surviving step earns 1 reward, up to {args.max_steps} steps."
        ),
        provider=OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120),
        config=Config(max_repairs=args.max_repairs),
    )
    executor = Executor(concurrency=args.concurrency)
    with (
        gym.make("CartPole-v1", max_episode_steps=args.max_steps) as environment,
        Run.create(
            name="cartpole-evolution", environment=environment, executor=executor, path=args.output
        ) as run,
    ):
        try:
            await run_search(
                generator,
                run,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
            )
        except Exception:
            # run_search has already displayed and saved the traceback.
            raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(main())
