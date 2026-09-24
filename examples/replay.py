"""Record the best saved policies in a separate Run, without model calls."""

import argparse
import asyncio
import logging
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import gymnasium as gym
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from examples.alphaevolve import TASKS, make_environment
from rsikit import Executor, Run


async def record_best(
    path, environment, *, top=3, seeds=(0,), output=None, executor=None, console=None
) -> Path:
    """Rank complete saved scores, then evaluate selected policies in a fresh run."""
    console = console or Console()
    with Run.open(path, environment=environment) as source:
        ranked = []
        for policy in source.policies():
            scores = list(source.scores(policy).values())
            if scores and all(score is not None for score in scores):
                ranked.append((fmean(scores), policy))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:top]
        if not selected:
            raise ValueError("No policies with complete scores to replay")
        name = f"{source.name}-videos"
        console.print(f"Source run: {source.path}", markup=False)

    async with Run.create(
        name=name, environment=environment, path=output, executor=executor
    ) as replay:
        console.print(f"Video run: {replay.path}", markup=False)
        scores = await replay.evaluate([policy for _, policy in selected], seeds=seeds)
        table = Table("Policy", "Original mean", "Replay mean")
        for original, policy in selected:
            table.add_row(Text(policy.name), f"{original:.1f}", f"{scores[policy.id]:.1f}")
        console.print(table)
        for _, policy in selected:
            console.print(policy.name, markup=False)
            for video in sorted((replay.path / "artifacts" / policy.id).rglob("*.mp4")):
                console.print(str(video), markup=False)
        return replay.path


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Source run directory; close the search first")
    parser.add_argument(
        "--env", choices=TASKS, required=True, help="Match the original environment"
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int, help="Match any original time-limit override")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, help="New video run directory")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
    with TemporaryDirectory() as temporary:
        with gym.wrappers.RecordVideo(
            make_environment(args.env, max_steps=args.max_steps, render_mode="rgb_array"),
            str(Path(temporary) / "videos"),
            episode_trigger=lambda _: True,
            disable_logger=True,
        ) as environment:
            try:
                await record_best(
                    args.run,
                    environment,
                    top=args.top,
                    seeds=args.seeds,
                    output=args.output,
                    executor=Executor(concurrency=args.concurrency),
                )
            except BlockingIOError:
                parser.error("The source run is open in another process; close the search first")


if __name__ == "__main__":
    asyncio.run(main())
