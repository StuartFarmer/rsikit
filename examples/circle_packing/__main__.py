"""Run a class-based packing program in the isolated policy worker."""

import argparse
import asyncio
from pathlib import Path

from rsikit import run_program
from rsikit.envs import CirclePackingEnv


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "program", nargs="?", type=Path, default=Path(__file__).with_name("initial.py")
    )
    args = parser.parse_args()
    episode = await run_program(
        args.program,
        CirclePackingEnv,
        env_seed=1,
        policy_seed=2,
        max_steps=1,
    )
    info = episode.infos[-1]
    print(info)


if __name__ == "__main__":
    asyncio.run(main())
