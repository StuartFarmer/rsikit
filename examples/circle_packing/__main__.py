"""Run a class-based packing program in the isolated policy worker."""

import argparse
import asyncio
from pathlib import Path

from rsikit import Executor, PolicyDefinition
from rsikit.envs import CirclePackingEnv
from rsikit.evaluation import PolicyError


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "program", nargs="?", type=Path, default=Path(__file__).with_name("initial.py")
    )
    args = parser.parse_args()
    policy = PolicyDefinition.from_file(args.program)
    with CirclePackingEnv() as environment:
        async with Executor() as executor:
            async for _, _, episode in executor.evaluate(
                [(policy.id, policy.source, 1)], environment, policy_seed=2, max_steps=1
            ):
                if episode.error is not None:
                    raise PolicyError(episode.error)
                print(episode.infos[-1])


if __name__ == "__main__":
    asyncio.run(main())
