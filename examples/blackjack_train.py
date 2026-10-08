"""Train a Blackjack test run and export leaders after every scored generation."""

import asyncio
import sys

from examples.elitesearch import main as elite_main


async def main(argv=None):
    await elite_main(
        [
            "--env",
            "Blackjack",
            "--elites",
            "4",
            "--population",
            "8",
            "--generations",
            "3",
            "--generation-concurrency",
            "4",
            "--concurrency",
            "4",
            "--shoes-per-seed",
            "24",
            "--seeds",
            *map(str, range(10)),
            "--heldout-seeds",
            *map(str, range(1000, 1010)),
            "--video-top",
            "4",
            *(sys.argv[1:] if argv is None else argv),
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
