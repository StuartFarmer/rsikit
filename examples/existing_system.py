"""Run a tiny EliteSearch on CartPole, then reload and evaluate its winner."""

import argparse
import asyncio
import json
import os
from functools import partial
from pathlib import Path
from statistics import fmean

import gymnasium as gym
from slick import prompts

from research import elitesearch
from research.elitesearch import Config, EliteSearch
from rsikit import Executor, PolicyDefinition, Run, search
from rsikit.evaluation import PolicyError, episode_error, episode_scores


class OfflineProvider:
    """Two fixed responses exercise the real optimizer without an API call."""

    def __init__(self):
        self.gains = iter((0.0, 0.5))

    async def acall(self, context):
        gain = next(self.gains)
        return json.dumps(
            {
                "name": f"Angle controller {gain}",
                "description": "Push toward the pole's angle and angular velocity.",
                "implementation": (
                    "from rsikit import Policy\n\n"
                    "class Solution(Policy):\n"
                    "    async def act(self, observation):\n"
                    f"        return int(observation[2] + {gain} * observation[3] > 0)\n"
                ),
            }
        ), []


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory (must not exist)")
    parser.add_argument("--model", help="OpenRouter model ID; omitted means offline")
    args = parser.parse_args(argv)
    if args.model:
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("Set OPENROUTER_API_KEY to use --model")
        from slick.providers import OpenRouterAPI

        provider = OpenRouterAPI(model=args.model, max_output_tokens=2048, timeout=60.0)
    else:
        provider = OfflineProvider()

    prompts.TEMPLATE_ROOT = Path(elitesearch.__file__).parent / "prompts"
    with gym.make("CartPole-v1", max_episode_steps=100) as environment:
        async with (
            Run.create(name="cartpole-elitesearch", path=args.output) as run,
            Executor(concurrency=1, episode_timeout=15.0) as executor,
        ):
            evaluate = partial(run.evaluate, environment=environment, executor=executor)
            optimizer = EliteSearch(
                "Maximize reward in CartPole-v1, capped at 100 steps. Observations are "
                "[cart position, cart velocity, pole angle, pole angular velocity]. "
                "Return integer action 0 (push left) or 1 (push right).",
                provider,
                config=Config(
                    generations=1,
                    population_size=2,
                    elite_size=1,
                    generation_concurrency=1,
                    generation_timeout=60.0,
                    max_repairs=0,
                ),
            )
            winner = await search(optimizer, partial(evaluate, seeds=(0, 1)))
            run.save(*optimizer.records())
            for row in optimizer.organisms:
                if row.error:
                    print(f"candidate {row.id}: {row.error}")
            if winner is None:
                raise RuntimeError("No successful candidate; inspect run.sqlite for errors")
            winner.to_file(run.path / "best.py")
            loaded = PolicyDefinition.from_file(run.path / "best.py")
            heldout = (await evaluate([loaded], seeds=(100, 101)))[loaded.id]
            if error := episode_error(heldout):
                raise PolicyError(error)
            print(f"training_mean={optimizer.elites[0].score:.1f}")
            print(f"heldout_mean={fmean(episode_scores(heldout).values()):.1f}")
            print(f"winner={run.path / 'best.py'}")
            return loaded


if __name__ == "__main__":
    asyncio.run(main())
