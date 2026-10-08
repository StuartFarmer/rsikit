"""Compose a deterministic optimizer with search, Run and Executor."""

import argparse
import asyncio
from functools import partial
from pathlib import Path
from statistics import fmean

import gymnasium as gym

from rsikit import Executor, PolicyDefinition, Run, search
from rsikit.evaluation import PolicyError, episode_error, episode_scores
from rsikit.optimization import validate_results


def candidate(gain):
    return PolicyDefinition.from_text(
        "from rsikit import Policy\n\n"
        "class Solution(Policy):\n"
        "    async def reset(self, *, seed=None):\n"
        "        await super().reset(seed=seed)\n"
        "        self.previous_angle = 0.0\n\n"
        "    async def act(self, observation):\n"
        "        angle = float(observation[2])\n"
        f"        signal = angle + {gain!r} * (angle - self.previous_angle)\n"
        "        self.previous_angle = angle\n"
        "        return int(signal > 0)\n",
        name=f"Difference controller {gain}",
    )


class GridSearch:
    """One complete round of three candidates; highest mean fitness wins."""

    def __init__(self):
        self.candidates = [candidate(gain) for gain in (0.0, 10.0, 25.0)]
        self.seeds = (0, 1)
        self.pending = False
        self.done = False
        self.best = None
        self.best_score = float("-inf")

    async def propose(self):
        if self.pending:
            raise RuntimeError("Update the outstanding round before proposing again")
        if self.done:
            return []
        self.pending = True
        return self.candidates

    def update(self, results):
        validate_results(
            results,
            [p.id for p in self.candidates] if self.pending else [],
            seed_panel=set(self.seeds),
        )
        # Validate every fitness before changing state. Failed panels cannot win.
        scores = {id: episode_scores(episodes) for id, episodes in results.items()}
        for policy in self.candidates:
            error = episode_error(results[policy.id])
            if error or not scores[policy.id]:
                print(f"rejected {policy.name}: {error or 'empty feedback'}")
                continue
            score = fmean(scores[policy.id].values())
            if score > self.best_score:
                self.best, self.best_score = policy, score
        self.pending, self.done = False, True


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory (must not exist)")
    args = parser.parse_args(argv)
    optimizer = GridSearch()
    with gym.make("CartPole-v1", max_episode_steps=100) as environment:
        async with (
            Run.create(name="cartpole-grid", path=args.output) as run,
            Executor(concurrency=1, episode_timeout=15.0) as executor,
        ):
            evaluate = partial(run.evaluate, environment=environment, executor=executor)
            winner = await search(optimizer, partial(evaluate, seeds=optimizer.seeds))
            if winner is None:
                raise RuntimeError("All candidates failed; inspect the saved episodes")
            winner.to_file(run.path / "best.py")
            loaded = PolicyDefinition.from_file(run.path / "best.py")
            heldout = (await evaluate([loaded], seeds=(100, 101)))[loaded.id]
            if error := episode_error(heldout):
                raise PolicyError(error)
            print(f"training_mean={optimizer.best_score:.1f}")
            print(f"heldout_mean={fmean(episode_scores(heldout).values()):.1f}")
            print(f"winner={run.path / 'best.py'}")
            return loaded


if __name__ == "__main__":
    asyncio.run(main())
