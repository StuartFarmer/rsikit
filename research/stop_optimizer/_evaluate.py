"""Trusted task-container entry point; policies run in ordinary episode processes."""

import asyncio
import json
import os
import sys
from pathlib import Path

from rsikit import Executor, PolicyDefinition, Run
from rsikit.envs.tasks import make_environment
from rsikit.evaluation import episode_error, episode_scores
from rsikit.policy import InvalidPolicy


async def evaluate(request):
    try:
        policy = PolicyDefinition.from_text(request["source"])
        policy.validate()
    except (InvalidPolicy, ValueError) as exc:
        return dict(scores={}, failure=str(exc), accepted=False, executed_episodes=0)
    path = Path("/task/run")
    run = (
        Run.open(path)
        if (path / "run.sqlite").exists()
        else Run.create(name=request["environment"], path=path)
    )
    with make_environment(request["environment"], max_steps=request["max_steps"]) as env:
        async with Executor() as executor, run:
            missing = sum(
                (cached := run.load_episode(policy, seed)) is None or cached.error is not None
                for seed in set(request["seeds"])
            )
            result = (
                await run.evaluate(
                    [policy], environment=env, executor=executor, seeds=request["seeds"]
                )
            )[policy.id]
            return dict(
                scores=episode_scores(result),
                failure=episode_error(result),
                accepted=episode_error(result) is None,
                policy_id=policy.id,
                executed_episodes=missing,
                instructions=env.instructions,
            )


def main():
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    request = json.loads(sys.stdin.buffer.readline(1024 * 1024 + 1))
    # Infrastructure errors exit nonzero; only candidate failures become measurements.
    result = asyncio.run(evaluate(request))
    print(json.dumps(dict(result=result), allow_nan=False), file=protocol, flush=True)


if __name__ == "__main__":
    main()
