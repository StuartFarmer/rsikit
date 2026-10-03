"""Gymnasium task definitions; the environment instances keep their native API."""

from contextlib import asynccontextmanager
from functools import partial

import gymnasium as gym
from pydantic import Field

from research.experiment import EnvironmentDefinition, Options


class BlackjackOptions(Options):
    shoes_per_episode: int = Field(default=24, ge=1)


def arguments(parser, *, blackjack=False):
    if blackjack:
        parser.add_argument("--env-shoes-per-episode", dest="shoes_per_episode", type=int)


def definition(name):
    return EnvironmentDefinition(
        Options=BlackjackOptions if name == "Blackjack" else Options,
        add_arguments=partial(arguments, blackjack=name == "Blackjack"),
        describe=partial(describe, name),
        open_evaluator=partial(open_evaluator, name),
    )


def make_environment(name, options, evaluation):
    from rsikit.envs.tasks import TASKS
    from rsikit.envs.tasks import make_environment as preset

    if name in TASKS:
        return preset(name, max_steps=evaluation.max_steps, **options.model_dump())
    kwargs = {} if evaluation.max_steps is None else {"max_episode_steps": evaluation.max_steps}
    env = gym.Wrapper(gym.make(name, **kwargs))
    env.instructions = f"{name}: observation {env.observation_space}; action {env.action_space}. Maximize episode reward."
    return env


def describe(name, options, evaluation):
    with make_environment(name, options, evaluation) as env:
        return (
            env.instructions + "\nScore: cumulative episode reward, averaged across seed episodes."
        )


@asynccontextmanager
async def open_evaluator(name, *, options, evaluation, run):
    from research.rewards import measure_rewards
    from research.rollouts import Rollouts
    from rsikit import Executor

    with make_environment(name, options, evaluation) as env:
        async with Executor(
            concurrency=evaluation.workers, episode_timeout=evaluation.timeout_per_seed
        ) as executor:
            rollouts = Rollouts(env, executor, run)

            async def evaluate(policies, seeds):
                return await measure_rewards(rollouts, policies, seeds)

            yield evaluate
