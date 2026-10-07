"""Gymnasium task definitions; the environment instances keep their native API."""

from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import gymnasium as gym
from pydantic import Field

from research.experiment import EnvironmentDefinition, Options, save_json


class BlackjackOptions(Options):
    shoes_per_episode: int = Field(default=24, ge=1)


class PriceSeriesOptions(Options):
    data_path: Path
    price_column: str = Field(default="price", min_length=1)
    time_column: str | None = None
    asset_name: str = Field(default="asset", min_length=1)
    fee_rate: float = Field(default=0.001, ge=0, lt=1)
    initial_cash: float = Field(default=10_000, gt=0)
    start_time: str | None = None
    end_time: str | None = None


def arguments(parser, *, name):
    if name == "Blackjack":
        parser.add_argument("--env-shoes-per-episode", dest="shoes_per_episode", type=int)
    elif name == "PriceSeries":
        for field in PriceSeriesOptions.model_fields:
            parser.add_argument(
                "--env-" + field.replace("_", "-"),
                dest=field,
                type=float if field in ("fee_rate", "initial_cash") else str,
            )


def definition(name):
    return EnvironmentDefinition(
        Options={"Blackjack": BlackjackOptions, "PriceSeries": PriceSeriesOptions}.get(
            name, Options
        ),
        add_arguments=partial(arguments, name=name),
        describe=partial(describe, name),
        open_evaluator=partial(open_evaluator, name),
        evaluation_defaults={"seeds": [0]} if name == "PriceSeries" else {},
        supported_evaluation_fields=frozenset({"score_key"})
        if name == "PriceSeries"
        else frozenset({"max_steps", "score_key"}),
    )


def make_environment(name, options, evaluation):
    from rsikit.envs import PriceSeriesEnv
    from rsikit.envs.tasks import TASKS
    from rsikit.envs.tasks import make_environment as preset

    if name == "PriceSeries":
        return PriceSeriesEnv(**options.model_dump())
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
        if name == "PriceSeries":
            save_json(run.path / "dataset.json", env.dataset)
        async with Executor(
            concurrency=evaluation.workers, episode_timeout=evaluation.timeout_per_seed
        ) as executor:
            rollouts = Rollouts(env, executor, run)

            async def evaluate(policies, seeds):
                return await measure_rewards(rollouts, policies, seeds)

            yield evaluate
