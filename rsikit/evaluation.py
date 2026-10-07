"""Evaluate a policy in a Gymnasium environment and collect its episode."""

import asyncio
import traceback
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter

import gymnasium as gym
import numpy as np

from .episode import Episode
from .policy import Policy


class InfrastructureError(RuntimeError):
    """An isolated execution backend failed."""


class PolicyError(RuntimeError):
    """A generated policy failed; batch failures map policy IDs to diagnostics."""

    def __init__(self, message: str):
        super().__init__(message)
        self.failures: dict[str, str] = {}


class PolicyTimeout(PolicyError):
    """An isolated policy exceeded its execution deadline."""


@contextmanager
def _policy_boundary():
    """Distinguish candidate errors from cancellation and infrastructure failures."""
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, (PolicyError, InfrastructureError, asyncio.CancelledError)):
            raise
        raise PolicyError(f"{type(exc).__name__}: {str(exc)[:2000]}") from exc


def _validate_seed(seed):
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")


class Evaluator:
    """Reset and evaluate caller-owned policy and environment instances.

    Each call starts a new episode. The caller owns construction and cleanup,
    including on failure or cancellation. Process isolation belongs to Executor.
    """

    def __init__(self, *, max_steps: int | None = None):
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer or None")
        self.max_steps = max_steps

    async def evaluate(
        self,
        policy: Policy,
        environment: gym.Env,
        seed: int | None = None,
    ) -> Episode:
        """Reset both objects, then collect transitions until termination or truncation.

        Both objects reset once with the same seed.
        Observations, actions, and infos are copied to preserve the trajectory.
        Successful episodes include reward/length/time statistics in their final info.
        """
        _validate_seed(seed)
        env = environment
        observation, info = env.reset(seed=seed)
        started = perf_counter()
        episode = Episode(observations=[deepcopy(observation)], infos=[deepcopy(info)])
        try:
            with _policy_boundary():
                await policy.reset(seed=seed)
            while True:
                policy_observation = deepcopy(observation)
                with _policy_boundary():
                    action = await policy.act(policy_observation)
                try:
                    valid = env.action_space.contains(action)
                except (ValueError, TypeError, OverflowError):
                    valid = False
                if not valid:
                    raise PolicyError("Action outside action_space")
                # Discrete.contains accepts scalar arrays, but toy-text uses dict keys.
                if isinstance(env.action_space, gym.spaces.Discrete):
                    action = int(action)
                elif isinstance(env.action_space, gym.spaces.Box):
                    # Box.contains validates lists via a temporary array; pass that representation.
                    action = np.asarray(action, dtype=env.action_space.dtype)
                applied_action = deepcopy(action)
                try:
                    observation, reward, terminated, truncated, info = env.step(action)
                except gym.error.InvalidAction as exc:
                    raise PolicyError(str(exc)) from exc
                if self.max_steps is not None and len(episode) + 1 >= self.max_steps:
                    truncated = True
                episode.observations.append(deepcopy(observation))
                episode.actions.append(applied_action)
                episode.rewards.append(float(reward))
                episode.terminations.append(bool(terminated))
                episode.truncations.append(bool(truncated))
                episode.infos.append(deepcopy(info))
                if terminated or truncated:
                    episode.infos[-1].setdefault(
                        "episode",
                        {
                            "r": episode.total_reward,
                            "l": len(episode),
                            "t": round(perf_counter() - started, 6),
                        },
                    )
                    return episode
        except PolicyError as exc:
            episode.error = "".join(traceback.format_exception(exc))
            return episode


async def evaluate(
    policy: Policy,
    environment: gym.Env,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> Episode:
    """Start one episode on caller-owned instances, resetting both with seed."""
    return await Evaluator(max_steps=max_steps).evaluate(policy, environment, seed=seed)
