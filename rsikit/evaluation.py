"""Evaluate a policy in a Gymnasium environment and collect its episode."""

import asyncio
import logging
import sys
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

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


class Evaluator:
    """Collect one episode from existing, already-reset environment and policy.

    The caller owns creation, seeding, reset, and cleanup, including on failure.
    This class neither isolates code nor prevents reuse of stateful instances.
    """

    def __init__(
        self,
        environment: gym.Env,
        policy: Policy,
        *,
        max_steps: int | None = None,
    ):
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer or None")
        self.environment = environment
        self.policy = policy
        self.max_steps = max_steps

    async def run(self, observation: Any, *, info: dict[str, Any] | None = None) -> Episode:
        """Start from the supplied reset result and stop at termination/truncation.

        Observations, applied actions, and infos are deep-copied for later analysis.
        max_steps caps this rollout without resetting or wrapping the environment.
        """
        env = self.environment
        episode = Episode(
            observations=[deepcopy(observation)],
            infos=[deepcopy(info) if info is not None else {}],
        )
        try:
            while True:
                policy_observation = deepcopy(observation)
                with _policy_boundary():
                    action = await self.policy.act(policy_observation)
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
                    return episode
        except PolicyError as exc:
            episode.error = "".join(traceback.format_exception(exc))
            return episode


async def _run_episode(
    make_env: str | Callable[[], gym.Env],
    make_policy: Callable[..., Policy],
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
) -> Episode:
    """Prepare and clean up instances for one episode.

    Accept an environment ID or a factory returning a fresh environment. Existing
    Gymnasium time limits apply unless max_steps supplies an additional cap.
    Instructions are optional. Candidate errors return partial episodes; other
    exceptions propagate after resources are closed.
    """
    env = gym.make(make_env) if isinstance(make_env, str) else make_env()
    policy = None
    episode = Episode()
    try:
        if max_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
        if instructions is None:
            instructions = (
                env.get_wrapper_attr("instructions") if env.has_wrapper_attr("instructions") else ""
            )
        with _policy_boundary():
            policy = make_policy(
                deepcopy(env.observation_space),
                deepcopy(env.action_space),
                instructions=instructions,
            )
        observation, info = env.reset(seed=env_seed)
        episode.observations = [deepcopy(observation)]
        episode.infos = [deepcopy(info)]
        with _policy_boundary():
            await policy.reset(seed=policy_seed)
        episode = await Evaluator(env, policy).run(observation, info=info)
        return episode
    except PolicyError as exc:
        episode.error = "".join(traceback.format_exception(exc))
        return episode
    finally:
        primary = sys.exc_info()[1]
        try:
            try:
                if policy is not None:
                    try:
                        with _policy_boundary():
                            await policy.close()
                    except PolicyError as exc:
                        if primary is None and episode.error is None:
                            episode.error = "".join(traceback.format_exception(exc))
                        else:
                            logging.getLogger(__name__).exception(
                                "Cleanup failed while handling an episode error"
                            )
            finally:
                env.close()
        except BaseException:
            if primary is None and episode.error is None:
                raise
            # Keep the original failure/cancellation; expose secondary cleanup errors.
            logging.getLogger(__name__).exception("Cleanup failed while handling an episode error")
