"""Evaluate a policy in a Gymnasium environment and collect its episode."""

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


class Evaluator:
    """Collect one episode from existing, already-reset environment and policy.

    The caller owns creation, seeding, reset, and cleanup, including on failure.
    This class neither isolates code nor prevents reuse of stateful instances.
    """

    def __init__(self, environment: gym.Env, policy: Policy, *, max_steps: int | None = None):
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
        while True:
            action = await self.policy.act(deepcopy(observation))
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
