"""Evaluate a policy in a Gymnasium environment and collect its episode."""

import logging
import math
import sys
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np

from .episode import Episode
from .policy import Policy


@dataclass(frozen=True)
class Measurement:
    """Measured evidence; optimizers own its interpretation and selection."""

    scores: dict[int, float] = field(default_factory=dict)
    feedback: str = ""
    failure: str | None = None
    accepted: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("scores", "metrics", "features"):
            values = getattr(self, name)
            if not isinstance(values, dict) or any(
                (
                    type(key) is not int
                    if name == "scores"
                    else not isinstance(key, str) or not key.strip()
                )
                or isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                for key, value in values.items()
            ):
                raise ValueError(f"Invalid {name}: expected named finite numeric measurements")
            object.__setattr__(self, name, dict(values))
        if not isinstance(self.feedback, str) or type(self.accepted) is not bool:
            raise ValueError("Measurement feedback must be text and accepted must be boolean")
        if self.failure is not None:
            if not isinstance(self.failure, str) or not self.failure.strip():
                raise ValueError("Measurement failure must be nonempty text")
            object.__setattr__(self, "accepted", False)
        if self.accepted and not (self.scores or self.metrics):
            raise ValueError("Accepted measurements require scores or metrics")


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
def _policy_boundary(convert):
    """Normalize generated-code calls; direct Evaluator callers keep native errors."""
    try:
        yield
    except BaseException as exc:
        if not convert or isinstance(exc, PolicyError):
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
        _policy_errors: bool = False,
    ):
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer or None")
        self.environment = environment
        self.policy = policy
        self.max_steps = max_steps
        self._policy_errors = _policy_errors

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
            policy_observation = deepcopy(observation)
            with _policy_boundary(self._policy_errors):
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


async def _run_episode(
    make_env: str | Callable[[], gym.Env],
    make_policy: Callable[..., Policy],
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
    _policy_errors: bool = False,
) -> Episode:
    """Prepare and clean up instances for one episode.

    Accept an environment ID or a factory returning a fresh environment. Existing
    Gymnasium time limits apply unless max_steps supplies an additional cap.
    Instructions are optional. Exceptions propagate after resources are closed.
    """
    env = gym.make(make_env) if isinstance(make_env, str) else make_env()
    policy = None
    try:
        if max_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
        if instructions is None:
            instructions = (
                env.get_wrapper_attr("instructions") if env.has_wrapper_attr("instructions") else ""
            )
        with _policy_boundary(_policy_errors):
            policy = make_policy(
                deepcopy(env.observation_space),
                deepcopy(env.action_space),
                instructions=instructions,
            )
        observation, info = env.reset(seed=env_seed)
        with _policy_boundary(_policy_errors):
            await policy.reset(seed=policy_seed)
        episode = await Evaluator(env, policy, _policy_errors=_policy_errors).run(
            observation, info=info
        )
        return episode
    finally:
        primary = sys.exc_info()[1]
        try:
            try:
                if policy is not None:
                    with _policy_boundary(_policy_errors):
                        await policy.close()
            finally:
                env.close()
        except BaseException:
            if primary is None:
                raise
            # Keep the original failure/cancellation; expose secondary cleanup errors.
            logging.getLogger(__name__).exception("Cleanup failed while handling an episode error")
