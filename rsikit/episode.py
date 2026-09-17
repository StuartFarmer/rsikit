"""One Gymnasium episode with explicit evidence and execution boundaries."""

import math
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from .policy import Policy


@dataclass(frozen=True)
class Transition:
    observation: Any
    action: Any
    next_observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class Episode:
    """Partial returns from failed episodes are evidence, not comparable fitness."""

    env_seed: int
    policy_seed: int
    instructions: str = ""
    initial_observation: Any = None
    reset_info: dict = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    status: str = "incomplete"
    failure: str | None = None
    attempted_action: Any = None
    cleanup_errors: list[str] = field(default_factory=list)

    @property
    def return_(self) -> float:
        return math.fsum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        return len(self.transitions)


class EpisodeError(RuntimeError):
    """Environment/infrastructure failure with the evidence collected so far."""

    def __init__(self, message: str, episode: Episode):
        super().__init__(message)
        self.episode = episode


class InfrastructureError(RuntimeError):
    """An execution backend failed independently of a policy's task outcome."""


class PolicyError(RuntimeError):
    """Isolated policy construction, reset, action, or cleanup failed."""


class PolicyTimeout(PolicyError):
    """The isolated policy exceeded an execution deadline."""


def _failed(episode: Episode, error: Exception) -> None:
    episode.status = "timeout" if isinstance(error, PolicyTimeout) else "policy_error"
    episode.failure = f"{type(error).__name__}: {error}"
    cleanup_error = getattr(error, "cleanup_error", None)
    if cleanup_error is not None:
        episode.cleanup_errors.append(f"Backend cleanup: {cleanup_error}")


async def _rollout(env: gym.Env, policy: Policy, episode: Episode) -> None:
    observation, info = env.reset(seed=episode.env_seed)
    if not env.observation_space.contains(observation):
        raise ValueError("Environment reset returned an observation outside observation_space")
    episode.initial_observation, episode.reset_info = deepcopy((observation, info))
    try:
        await policy.reset(seed=episode.policy_seed)
    except InfrastructureError:
        raise
    except Exception as exc:
        _failed(episode, exc)
        return

    while True:
        before = deepcopy(observation)
        try:
            action = await policy.act(deepcopy(observation))
        except InfrastructureError:
            raise
        except Exception as exc:
            _failed(episode, exc)
            return
        episode.attempted_action = deepcopy(action)
        try:
            valid = env.action_space.contains(action)
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            episode.status, episode.failure = "invalid_action", "Action outside action_space"
            return
        observation, reward, terminated, truncated, info = env.step(action)
        if not env.observation_space.contains(observation):
            raise ValueError("Environment step returned an observation outside observation_space")
        if not math.isfinite(reward):
            raise ValueError("Environment returned a nonfinite reward")
        episode.transitions.append(
            Transition(
                before,
                episode.attempted_action,
                deepcopy(observation),
                float(reward),
                bool(terminated),
                bool(truncated),
                deepcopy(info),
            )
        )
        episode.attempted_action = None
        if terminated or truncated:
            episode.status = "completed"
            return


async def run_episode(
    make_env: Callable[[], gym.Env],
    make_policy: Callable[..., Policy],
    *,
    env_seed: int,
    policy_seed: int,
    max_steps: int,
    instructions: str | None = None,
) -> Episode:
    """Run trusted classes locally, closing fresh instances on every exit.

    Generated source belongs in run_program, which supplies an isolated policy.
    Local async policies can block Python; hard execution limits need isolation.
    Instructions override env.unwrapped.instructions, including an empty string.
    """
    episode = Episode(env_seed, policy_seed)
    env, policy = None, None
    try:
        try:
            env = make_env()
        except Exception as exc:
            raise EpisodeError(f"Cannot create environment: {exc}", episode) from exc
        if instructions is None:
            try:
                instructions = env.unwrapped.instructions
            except AttributeError as exc:
                raise ValueError("Provide instructions= or define env.instructions") from exc
        episode.instructions = instructions
        try:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
            observation_space, action_space = deepcopy((env.observation_space, env.action_space))
            try:
                policy = make_policy(observation_space, action_space, instructions=instructions)
            except InfrastructureError:
                raise
            except Exception as exc:
                _failed(episode, exc)
                return episode
            await _rollout(env, policy, episode)
        except Exception as exc:
            raise EpisodeError(f"Episode evaluation failed: {exc}", episode) from exc
        return episode
    finally:
        # Cleanup never replaces the original exception or recorded policy failure.
        primary = sys.exc_info()[1]
        cleanup_failure = None
        try:
            if policy is not None:
                try:
                    await policy.close()
                except Exception as exc:
                    episode.cleanup_errors.append(f"Policy cleanup: {type(exc).__name__}: {exc}")
                    if primary is None and episode.status == "completed":
                        if isinstance(exc, InfrastructureError):
                            cleanup_failure = exc
                        else:
                            _failed(episode, exc)
        finally:
            # Cancellation during policy.close must still close the environment.
            closing_exception = sys.exc_info()[1]
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    episode.cleanup_errors.append(
                        f"Environment cleanup: {type(exc).__name__}: {exc}"
                    )
                    cleanup_failure = cleanup_failure or exc
            if (
                cleanup_failure is not None
                and primary is None
                and closing_exception is None
                and episode.status == "completed"
            ):
                episode.status = "incomplete"
                raise EpisodeError(
                    f"Episode cleanup failed: {cleanup_failure}", episode
                ) from cleanup_failure
