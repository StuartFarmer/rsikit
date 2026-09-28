"""Run complete generated-policy episodes inside Docker."""

import logging
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import cloudpickle
import gymnasium as gym

from rsikit.episode import Episode
from rsikit.evaluation import Evaluator, InfrastructureError, PolicyError
from rsikit.policy import Policy

from .codec import MAX_SOURCE


async def run_program(
    program: Path,
    make_env,
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
    episode_timeout: float = 60.0,
) -> Episode:
    """Run a complete episode in Docker, including environment reset and scoring."""
    from .docker import DockerSandbox

    try:
        with Path(program).open(encoding="utf-8") as stream:
            source = stream.read(MAX_SOURCE + 1)
    except (OSError, UnicodeError) as exc:
        raise InfrastructureError(f"Cannot read policy source: {exc}") from exc
    if len(source.encode()) > MAX_SOURCE:
        raise PolicyError("Source exceeds 64 KiB")
    sandbox = DockerSandbox(episode_timeout=episode_timeout)
    env = gym.make(make_env) if isinstance(make_env, str) else make_env()
    try:
        definition = cloudpickle.dumps(env)
        await sandbox.start(1)
        return await sandbox.evaluate(
            source,
            definition,
            env_seed,
            policy_seed=policy_seed,
            max_steps=max_steps,
            instructions=instructions,
        )
    finally:
        primary = sys.exc_info()[1]
        try:
            try:
                await sandbox.close()
            finally:
                env.close()
        except BaseException:
            if primary is None:
                raise
            logging.getLogger(__name__).exception("Sandbox cleanup failed during evaluation")


async def _run_episode(
    make_env: str | Callable[[], gym.Env],
    make_policy: Callable[..., Policy],
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
) -> Episode:
    """Prepare and clean up instances on behalf of sandbox execution.

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
        policy = make_policy(
            deepcopy(env.observation_space),
            deepcopy(env.action_space),
            instructions=instructions,
        )
        observation, info = env.reset(seed=env_seed)
        await policy.reset(seed=policy_seed)
        episode = await Evaluator(env, policy).run(observation, info=info)
        return episode
    finally:
        primary = sys.exc_info()[1]
        try:
            try:
                if policy is not None:
                    await policy.close()
            finally:
                env.close()
        except BaseException:
            if primary is None:
                raise
            # Keep the original failure/cancellation; expose secondary cleanup errors.
            logging.getLogger(__name__).exception("Cleanup failed while handling an episode error")
