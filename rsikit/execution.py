"""Execute one policy evaluation; Run owns batching and score persistence."""

from typing import Protocol

import gymnasium as gym

from .episode import run_episode
from .sandbox import SandboxPolicy


class Executor(Protocol):
    """A serializable request in, one score out. Clean up before returning or raising."""

    async def evaluate(
        self,
        implementation: str,
        *,
        environment: str,
        seed: int,
        environment_kwargs: dict,
        max_steps: int | None,
        instructions: str | None,
        call_timeout: float,
        video_dir: str | None,
    ) -> float: ...


class DockerExecutor:
    """Run an agent in a fresh Docker sandbox, with its Gymnasium environment on the host."""

    def __init__(self, *, image: str = "rsikit-sandbox:local"):
        self.image = image

    async def evaluate(
        self,
        implementation: str,
        *,
        environment: str,
        seed: int,
        environment_kwargs: dict,
        max_steps: int | None,
        instructions: str | None,
        call_timeout: float,
        video_dir: str | None,
    ) -> float:
        def make_env():
            kwargs = dict(environment_kwargs)
            if video_dir is not None:
                kwargs["render_mode"] = "rgb_array"
            env = gym.make(environment, **kwargs)
            if video_dir is not None:
                try:
                    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
                except BaseException:
                    env.close()
                    raise
            return env

        def make_policy(observation_space, action_space, *, instructions):
            return SandboxPolicy(
                observation_space,
                action_space,
                instructions=instructions,
                source=implementation,
                image=self.image,
                call_timeout=call_timeout,
            )

        *_, info = await run_episode(
            make_env,
            make_policy,
            env_seed=seed,
            policy_seed=seed,
            max_steps=max_steps,
            instructions=instructions,
        )
        return float(info["episode"]["r"])
