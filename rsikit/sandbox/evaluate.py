"""Container-only whole-episode execution; no per-action communication."""

import asyncio
import base64
import json
import os
import resource
import sys
from pathlib import Path

import cloudpickle
import gymnasium as gym

from rsikit.evaluation import InfrastructureError, PolicyError, PolicyTimeout
from rsikit.policy import Policy
from rsikit.sandbox import _run_episode
from rsikit.sandbox.codec import encode_episode
from rsikit.sandbox.worker import block_connections, load_policy


class DirectPolicy(Policy):
    def __init__(self, observation_space, action_space, *, instructions, source):
        super().__init__(observation_space, action_space, instructions=instructions)
        try:
            self.policy = load_policy(source, observation_space, action_space, instructions)
        except BaseException as exc:
            raise PolicyError(f"{type(exc).__name__}: {str(exc)[:2000]}") from exc

    async def _call(self, method, *args, **kwargs):
        try:
            return await getattr(self.policy, method)(*args, **kwargs)
        except BaseException as exc:
            raise PolicyError(f"Policy {method}: {type(exc).__name__}: {str(exc)[:2000]}") from exc

    async def reset(self, *, seed=None):
        await self._call("reset", seed=seed)

    async def act(self, observation):
        return await self._call("act", observation)

    async def close(self):
        await self._call("close")


async def evaluate(request, *, directory):
    if request["python"] != list(sys.version_info[:2]):
        raise InfrastructureError(
            "Host and sandbox Python minor versions must match; rebuild with --build-arg PYTHON_VERSION=X.Y"
        )
    os.chdir(directory)
    env = cloudpickle.loads(base64.b64decode(request["environment"], validate=True))
    # Only relocate output paths into this worker's filesystem; recording options stay on Gymnasium.
    wrapper, index = env, 0
    while isinstance(wrapper, gym.Wrapper):
        if isinstance(wrapper, gym.wrappers.RecordVideo):
            wrapper.video_folder = str(Path(directory) / "videos" / str(index))
            Path(wrapper.video_folder).mkdir(parents=True)
            index += 1
        wrapper = wrapper.env

    def make_policy(observation_space, action_space, *, instructions):
        return DirectPolicy(
            observation_space,
            action_space,
            instructions=instructions,
            source=request["implementation"],
        )

    episode = await _run_episode(
        lambda: env,
        make_policy,
        env_seed=request["seed"],
        policy_seed=request.get("policy_seed", request["seed"]),
        max_steps=request.get("max_steps"),
        instructions=request.get("instructions"),
    )
    artifacts = dict(episode.infos[-1].get("artifacts", {}))
    for path in Path(directory).rglob("*"):
        if path.is_file() and not path.is_symlink():
            artifacts[str(path.relative_to(directory))] = path.read_bytes()
    episode.artifacts.update(artifacts)
    return {"episode": encode_episode(episode)}


def error_result(exc):
    kind = (
        "timeout"
        if isinstance(exc, PolicyTimeout)
        else "policy"
        if isinstance(exc, PolicyError)
        else "infrastructure"
    )
    return {"kind": kind, "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}


def result_bytes(request, **kwargs):
    try:
        result = asyncio.run(evaluate(request, **kwargs))
        data = json.dumps(result, allow_nan=False).encode()
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Evaluation artifacts exceed 64 MiB")
        return data + b"\n"
    except Exception as exc:
        return json.dumps(error_result(exc)).encode() + b"\n"


def run_evaluation(request, output, directory):
    # Forkserver children do not inherit the service interpreter's -u flag.
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
    # Give the supervisor a process group to reap, including video encoders or
    # other subprocesses. Docker still bounds the whole container's PIDs/CPU/RAM.
    os.setsid()
    block_connections(keep_process_group=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with output:
        result = result_bytes(request, directory=directory)
        sys.stdout.flush()
        sys.stderr.flush()
        output.sendall(result)
