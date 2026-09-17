"""Container-only environment evaluation; the generated agent runs in a separate process."""

import asyncio
import base64
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
import gymnasium as gym

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout, run_episode
from rsikit.sandbox import SandboxPolicy
from rsikit.sandbox.codec import MAX_MESSAGE, loads


class ProcessPolicy(SandboxPolicy):
    async def reset(self, *, seed=None):
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-u",
            "-c",
            "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.worker import main; main()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=MAX_MESSAGE + 1,
        )
        if loads(await asyncio.wait_for(self.process.stdout.readline(), 60)) != {"ready": True}:
            raise InfrastructureError("Policy worker did not start")
        self.ready = True
        await self._request(
            {
                "command": "start",
                "source": self._implementation,
                "observation_space": self.space_definitions[0],
                "action_space": self.space_definitions[1],
                "instructions": self.instructions,
                "seed": seed,
            }
        )

    async def _destroy(self):
        self.ready = False
        if self.process is None:
            return
        process, self.process = self.process, None
        process.stdin.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def evaluate(request):
    if request["python"] != list(sys.version_info[:2]):
        raise InfrastructureError(
            "Host and sandbox Python minor versions must match; rebuild with --build-arg PYTHON_VERSION=X.Y"
        )
    with TemporaryDirectory() as directory:
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
            return ProcessPolicy(
                observation_space,
                action_space,
                instructions=instructions,
                source=request["implementation"],
                call_timeout=request["call_timeout"],
            )

        *_, info = await run_episode(
            lambda: env, make_policy, env_seed=request["seed"], policy_seed=request["seed"]
        )
        artifacts = dict(info.get("artifacts", {}))
        for path in Path(directory).rglob("*"):
            if path.is_file() and not path.is_symlink():
                artifacts[str(path.relative_to(directory))] = path.read_bytes()
        return {
            "score": float(info["episode"]["r"]),
            "artifacts": {
                name: base64.b64encode(data).decode() for name, data in artifacts.items()
            },
        }


def main():
    # Environment prints and native library output must not corrupt the result channel.
    with os.fdopen(os.dup(1), "w") as output:
        os.dup2(2, 1)
        try:
            raw = sys.stdin.buffer.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("Environment request exceeds 64 MiB")
            result = asyncio.run(evaluate(json.loads(raw)))
            data = json.dumps(result, allow_nan=False)
            if len(data.encode()) > 64 * 1024 * 1024:
                raise ValueError("Evaluation artifacts exceed 64 MiB")
        except Exception as exc:
            kind = (
                "timeout"
                if isinstance(exc, PolicyTimeout)
                else "policy"
                if isinstance(exc, PolicyError)
                else "infrastructure"
            )
            data = json.dumps({"kind": kind, "error": f"{type(exc).__name__}: {exc}"})
        output.write(data + "\n")
        output.flush()
