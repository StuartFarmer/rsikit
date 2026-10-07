"""Docker transport for untrusted controllers and policies; JSON only, no host mounts."""

import asyncio
import json
import math
from uuid import uuid4

import numpy as np

from rsikit.evaluation import PolicyError

MAX_MESSAGE = 2 * 1024 * 1024


def invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        invalid_constant(value)
    return number


class Worker:
    def __init__(self, image, *, max_message=MAX_MESSAGE):
        self.image = image
        self.max_message = max_message
        self.name = "rsikit-meta-" + uuid4().hex
        self.process = None

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--init",
            "--name",
            self.name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--cpus=2",
            "--memory=1g",
            "--memory-swap=1g",
            "--pids-limit=64",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "--workdir=/tmp",
            "--log-driver=none",
            self.image,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=self.max_message,
        )
        return self

    async def send(self, value):
        data = json.dumps(value, allow_nan=False).encode() + b"\n"
        if len(data) > self.max_message:
            raise PolicyError("Message exceeds worker transport limit")
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def receive(self):
        try:
            data = await self.process.stdout.readline()
            if not data or len(data) > self.max_message:
                raise ValueError("Worker exited or returned an oversized message")
            value = json.loads(data, parse_constant=invalid_constant, parse_float=finite_float)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            if "error" in value:
                raise ValueError(str(value["error"])[:8192])
            return value
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise PolicyError(str(exc)) from exc

    async def __aexit__(self, *exc):
        # Killing only `docker run` can leave the container running; remove it by unique name.
        cleanup = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            self.name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(cleanup.wait(), 15)
        finally:
            if cleanup.returncode is None:
                cleanup.kill()
                await cleanup.wait()
            if self.process.returncode is None:
                self.process.kill()
            await self.process.wait()


async def run_controller(source, context, handle, *, image, timeout):
    async with Worker(image) as worker:

        async def run():
            await worker.send(dict(mode="controller", source=source, context=context))
            # ponytail: serial pilot requests; add bounded concurrency after isolation timing.
            # Bound even controllers that endlessly request invalid/free operations.
            limit = 8 * context.get("budget", {}).get("evaluations", 50) + 32
            for _ in range(limit):
                request = await worker.receive()
                if len(json.dumps(request).encode()) > 65536 + 1024:
                    raise PolicyError("Controller request exceeds 65 KiB")
                if request.get("op") == "done":
                    return
                await worker.send(await handle(request))
            raise PolicyError(f"Controller exceeded {limit} protocol requests")

        await asyncio.wait_for(run(), timeout)


class RemotePolicy:
    """Drop-in trusted policy factory: only JSON actions cross back from generated code."""

    def __init__(self, source, observation_space, action_space, instructions, *, image):
        self.worker = Worker(image)
        self.request = dict(
            mode="policy",
            source=source,
            observation_space=dict(
                low=observation_space.low.tolist(),
                high=observation_space.high.tolist(),
                dtype=str(observation_space.dtype),
            ),
            nvec=action_space.nvec.tolist(),
        )

    async def reset(self, *, seed=None):
        await self.worker.__aenter__()
        await self.worker.send(self.request)
        await asyncio.wait_for(self.worker.receive(), 30)

    async def act(self, observation):
        await self.worker.send(dict(observation=observation.tolist()))
        try:
            value = await asyncio.wait_for(self.worker.receive(), 30)
            action = np.asarray(value["action"])
            if action.dtype.kind not in "iu":
                raise ValueError("Discrete actions must be integers")
            return action
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise PolicyError(f"Invalid action reply: {exc}") from exc

    async def close(self):
        if self.worker.process is not None:
            await self.worker.__aexit__()
