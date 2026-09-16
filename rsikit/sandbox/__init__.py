"""One restricted Docker container per run, with a warm Python fork server."""

import asyncio
import json
from uuid import uuid4

from rsikit.evaluation import EvaluationError


class PythonSandbox:
    """Use as an async context manager; execute requests sequentially.

    Candidate code stays inside Docker, with no host mounts, credentials, or
    network. Each call forks clean interpreter state and calls the entry point
    once. This is a local research container, not a hostile multi-tenant service.
    """

    def __init__(self, *, image="rsikit-sandbox:local", timeout=10):
        self.image, self.timeout = image, timeout
        self.name = f"rsikit-{uuid4().hex}"
        self.process = None

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            self.image,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=131_072,
        )
        try:
            ready = await asyncio.wait_for(self.process.stdout.readline(), 60)
            if json.loads(ready) != {"ready": True}:
                raise ValueError("Missing ready message")
        except asyncio.CancelledError:
            await self.__aexit__(None, None, None)
            raise
        except (TimeoutError, ValueError, UnicodeError):
            await self.__aexit__(None, None, None)
            raise EvaluationError(
                "Sandbox failed to start. Start Docker and build: "
                "docker build -t rsikit-sandbox:local rsikit/sandbox"
            ) from None
        return self

    async def __aexit__(self, *_):
        if self.process is not None:
            # EOF also stops a worker created after an early cancellation's rm.
            self.process.stdin.close()
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self.name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await cleanup.wait()
            await self.process.wait()
            self.process = None

    async def __call__(self, source: str, *, function="pack_circles") -> dict:
        if len(source.encode()) > 65_536:
            return {"error": "Source exceeds 64 KiB"}
        request = {"source": source, "function": function, "timeout": self.timeout}
        try:
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout + 5)
            result = json.loads(line)
            if not isinstance(result, dict) or set(result) not in ({"value"}, {"error"}):
                raise ValueError("Invalid worker response")
            return result
        except (
            TimeoutError,
            ValueError,
            UnicodeError,
            BrokenPipeError,
            ConnectionResetError,
        ) as exc:
            raise EvaluationError("Sandbox worker failed; candidate was not retried") from exc
