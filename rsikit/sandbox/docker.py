"""One Docker container per batch, with a separate evaluation process for each job."""

import asyncio
import base64
import json
import sys
from uuid import uuid4

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout

MAX_RESULT = 64 * 1024 * 1024
WORKER = "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.evaluate import main; main()"


async def _spawn(*args, **kwargs):
    """Finish acquiring the process handle before cancellation attempts cleanup."""
    starting = asyncio.create_task(asyncio.create_subprocess_exec(*args, **kwargs))
    try:
        return await asyncio.shield(starting)
    except asyncio.CancelledError:
        process = await starting
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise


class DockerSandbox:
    def __init__(self, *, image: str = "rsikit-sandbox:local"):
        self.image = image
        self.name = None

    async def start(self, workers: int) -> None:
        self.name = f"rsikit-{uuid4().hex}"
        process = await _spawn(
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "-d",
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
            str(64 * workers),
            "--memory",
            f"{workers}g",
            "--memory-swap",
            f"{workers}g",
            "--cpus",
            str(workers),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--entrypoint",
            "sleep",
            self.image,
            "infinity",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, error = await asyncio.wait_for(process.communicate(), 60)
            if process.returncode:
                raise InfrastructureError(
                    f"Docker sandbox failed to start: {error.decode(errors='replace')}"
                )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def evaluate(
        self, implementation: str, environment: bytes, seed: int, call_timeout: float
    ) -> tuple[float, dict[str, bytes]]:
        request = json.dumps(
            {
                "implementation": implementation,
                "environment": base64.b64encode(environment).decode(),
                "python": list(sys.version_info[:2]),
                "seed": seed,
                "call_timeout": call_timeout,
            }
        ).encode()
        process = await _spawn(
            "docker",
            "exec",
            "-i",
            self.name,
            "python",
            "-I",
            "-u",
            "-c",
            WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=MAX_RESULT + 1,
        )
        try:
            process.stdin.write(request)
            await process.stdin.drain()
            process.stdin.close()
            response = await process.stdout.readline()
            await process.wait()
            if process.returncode or not response or len(response) > MAX_RESULT:
                raise InfrastructureError(
                    "Sandbox evaluation failed or returned an oversized result"
                )
            result = json.loads(response)
            if "error" in result:
                error = {"policy": PolicyError, "timeout": PolicyTimeout}.get(
                    result.get("kind"), InfrastructureError
                )
                raise error(result["error"])
            return result["score"], {
                name: base64.b64decode(data, validate=True)
                for name, data in result["artifacts"].items()
            }
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def close(self) -> None:
        if self.name is None:
            return
        name, self.name = self.name, None
        process = await _spawn(
            "docker",
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), 15)
            if process.returncode:
                raise InfrastructureError(f"Could not remove Docker sandbox {name}")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
