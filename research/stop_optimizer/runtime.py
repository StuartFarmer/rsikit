"""Host-side Docker execution for improvers and downstream policy evaluation."""

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from rsikit.evaluation import InfrastructureError

from .agent import BudgetExhausted, ImproverExecutionError

MAX_FRAME = 1024 * 1024


class DockerRuntime:
    """Keep generated code in containers with host-owned capability budgets."""

    def __init__(self, *, image="rsikit:local", timeout=1200.0):
        self.image, self.timeout = image, timeout
        self._locks = {}

    async def preflight(self):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            self.image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, error = await process.communicate()
        if process.returncode:
            raise InfrastructureError(
                f"Docker image {self.image} unavailable: {error.decode(errors='replace')[-2000:]}"
            )

    async def execute(self, source, initial, capabilities):
        result = await self._run(
            "_worker",
            dict(
                source=source,
                initial=initial,
                generations=capabilities.generations_left,
                evaluations=capabilities.evaluations_left,
            ),
            capabilities=capabilities,
        )
        if not isinstance(result, str):
            raise ImproverExecutionError("Improver must return source text")
        return result

    async def evaluate_policy(self, source, *, path, environment, seeds, max_steps=None):
        path = Path(path).resolve()
        async with self._locks.setdefault(path, asyncio.Lock()):
            path.mkdir(parents=True, exist_ok=True)
            return await self._run(
                "_evaluate",
                dict(
                    source=source, environment=environment, seeds=list(seeds), max_steps=max_steps
                ),
                output=path,
            )

    async def _run(self, module, request, *, capabilities=None, output=None):
        name = f"rsikit-stop-{uuid4().hex}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--init",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=64m",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
        ]
        if output is not None:
            command += ["--mount", f"type=bind,src={output},dst=/task"]
        command += [self.image, "python", "-u", f"/app/research/stop_optimizer/{module}.py"]
        payload = json.dumps(request, allow_nan=False).encode() + b"\n"
        if len(payload) > MAX_FRAME:
            raise ImproverExecutionError("Worker input exceeds 1 MiB")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=MAX_FRAME + 1,
        )
        diagnostics = bytearray()

        async def drain_errors():
            while chunk := await process.stderr.read(8192):
                diagnostics.extend(chunk)
                del diagnostics[:-8192]

        errors = asyncio.create_task(drain_errors())

        async def send(data):
            frame = json.dumps(data, allow_nan=False).encode() + b"\n"
            if len(frame) > MAX_FRAME:
                raise ImproverExecutionError("Worker response exceeds 1 MiB")
            process.stdin.write(frame)
            await process.stdin.drain()

        async def converse():
            process.stdin.write(payload)
            await process.stdin.drain()
            expected_id = 0
            while True:
                try:
                    frame = await process.stdout.readline()
                    if len(frame) > MAX_FRAME:
                        raise ValueError("frame exceeds 1 MiB")
                    if not frame:
                        await process.wait()
                        await errors
                        raise InfrastructureError(
                            f"Worker exited without a result ({process.returncode}): {diagnostics.decode(errors='replace')}"
                        )
                    message = json.loads(frame)
                except (ValueError, UnicodeError) as exc:
                    raise ImproverExecutionError("Invalid worker JSON frame") from exc
                if not isinstance(message, dict):
                    raise ImproverExecutionError("Worker frame must be an object")
                if set(message) == {"result"}:
                    return message["result"]
                if set(message) == {"error"} and isinstance(message["error"], str):
                    raise ImproverExecutionError(message["error"])
                if capabilities is None:
                    raise ImproverExecutionError("Evaluation worker cannot request capabilities")
                if (
                    set(message) - {"id", "method", "source", "guidance"}
                    or type(message.get("id")) is not int
                    or message["id"] != expected_id
                    or message.get("method") not in ("suggest", "evaluate")
                    or not isinstance(message.get("source"), str)
                    or not isinstance(message.get("guidance", ""), str)
                ):
                    raise ImproverExecutionError("Invalid capability request")
                expected_id += 1
                try:
                    if message["method"] == "suggest":
                        value = await capabilities.suggest(
                            message["source"], message.get("guidance", "")
                        )
                    else:
                        value = await capabilities.evaluate(message["source"])
                except BudgetExhausted as exc:
                    await send(dict(id=message["id"], error=str(exc)))
                except TimeoutError as exc:
                    raise InfrastructureError("Host capability timed out") from exc
                else:
                    await send(dict(id=message["id"], result=value))

        try:
            return await asyncio.wait_for(converse(), self.timeout)
        except asyncio.TimeoutError as exc:
            raise ImproverExecutionError("Improver execution timed out") from exc
        finally:
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await cleanup.wait()
            if process.returncode is None:
                process.kill()
            if not process.stdin.is_closing():
                process.stdin.close()
            while await process.stdout.read(65536):
                pass
            await process.wait()
            await errors
