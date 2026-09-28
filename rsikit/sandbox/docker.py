"""One Docker container and persistent Python service for each execution context."""

import asyncio
import base64
import codecs
import json
import logging
import math
import sys
from uuid import uuid4

from rsikit.episode import Episode
from rsikit.evaluation import InfrastructureError, PolicyError, PolicyTimeout
from rsikit.sandbox.codec import decode_episode

MAX_RESULT = 64 * 1024 * 1024
MAX_LOG = 1024 * 1024


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
    """Policy, environment and scoring share a fresh process inside Docker.

    Protects the host, not scoring integrity. The supervisor bounds the whole
    episode and reaps its descendants; it does not impose per-action deadlines.
    """

    def __init__(self, *, image: str = "rsikit-sandbox:local", episode_timeout: float = 60.0):
        if not math.isfinite(episode_timeout) or episode_timeout <= 0:
            raise ValueError("episode_timeout must be positive and finite")
        self.image = image
        self.episode_timeout = episode_timeout
        self.name = None
        self.process = None
        self.ready = None
        self._reader = None
        self._logs = None
        self._cleanup_task = None
        self._pending = {}
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._failure = None
        self._closing = False
        self._sequence = 0

    async def start(self, workers: int) -> None:
        if self._cleanup_task is not None:
            await asyncio.shield(self._cleanup_task)
            self._cleanup_task = None
        if self.name is not None:
            raise InfrastructureError("Sandbox is already started")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        self._failure = None
        self._closing = False
        self.name = f"rsikit-{uuid4().hex}"
        try:
            self.process = await _spawn(
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--init",
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
                "python",
                self.image,
                "-I",
                "-u",
                "-c",
                f"from rsikit.sandbox.service import main; main({workers})",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_RESULT + 1,
            )
            self._logs = asyncio.create_task(self._read_logs())
            line = await asyncio.wait_for(self.process.stdout.readline(), 60)
            ready = json.loads(line)
            if (
                not isinstance(ready, dict)
                or ready.get("ready") is not True
                or ready.get("protocol") != 3
                or any(
                    type(ready.get(key)) is not int or ready[key] <= 0
                    for key in ("supervisor_pid", "forkserver_pid")
                )
            ):
                raise InfrastructureError(
                    "Invalid sandbox readiness message; rebuild the Docker image"
                )
            self.ready = ready
            self._reader = asyncio.create_task(self._read_results())
        except BaseException as exc:
            try:
                await self.close()
            except Exception:
                logging.getLogger(__name__).exception("Sandbox startup cleanup failed")
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise InfrastructureError(f"Docker sandbox failed to start: {exc}") from exc

    async def _read_logs(self):
        logger = logging.getLogger(__name__)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        remaining = MAX_LOG
        # ponytail: cap forwarded logs per container; add per-job budgets if needed.
        # Continue draining after the cap so noisy policies cannot block on stderr.
        while chunk := await self.process.stderr.read(4096):
            if not remaining:
                continue
            text = decoder.decode(chunk[:remaining], final=len(chunk) >= remaining)
            remaining -= min(len(chunk), remaining)
            text = "".join(c if c.isprintable() or c in "\n\t" else "\ufffd" for c in text)
            if text:
                logger.info("Sandbox: %s", text.rstrip("\n"), extra={"event": "sandbox_log"})
            if not remaining:
                logger.info("Sandbox log limit reached; further output discarded")

    def _fail_pending(self, error):
        self._failure = self._failure or error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(self._failure)
        self._pending.clear()

    async def _read_results(self):
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line.endswith(b"\n") or len(line) > MAX_RESULT + 1:
                    raise ValueError("Missing or oversized evaluation result")
                envelope = json.loads(line)
                if (
                    not isinstance(envelope, dict)
                    or set(envelope) != {"id", "result"}
                    or not isinstance(envelope["id"], str)
                    or envelope["id"] not in self._pending
                ):
                    raise ValueError("Unknown, duplicate, or malformed evaluation result")
                result = envelope["result"]
                if not isinstance(result, dict):
                    raise ValueError("Evaluation result must be an object")
                if set(result) == {"kind", "error"}:
                    if result["kind"] not in (
                        "policy",
                        "timeout",
                        "infrastructure",
                    ) or not isinstance(result["error"], str):
                        raise ValueError("Malformed evaluation error")
                    error_type = {"policy": PolicyError, "timeout": PolicyTimeout}.get(
                        result["kind"], InfrastructureError
                    )
                    if error_type is InfrastructureError:
                        raise InfrastructureError(result["error"])
                    error = error_type(result["error"])
                    value = None
                else:
                    if set(result) != {"episode"}:
                        raise ValueError("Malformed evaluation episode")
                    value = decode_episode(result["episode"])
                    error = None
                future = self._pending.pop(envelope["id"])
                if future.done():
                    raise ValueError("Response to an abandoned evaluation")
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(value)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(InfrastructureError(f"Sandbox connection failed: {exc}"))
            if self.name is not None and not self._closing:
                self._cleanup_task = asyncio.create_task(self.close())
                self._cleanup_task.add_done_callback(self._cleanup_finished)

    @staticmethod
    def _cleanup_finished(task):
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error("Sandbox cleanup failed: %s", task.exception())

    async def evaluate(
        self,
        implementation: str,
        environment: bytes,
        seed: int | None,
        *,
        policy_seed=...,
        max_steps: int | None = None,
        instructions: str | None = None,
    ) -> Episode:
        if self._failure is not None:
            raise self._failure
        if self.process is None or self.name is None or self._reader is None:
            raise InfrastructureError("Sandbox is not running")
        self._sequence += 1
        job_id = str(self._sequence)
        payload = json.dumps(
            {
                "id": job_id,
                "request": {
                    "implementation": implementation,
                    "environment": base64.b64encode(environment).decode(),
                    "python": list(sys.version_info[:2]),
                    "seed": seed,
                    "policy_seed": seed if policy_seed is ... else policy_seed,
                    "max_steps": max_steps,
                    "instructions": instructions,
                    "episode_timeout": self.episode_timeout,
                },
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if len(payload) > MAX_RESULT:
            raise InfrastructureError("Evaluation request exceeds 64 MiB")
        future = asyncio.get_running_loop().create_future()
        self._pending[job_id] = future
        try:
            async with self._write_lock:
                if self._failure is not None:
                    raise self._failure
                self.process.stdin.write(payload + b"\n")
                await self.process.stdin.drain()
            return await future
        except PolicyError:
            raise
        except BaseException as exc:
            self._fail_pending(InfrastructureError("Sandbox evaluation interrupted"))
            try:
                await self.close()
            except Exception:
                logging.getLogger(__name__).exception("Sandbox cleanup failed during evaluation")
            if isinstance(exc, (asyncio.CancelledError, InfrastructureError)):
                raise
            raise InfrastructureError(f"Sandbox request failed: {exc}") from exc
        finally:
            self._pending.pop(job_id, None)
            if future.done() and not future.cancelled():
                future.exception()  # A cancelled write may never have awaited this future.

    async def close(self) -> None:
        async with self._close_lock:
            self._closing = True
            self._fail_pending(InfrastructureError("Sandbox closed"))
            try:
                if self.name is not None:
                    cleanup = await _spawn(
                        "docker",
                        "rm",
                        "-f",
                        self.name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    try:
                        await asyncio.wait_for(cleanup.wait(), 15)
                        if cleanup.returncode:
                            raise InfrastructureError(
                                f"Could not remove Docker sandbox {self.name}"
                            )
                        self.name = None
                    finally:
                        if cleanup.returncode is None:
                            cleanup.kill()
                            await cleanup.wait()
            finally:
                if self._reader is not None:
                    self._reader.cancel()
                    await asyncio.gather(self._reader, return_exceptions=True)
                    self._reader = None
                if self._logs is not None:
                    self._logs.cancel()
                    await asyncio.gather(self._logs, return_exceptions=True)
                    self._logs = None
                if self.process is not None:
                    if self.process.stdin is not None:
                        self.process.stdin.close()
                    try:
                        await asyncio.wait_for(self.process.communicate(), 15)
                    except asyncio.TimeoutError:
                        if self.process.returncode is None:
                            self.process.kill()
                        await self.process.communicate()
                    self.process = None
                self.ready = None
