"""One persistent Docker evaluator, multiplexed tables and per-table cancellation."""

import asyncio
import json
import logging
from uuid import uuid4

from rsikit.evaluation import InfrastructureError

logger = logging.getLogger(f"{__package__}.tournament")
MAX_FRAME = 8 * 1024 * 1024
STARTUP_TIMEOUT = 60
CLEANUP_TIMEOUT = 15


def log_progress(request, stats):
    slow = stats.get("slowest")
    names = request.get("names", [])
    name = names[slow] if slow is not None and slow < len(names) else f"player {slow}"
    logger.info(
        "%s: %s/%s hands, %.1fs, %.1f hands/s; %s; slowest %s %.2f ms/call; warm-up %.1fs",
        request.get("label", "Table"),
        stats["hands"],
        stats["total"],
        stats["elapsed"],
        stats["hands"] / max(stats["elapsed"], 0.001),
        stats["phase"],
        name,
        stats["slowest_ms"],
        stats["warmup_seconds"],
    )


class TablePool:
    def __init__(self, config):
        self.config = config
        self.name = self.process = self.ready = None
        self.reader = self.stderr_reader = None
        self.stderr = b""
        self.pending = {}
        self.sequence = 0
        self.failure = None
        self.closed = False
        self.cleanup_task = None
        self.start_lock, self.write_lock, self.close_lock = (asyncio.Lock() for _ in range(3))
        self.slots = asyncio.Semaphore(config.workers)

    async def __aenter__(self):
        return self  # Start on the first evaluation, after policy generation.

    async def __aexit__(self, *exc):
        await self.close()

    async def start(self):
        async with self.start_lock:
            if self.failure:
                raise self.failure
            if self.closed:
                raise InfrastructureError("Evaluator pool is closed")
            if self.ready:
                return
            self.name = f"elitetable-poker-{uuid4().hex}"
            workers = self.config.workers
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
                    "--cap-add",
                    "SETUID",
                    "--cap-add",
                    "SETGID",
                    "--cap-add",
                    "KILL",
                    "--cap-add",
                    "CHOWN",
                    "--cap-add",
                    "DAC_OVERRIDE",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    str(64 * workers),
                    "--memory",
                    f"{self.config.memory_gb}g",
                    "--memory-swap",
                    f"{self.config.memory_gb}g",
                    "--cpus",
                    str(workers),
                    "--tmpfs",
                    f"/tmp:rw,noexec,nosuid,size={64 * workers}m",
                    self.config.image,
                    "--workers",
                    str(workers),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=MAX_FRAME + 1,
                )
                self.stderr_reader = asyncio.create_task(self._read_stderr())
                line = await asyncio.wait_for(self.process.stdout.readline(), STARTUP_TIMEOUT)
                ready = json.loads(line)
                if ready.get("poker_protocol") != 1 or ready.get("ready") is not True:
                    raise InfrastructureError(
                        "Rebuild the poker image for the persistent evaluator"
                    )
                self.ready = ready
                self.reader = asyncio.create_task(self._read_results())
                logger.info(
                    "Evaluator container %s ready: %s concurrent tables, %s GiB RAM, shared forkserver %s",
                    self.name,
                    workers,
                    self.config.memory_gb,
                    ready["forkserver_pid"],
                )
            except BaseException as exc:
                self.failure = InfrastructureError(
                    f"Evaluator startup failed: {exc}; {self.stderr.decode(errors='replace')}"
                )
                try:
                    await self.close()
                except Exception:
                    logger.exception("Evaluator startup cleanup failed: %s", self.name)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise self.failure from exc

    async def _read_stderr(self):
        while chunk := await self.process.stderr.read(4096):
            self.stderr = (self.stderr + chunk)[-4000:]

    def _fail_pending(self, exc):
        self.failure = self.failure or exc
        for future, _ in self.pending.values():
            if not future.done():
                future.set_exception(self.failure)

    async def _read_results(self):
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line.endswith(b"\n"):
                    raise InfrastructureError(
                        f"Evaluator disconnected: {self.stderr.decode(errors='replace')}"
                    )
                message = json.loads(line)
                future, request = self.pending[message["id"]]
                if future.done():
                    raise InfrastructureError("Duplicate table result")
                if "progress" in message:
                    log_progress(request, message["progress"])
                elif "error" in message:
                    future.set_exception(InfrastructureError(message["error"]))
                elif "result" in message or message.get("cancelled") is True:
                    future.set_result(message.get("result"))
                else:
                    raise InfrastructureError("Malformed table result")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(InfrastructureError(f"Evaluator connection failed: {exc}"))

    async def _send(self, message):
        data = json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(data) > MAX_FRAME:
            raise InfrastructureError("Table request exceeds 8 MiB")
        async with self.write_lock:
            if self.failure:
                raise self.failure
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def __call__(self, request, config):
        await self.start()
        async with self.slots:
            self.sequence += 1
            job_id = str(self.sequence)
            future = asyncio.get_running_loop().create_future()
            self.pending[job_id] = future, request
            try:
                await self._send({"id": job_id, "request": request})
                return await asyncio.wait_for(asyncio.shield(future), config.block_timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
                try:
                    await asyncio.wait_for(self._send({"cancel": job_id}), CLEANUP_TIMEOUT)
                    # Acknowledgment arrives only after all table/player processes are reaped.
                    await asyncio.wait_for(asyncio.shield(future), CLEANUP_TIMEOUT)
                except Exception:
                    await self.close()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise InfrastructureError("Table block deadline exceeded; table cancelled") from exc
            except Exception:
                await self.close()
                raise
            finally:
                self.pending.pop(job_id, None)
                if future.done() and not future.cancelled():
                    future.exception()

    async def close(self):
        if self.cleanup_task is None or (
            self.cleanup_task.done()
            and (self.cleanup_task.cancelled() or self.cleanup_task.exception() is not None)
        ):
            self.cleanup_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self.cleanup_task)
        except asyncio.CancelledError:
            await asyncio.shield(self.cleanup_task)
            raise

    async def _close(self):
        async with self.close_lock:
            if self.closed:
                return
            self._fail_pending(InfrastructureError("Evaluator pool closed"))
            for task in (self.reader, self.stderr_reader):
                if task:
                    task.cancel()
            await asyncio.gather(
                *(t for t in (self.reader, self.stderr_reader) if t), return_exceptions=True
            )
            if self.name:
                try:
                    await remove_container(self.name, self.process)
                except Exception as exc:
                    raise InfrastructureError(
                        f"Could not remove evaluator {self.name}: {exc}"
                    ) from exc
                logger.info("Evaluator container %s stopped", self.name)
            self.closed = True


async def docker_block(request, config):
    """Standalone block helper; normal runs share one TablePool across tournaments."""
    async with TablePool(config) as pool:
        return await pool(request, config)


async def remove_container(name, process):
    cleanup = None
    try:
        cleanup = await asyncio.wait_for(
            _spawn(
                "docker",
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            ),
            CLEANUP_TIMEOUT,
        )
        code = await asyncio.wait_for(cleanup.wait(), CLEANUP_TIMEOUT)
        if code:
            raise InfrastructureError(f"Docker failed to remove evaluator {name} (exit {code})")
    finally:
        for child in (cleanup, process):
            if child is not None and child.returncode is None:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(child.wait(), CLEANUP_TIMEOUT)


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
