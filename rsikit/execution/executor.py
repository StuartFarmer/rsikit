"""Own the Huey queue and collect results without blocking the event loop."""

import asyncio
import math
import sys
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
from huey import SqliteHuey
from huey.api import Result
from huey.consumer import Consumer
from huey.exceptions import TaskException

from ..evaluation import InfrastructureError
from . import worker
from .job import Job
from .results import attach_result


async def _queue_io(function, *args, **kwargs):
    """Finish a storage operation before propagating cancellation to its owner."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await asyncio.gather(operation, return_exceptions=True)
        raise


class _EmbeddedConsumer(Consumer):
    def _set_signal_handlers(self):
        # The embedding application owns signals and the Executor context.
        pass


class Executor:
    """Own a SQLite queue and Huey process workers across evaluation batches.

    database=None uses temporary storage. A file path retains pending tasks and
    results across restarts. Cancellation revokes queued tasks; running tasks finish
    under Huey's signal-based timeout. Each instance requires one async context;
    finish batch coroutines and close partial iterators before exiting it.
    """

    def __init__(
        self,
        *,
        concurrency: int = 1,
        database: str | Path | None = None,
        episode_timeout: float = 60.0,
    ):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not math.isfinite(episode_timeout) or episode_timeout <= 0:
            raise ValueError("episode_timeout must be positive and finite")
        if database is not None and (not str(database) or str(database) == ":memory:"):
            raise ValueError("database must be a file path, or None for temporary storage")
        self.concurrency = concurrency
        self.database = Path(database).expanduser().resolve() if database is not None else None
        self.episode_timeout = episode_timeout
        self._queue = None
        self._directory = None
        self._consumer = None
        self._state = "new"

    async def __aenter__(self):
        if self._state != "new":
            raise RuntimeError("Executor has one context lifetime; create a new Executor")
        self._state = "starting"
        try:
            database = self.database
            if database is None:
                self._directory = TemporaryDirectory(prefix="rsikit-queue-")
                database = Path(self._directory.name) / "queue.sqlite"
            self._queue = await _queue_io(
                SqliteHuey, "evaluations", filename=str(database), fsync=True
            )
            # Keep queued tasks readable after moving the worker into this package.
            self._task = self._queue.task(name="rsikit_evaluate", __module__="rsikit.execution")(
                worker.run_episode
            )
            self._queue.storage.close()  # Workers must open their own SQLite connection.
            self._consumer = _EmbeddedConsumer(
                self._queue,
                workers=self.concurrency,
                worker_type="process",
                periodic=False,
                check_worker_health=False,
                initial_delay=0.01,
                max_delay=0.05,
            )
            self._consumer.start()
            self._state = "active"
            return self
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(self, *exc):
        self._state = "closed"
        try:
            if self._consumer is not None:
                # Startup may have failed after starting only some of Huey's processes.
                self._consumer.worker_threads = [
                    (worker, process)
                    for worker, process in self._consumer.worker_threads
                    if process.pid is not None
                ]
                await _queue_io(
                    self._consumer.stop, graceful=self._consumer.scheduler.pid is not None
                )
        finally:
            try:
                if self._queue is not None:
                    await _queue_io(self._queue.storage.close)
            finally:
                if self._directory is not None:
                    self._directory.cleanup()
                    self._directory = None

    async def execute(self, jobs: Iterable[Job]) -> list[Job]:
        """Return original jobs in observed readiness order, with results attached."""
        return [job async for job in self.iterate(jobs)]

    def _reserve(self, jobs: Iterable[Job]) -> list[Job]:
        """Validate and reserve the entire batch before submission can yield."""
        if self._state != "active":
            raise RuntimeError("Execution requires an open Executor context")
        jobs = list(jobs)
        if any(not isinstance(job, Job) for job in jobs):
            raise TypeError("execute expects Job instances")
        if len({id(job) for job in jobs}) != len(jobs) or any(job._submitted for job in jobs):
            raise ValueError("Each Job can only be submitted once")
        for job in jobs:
            job._submitted = True
        return jobs

    async def _enqueue(self, job: Job, definition: bytes, pending):
        """Register a task for cancellation before its enqueue can be interrupted."""
        signature = self._task.s(
            definition,
            job.seed,
            job.max_steps,
            job.instructions,
            timeout=self.episode_timeout,
        )
        job.task_id = signature.id
        pending[Result(self._queue, signature)] = job
        try:
            await _queue_io(self._queue.enqueue, signature)
        except Exception as exc:
            raise InfrastructureError(f"Cannot enqueue job: {exc}") from exc

    def _read_ready(self, pending):
        """Read each handle once, retaining successful outcomes alongside failures."""
        # ponytail: linear scan; batch result queries if polling becomes a bottleneck.
        ready = []
        for result in pending:
            try:
                data = result.get(preserve=self.database is not None)
            except Exception as exc:
                data = exc
            if data is not None:
                ready.append((result, data))
        return ready

    async def iterate(self, jobs: Iterable[Job]) -> AsyncIterator[Job]:
        """Stream completed jobs; use contextlib.aclosing when stopping early.

        Infrastructure failures are raised after successful siblings finish.
        Cancellation revokes queued work; running jobs continue in Huey. Results
        ready in the same sweep have no chronological ordering guarantee.
        """
        jobs = self._reserve(jobs)
        pending = {}
        error = None

        try:
            for job in jobs:
                try:
                    definition = cloudpickle.dumps((job.policy, job.environment))
                except Exception as exc:
                    error = error or InfrastructureError(
                        f"Cannot serialize {type(job.policy).__qualname__} job inputs: {exc}"
                    )
                    continue
                await self._enqueue(job, definition, pending)

            while pending:
                storage_error = None
                for result, data in await _queue_io(self._read_ready, pending):
                    if isinstance(data, Exception) and not isinstance(data, TaskException):
                        storage_error = InfrastructureError(f"Cannot read Huey result: {data}")
                        continue
                    job = pending.pop(result)
                    try:
                        attach_result(job, data, self.episode_timeout)
                    except InfrastructureError as exc:
                        error = error or exc
                        continue
                    yield job
                if storage_error is not None:
                    raise storage_error
                if pending:
                    if any(not p.is_alive() for _, p in self._consumer.worker_threads):
                        raise InfrastructureError(
                            "Huey worker exited without a result; create a new Executor"
                        )
                    await asyncio.sleep(0.01)
            if error is not None:
                raise error
        finally:
            for result in pending:
                await _queue_io(result.revoke)


async def execute(
    jobs: Iterable[Job],
    *,
    concurrency: int = 1,
    database: str | Path | None = None,
    episode_timeout: float = 60.0,
) -> list[Job]:
    """Execute one batch; use an Executor context to share workers across batches."""
    async with Executor(
        concurrency=concurrency, database=database, episode_timeout=episode_timeout
    ) as executor:
        return await executor.execute(jobs)
