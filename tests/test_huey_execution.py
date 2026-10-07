"""Executor owns its queue; native Huey workers run and persist real jobs."""

import asyncio
import os
import signal
import sqlite3
import tempfile
import threading
import unittest
from contextlib import aclosing
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import cloudpickle
from huey import SqliteHuey
from huey.api import Result

from rsikit import Executor, Job, PolicyDefinition, execute
from rsikit.evaluation import InfrastructureError
from tests.test_execution import SOURCE, ProcessEnv


def jobs(seeds=range(3), source=SOURCE):
    return [Job(PolicyDefinition(source=source), ProcessEnv(), seed=i) for i in seeds]


async def wait_for_file(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


class HueyExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_requires_context(self):
        executor = Executor()
        with self.assertRaisesRegex(RuntimeError, "context"):
            await executor.execute([])
        async with executor:
            self.assertEqual(await executor.execute([]), [])
        for batch in ([], jobs(range(1))):
            with self.subTest(empty=not batch), self.assertRaisesRegex(RuntimeError, "context"):
                await executor.execute(batch)

    async def test_executor_cannot_reenter(self):
        executor = Executor()
        async with executor:
            with self.assertRaises(RuntimeError):
                await executor.__aenter__()
        with self.assertRaises(RuntimeError):
            await executor.__aenter__()

    async def test_concurrent_submission_cannot_enqueue_the_same_job_twice(self):
        async with Executor() as executor:
            batch = jobs(range(1))
            started, release = threading.Event(), threading.Event()
            enqueue = executor._queue.enqueue

            def delayed(signature):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("enqueue barrier")
                return enqueue(signature)

            with patch.object(executor._queue, "enqueue", side_effect=delayed) as submit:
                first = asyncio.create_task(executor.execute(batch))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    with self.assertRaises(ValueError):
                        await executor.execute(batch)
                finally:
                    release.set()
                    completed = await first
                submit.assert_called_once()
                self.assertEqual(submit.call_args.args[0].id, batch[0].task_id)
            self.assertIs(completed[0], batch[0])
            self.assertTrue(batch[0].done)

    async def test_batch_collection_uses_bounded_async_work(self):
        from rsikit.execution.executor import _queue_io

        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            source = SOURCE.replace(
                "return 0",
                f"""import asyncio
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
            )
            async with Executor(episode_timeout=10) as executor:
                active = peak = 0

                async def counted(function, *args, **kwargs):
                    nonlocal active, peak
                    active += 1
                    peak = max(peak, active)
                    try:
                        return await _queue_io(function, *args, **kwargs)
                    finally:
                        active -= 1

                baseline = len(asyncio.all_tasks())
                batch = jobs(range(128), source)
                with patch("rsikit.execution.executor._queue_io", counted):
                    collection = asyncio.create_task(executor.execute(batch))
                    try:

                        async def enqueued():
                            while (
                                any(job.task_id is None for job in batch)
                                or executor._queue.pending_count() < 127
                            ):
                                await asyncio.sleep(0.01)

                        await asyncio.wait_for(enqueued(), 5)
                        self.assertLessEqual(len(asyncio.all_tasks()), baseline + 4)
                        self.assertEqual(peak, 1)
                    finally:
                        collection.cancel()
                        await asyncio.gather(collection, return_exceptions=True)
                        release.touch()

    async def test_cancellation_during_enqueue_revokes_the_submitted_task(self):
        with tempfile.TemporaryDirectory() as directory:
            started, release = (Path(directory) / name for name in ("started", "release"))
            source = SOURCE.replace(
                "return 0",
                f"""import asyncio
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
            )
            async with Executor(episode_timeout=10) as executor:
                first = asyncio.create_task(executor.execute(jobs(range(1), source)))
                enqueue_started, enqueue_release = threading.Event(), threading.Event()
                target = jobs(range(1))[0]
                signatures = []
                enqueue = executor._queue.enqueue

                def delayed(signature):
                    signatures.append(signature)
                    enqueue_started.set()
                    if not enqueue_release.wait(5):
                        raise TimeoutError("enqueue barrier")
                    return enqueue(signature)

                try:
                    await wait_for_file(started)
                    with patch.object(executor._queue, "enqueue", delayed):
                        submission = asyncio.create_task(executor.execute([target]))
                        await asyncio.wait_for(asyncio.to_thread(enqueue_started.wait, 5), 6)
                        submission.cancel()
                        await asyncio.sleep(0.02)
                        self.assertFalse(submission.done())
                        enqueue_release.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await submission
                    self.assertEqual(target.task_id, signatures[0].id)
                    self.assertTrue(executor._queue.is_revoked(signatures[0]))
                    self.assertIsNone(target.result)
                finally:
                    enqueue_release.set()
                    release.touch()
                    completed = await first
                self.assertEqual(completed[0].result.total_reward, 2)
                self.assertEqual(len(await executor.execute(jobs(range(1)))), 1)

    async def test_ready_sibling_is_yielded_before_slow_first_job(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            source = SOURCE.replace(
                "return 0",
                f"""import asyncio
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
            )
            slow, fast = jobs(range(1), source)[0], jobs(range(1))[0]
            async with Executor(concurrency=2, episode_timeout=10) as executor:
                async with aclosing(executor.iterate([slow, fast])) as results:
                    try:
                        self.assertIs(await asyncio.wait_for(anext(results), 5), fast)
                        self.assertIsNone(slow.result)
                    finally:
                        release.touch()
                    self.assertEqual([job async for job in results], [slow])

    async def test_enqueue_failure_is_an_infrastructure_error(self):
        async with Executor() as executor:
            batch = jobs(range(1))
            with patch.object(
                executor._queue, "enqueue", side_effect=sqlite3.OperationalError("disk full")
            ):
                with self.assertRaisesRegex(InfrastructureError, "disk full"):
                    await executor.execute(batch)
            self.assertFalse(batch[0].done)

    async def test_closing_partial_iterator_revokes_its_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            started, release, queued = (
                Path(directory) / name for name in ("started", "release", "queued")
            )
            source = SOURCE.replace(
                "return 0",
                f"""import asyncio
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
            )
            fast = jobs(range(1))[0]
            slow = jobs(range(1), source)[0]
            last = jobs(range(1), SOURCE + f"\nPath({str(queued)!r}).touch()")[0]
            async with Executor(episode_timeout=10) as executor:
                try:
                    async with aclosing(executor.iterate([fast, slow, last])) as results:
                        self.assertIs(await asyncio.wait_for(anext(results), 5), fast)
                        await wait_for_file(started)
                    self.assertTrue(fast.done)
                    self.assertFalse(slow.done or last.done)
                    # The queued third task is revoked before releasing the occupied worker.
                    signature = executor._task.s(b"", None, None, None, id=last.task_id)
                    self.assertTrue(Result(executor._queue, signature).is_revoked())
                finally:
                    release.touch()
            self.assertFalse(queued.exists())

    async def test_partial_worker_start_failure_cleans_up(self):
        from multiprocessing.context import ForkProcess

        start = ForkProcess.start
        started = []

        def fail_third(process):
            if len(started) == 2:
                raise OSError("cannot start worker")
            start(process)
            started.append(process)

        executor = Executor(concurrency=2)
        with patch.object(ForkProcess, "start", fail_third):
            with self.assertRaisesRegex(OSError, "cannot start worker"):
                await executor.__aenter__()
        self.assertFalse(any(process.is_alive() for process in started))
        self.assertIsNone(executor._directory)

    async def test_public_api_reuses_workers_and_retains_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.sqlite"
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            async with Executor(database=path, concurrency=2) as executor:
                consumer = executor._consumer
                pids = {p.pid for _, p in consumer.worker_threads}
                self.assertEqual(consumer.worker_type, "process")
                self.assertNotIn(os.getpid(), pids)
                self.assertTrue(executor._queue.storage._fsync)
                first, second = jobs(), jobs(range(3, 6))
                for batch in (first, second):
                    completed = await executor.execute(batch)
                    self.assertEqual(set(completed), set(batch))
                    self.assertTrue(all(job.result.total_reward == 2 for job in batch))
                    self.assertTrue(all(job.result.infos[0]["pid"] in pids for job in batch))
                self.assertEqual(await executor.execute([]), [])
            self.assertTrue(path.is_file())
            self.assertFalse(consumer.scheduler.is_alive())
            self.assertFalse(any(p.is_alive() for _, p in consumer.worker_threads))
            self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})
            queue = SqliteHuey("evaluations", filename=str(path))
            try:
                for job in first + second:
                    self.assertEqual(
                        queue.result(job.task_id, preserve=True)["rewards"], [1.0, 1.0]
                    )
            finally:
                queue.storage.close()
            async with Executor(database=path) as reopened:
                self.assertEqual(len(await reopened.execute(jobs(range(1)))), 1)

    async def test_context_reuses_workers_then_cleans_up(self):
        async with Executor() as executor:
            path = Path(executor._queue.storage.filename)
            self.assertTrue(path.exists())
            first = await executor.execute(jobs(range(1)))
            second = await executor.execute(jobs(range(1)))
            self.assertEqual(first[0].result.infos[0]["pid"], second[0].result.infos[0]["pid"])
        self.assertFalse(path.exists())
        for database in ("", ":memory:"):
            with self.assertRaises(ValueError):
                Executor(database=database)
        self.assertEqual(len(await execute(jobs(range(1)))), 1)

    async def test_sqlite_write_contention_does_not_block_event_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.sqlite"
            async with Executor(database=path) as executor:
                locked = threading.Event()

                def hold_write_lock():
                    with sqlite3.connect(path) as db:
                        db.execute("BEGIN IMMEDIATE")
                        locked.set()
                        sleep(0.4)

                holder = asyncio.create_task(asyncio.to_thread(hold_write_lock))
                await asyncio.to_thread(locked.wait)
                evaluation = asyncio.create_task(executor.execute(jobs(range(1))))
                started = monotonic()
                try:
                    await asyncio.sleep(0.05)
                    self.assertLess(monotonic() - started, 0.25)
                finally:
                    evaluation.cancel()
                    await asyncio.gather(evaluation, holder, return_exceptions=True)

    async def test_pending_payload_survives_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.sqlite"
            queue = SqliteHuey("evaluations", filename=str(path))
            # Encode the original module's task identity independently of Executor.
            legacy_task = queue.task(name="rsikit_evaluate", __module__="rsikit.execution")(
                lambda *args: None
            )
            try:
                signature = legacy_task.s(
                    cloudpickle.dumps((PolicyDefinition(source=SOURCE), ProcessEnv())),
                    42,
                    None,
                    None,
                    timeout=10,
                )
                queue.enqueue(signature)
                async with Executor(database=path):

                    async def result():
                        while (data := queue.result(signature.id, preserve=True)) is None:
                            await asyncio.sleep(0.01)
                        return data

                    data = await asyncio.wait_for(result(), 5)
                    self.assertEqual(data["infos"][0]["seed"], 42)
            finally:
                queue.storage.close()

    async def test_cancellation_revokes_queued_jobs_and_shutdown_drains_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            started, release, closed, queued = (
                Path(directory) / name for name in ("started", "release", "closed", "queued")
            )
            source = (
                SOURCE.replace(
                    "return 0",
                    f"""import asyncio
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
                )
                + f"\n    async def close(self):\n        Path({str(closed)!r}).touch()\n"
            )
            executor = Executor(episode_timeout=5)
            async with executor:
                batch = jobs(range(1), source) + jobs(
                    range(1), SOURCE + f"\nPath({str(queued)!r}).touch()"
                )
                evaluation = asyncio.create_task(executor.execute(batch))
                await wait_for_file(started)
                evaluation.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await evaluation
                self.assertTrue(all(job.result is None for job in batch))
                closing = asyncio.create_task(executor.__aexit__(None, None, None))
                try:
                    await asyncio.sleep(0.05)
                    self.assertFalse(closing.done())
                    release.touch()
                    await asyncio.wait_for(closing, 5)
                    self.assertTrue(closed.exists())
                    self.assertFalse(queued.exists())
                finally:
                    release.touch()
                    await closing

    async def test_batches_share_a_worker_without_cancellation_spilling_over(self):
        with tempfile.TemporaryDirectory() as directory:
            started, release = (Path(directory) / name for name in ("started", "release"))
            source = SOURCE.replace(
                "return 0",
                f"""import asyncio
        Path({str(started)!r}).touch()
        while not Path({str(release)!r}).exists():
            await asyncio.sleep(0.01)
        return 0""",
            )
            async with Executor(episode_timeout=5) as executor:
                first = asyncio.create_task(executor.execute(jobs(range(1), source)))
                try:
                    await wait_for_file(started)
                    second = asyncio.create_task(executor.execute(jobs()))
                    await asyncio.sleep(0.05)
                    second.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await second
                    release.touch()
                    self.assertEqual((await asyncio.wait_for(first, 5))[0].result.total_reward, 2)
                    self.assertEqual(len(await executor.execute(jobs(range(1)))), 1)
                finally:
                    release.touch()
                    await first
