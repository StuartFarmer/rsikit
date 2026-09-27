"""Container service: shared clean forkserver and cancellable concurrent tables."""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import socket
import sys
from tempfile import TemporaryDirectory

from poker.worker import probe, serve_table
from rsikit.sandbox.service import frame, read_frame, reap

MAX_FRAME = 8 * 1024 * 1024


async def serve(workers):
    mp.set_forkserver_preload(["numba", "poker.worker"])
    context = mp.get_context("forkserver")
    parent, child = socket.socketpair()
    starting = context.Process(target=probe, args=(child,))
    writer = None
    try:
        starting.start()
        child.close()
        reader, writer = await asyncio.open_connection(sock=parent, limit=MAX_FRAME + 1)
        ready = await asyncio.wait_for(read_frame(reader), 30)
    finally:
        child.close()
        if writer:
            writer.close()
            await writer.wait_closed()
        else:
            parent.close()
        reap(starting)

    def send(message):
        sys.stdout.buffer.write(frame(message))
        sys.stdout.buffer.flush()

    send({"ready": True, "poker_protocol": 1, "workers": workers, **ready})
    jobs = {}
    sequence = 0

    async def job(job_id, request, uid_base):
        response = None
        try:
            with TemporaryDirectory(prefix="table-") as directory:
                os.chmod(directory, 0o711)  # Seats can traverse to their own 0700 directories.
                parent, child = socket.socketpair()
                process = context.Process(
                    target=serve_table, args=(child, request, uid_base, directory)
                )
                writer = None
                try:
                    process.start()
                    child.close()
                    reader, writer = await asyncio.open_connection(sock=parent, limit=MAX_FRAME + 1)
                    while True:
                        message = await read_frame(reader)
                        if "progress" in message:
                            send({"id": job_id, **message})
                        else:
                            response = message
                            break
                finally:
                    # Group includes the table and every seat, including a hung/JIT-compiling seat.
                    reap(process, group=True)
                    child.close()
                    if writer:
                        writer.close()
                        await writer.wait_closed()
                    else:
                        parent.close()
        except asyncio.CancelledError:
            response = {"cancelled": True}
        except Exception as exc:
            response = {"error": f"Table infrastructure failure: {type(exc).__name__}: {exc}"}
        finally:
            jobs.pop(job_id, None)
        send({"id": job_id, **response})

    reader = asyncio.StreamReader(limit=MAX_FRAME + 1)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
    )
    try:
        while line := await reader.readline():
            message = json.loads(line)
            if set(message) == {"cancel"}:
                task = jobs.get(message["cancel"])
                if task:
                    task.cancel()
                continue
            if (
                set(message) != {"id", "request"}
                or not isinstance(message["id"], str)
                or not isinstance(message["request"], dict)
                or message["id"] in jobs
            ):
                raise ValueError("Malformed or duplicate table request")
            if len(jobs) >= workers:
                send({"id": message["id"], "error": "Evaluator table concurrency exceeded"})
                continue
            sequence += 1
            uid_base = 1000 + (sequence - 1) * 36
            request = {**message["request"], "stream_progress": True}
            task = asyncio.create_task(job(message["id"], request, uid_base))
            jobs[message["id"]] = task
            # Enter the job's cleanup scope before accepting an immediate cancellation.
            await asyncio.sleep(0)
    finally:
        transport.close()
        tasks = list(jobs.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    asyncio.run(serve(args.workers))


if __name__ == "__main__":
    main()
