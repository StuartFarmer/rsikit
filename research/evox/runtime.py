"""Run selection programs in fresh bounded processes inside the application container."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

MAX_MESSAGE = 1024 * 1024


async def run_python_strategy(code, population, state, seed, *, timeout=5.0) -> dict:
    """Execute select(population, state, rng); a process is not a security sandbox."""
    payload = json.dumps(
        dict(code=code, population=population, state=state, seed=seed), allow_nan=False
    ).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("strategy input exceeds 1 MiB")
    with TemporaryDirectory(prefix="evox-strategy-") as directory:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(Path(__file__).with_name("_worker.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=directory,
            env={},
            start_new_session=True,
        )

        async def exchange():
            process.stdin.write(payload)
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()
            output = bytearray()
            while chunk := await process.stdout.read(65536):
                output.extend(chunk)
                if len(output) > MAX_MESSAGE:
                    raise ValueError("strategy output exceeds 1 MiB")
            await process.wait()
            return output

        try:
            output = await asyncio.wait_for(exchange(), timeout)
            if process.returncode:
                raise ValueError(f"strategy exited with status {process.returncode}")
        except asyncio.TimeoutError as exc:
            raise ValueError("strategy execution timed out") from exc
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if sys.platform != "darwin" or process.returncode is None:
                    raise
            # Drain the killed child's pipe so asyncio can close its transport.
            while await process.stdout.read(65536):
                pass
            await process.wait()
    try:
        result = json.loads(output)
        if result.get("error"):
            raise ValueError(result["error"])
        selection = result["selection"]
        if not isinstance(selection, dict):
            raise ValueError("strategy must return a dictionary")
        return selection
    except (KeyError, TypeError, AttributeError, UnicodeError) as exc:
        raise ValueError("strategy returned invalid JSON output") from exc
