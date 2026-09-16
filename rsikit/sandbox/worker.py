"""Container-only JSON-lines worker; fork from warm numerical imports per call."""

import contextlib
import json
import os
import resource
import select
import signal
import sys
import time

import numpy  # noqa: F401
import scipy.optimize  # noqa: F401

LIMIT = 65_536


def execute(request):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            os.setsid()
            # Numerical candidates use one process; prevent escaped descendants.
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # Candidate output cannot corrupt the supervisor's JSON protocol.
            with open(os.devnull, "r+b", buffering=0) as devnull:
                for fd in (0, 1, 2):
                    os.dup2(devnull.fileno(), fd)
            try:
                namespace = {"__name__": "candidate"}
                exec(compile(request["source"], "candidate.py", "exec"), namespace)
                value = namespace[request["function"]]()
                result = json.dumps({"value": value}, allow_nan=False).encode()
                if len(result) > LIMIT:
                    raise ValueError("Returned JSON exceeds 64 KiB")
            except BaseException as exc:
                result = json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}).encode()
            with os.fdopen(write_fd, "wb") as output:
                output.write(result)
        finally:
            os._exit(0)
    os.close(write_fd)
    deadline = time.monotonic() + request["timeout"]
    data = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
                return {"error": f"Execution timeout after {request['timeout']}s"}
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > LIMIT:
                return {"error": "Returned JSON exceeds 64 KiB"}
        try:
            return json.loads(data)
        except (ValueError, UnicodeError):
            return {"error": "Candidate exited without a JSON result"}
    finally:
        os.close(read_fd)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


if __name__ == "__main__":
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        print(json.dumps(execute(request), allow_nan=False), flush=True)
