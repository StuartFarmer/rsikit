"""Container-only supervisor and one persistent candidate process per episode."""

import asyncio
import os
import resource
import socket
import sys

from rsikit.policy import Policy
from rsikit.sandbox.codec import MAX_MESSAGE, decode, decode_space, dumps, encode, loads


async def candidate(channel):
    policy = None
    while line := channel.readline(MAX_MESSAGE + 2):
        request = loads(line)
        try:
            command = request["command"]
            if command == "start":
                namespace = {"__name__": "candidate"}
                exec(compile(request["source"], "candidate.py", "exec"), namespace)
                solution = namespace["Solution"]
                if not isinstance(solution, type) or not issubclass(solution, Policy):
                    raise TypeError("Solution must subclass rsikit.policy.Policy")
                policy = solution(
                    decode_space(request["observation_space"]),
                    decode_space(request["action_space"]),
                    instructions=request["instructions"],
                )
                await policy.reset(seed=request["seed"])
                response = {"ok": True}
            elif command == "act":
                response = {"action": encode(await policy.act(decode(request["observation"])))}
            elif command == "close":
                if policy is not None:
                    await policy.close()
                response = {"ok": True}
            else:
                raise ValueError("Unknown command")
            output = dumps(response)
        except BaseException as exc:
            output = dumps({"error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
        channel.write(output)
        channel.flush()
        if request["command"] == "close":
            return


def main():
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with open(os.devnull, "r+b", buffering=0) as sink:
            for fd in (0, 1, 2):
                os.dup2(sink.fileno(), fd)
        try:
            with child.makefile("rwb") as channel:
                asyncio.run(candidate(channel))
        finally:
            os._exit(0)
    child.close()
    try:
        with parent.makefile("rwb") as channel:
            sys.stdout.buffer.write(dumps({"ready": True}))
            sys.stdout.buffer.flush()
            while line := sys.stdin.buffer.readline(MAX_MESSAGE + 2):
                request = loads(line)
                try:
                    channel.write(dumps(request))
                    channel.flush()
                    response = loads(channel.readline(MAX_MESSAGE + 2))
                    if not isinstance(response, dict) or set(response) not in (
                        {"ok"},
                        {"action"},
                        {"error"},
                    ):
                        raise ValueError("Malformed candidate response")
                except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                    response = {"error": f"Candidate transport failed: {str(exc)[:2000]}"}
                sys.stdout.buffer.write(dumps(response))
                sys.stdout.buffer.flush()
                if request["command"] == "close":
                    break
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
