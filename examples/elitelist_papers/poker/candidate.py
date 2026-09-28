"""Private poker seat process; hidden-card isolation requires a separate player."""

import asyncio
import os
import resource

from poker.codec import decode_space, frame_size, pack, unpack
from rsikit.sandbox.worker import block_connections, load_policy


async def candidate(channel):
    policy = None
    while header := channel.read(4):
        size = frame_size(header)
        body = channel.read(size)
        if len(body) != size:
            raise ValueError("Incomplete frame")
        request = unpack(body)
        try:
            command = request["command"]
            if command == "start":
                policy = load_policy(
                    request["source"],
                    decode_space(request["observation_space"]),
                    decode_space(request["action_space"]),
                    request["instructions"],
                )
                await policy.reset(seed=request["seed"])
                response = {"ok": True}
            elif command == "act":
                observation = request["observation"]
                action = await policy.act(observation)
                response = {"action": action}
            elif command == "close":
                if policy is not None:
                    await policy.close()
                response = {"ok": True}
            else:
                raise ValueError("Unknown command")
            output = pack(response)
        except BaseException as exc:
            error = {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
            output = pack(error)
        channel.write(output)
        channel.flush()
        if request["command"] == "close":
            return


def run_candidate(channel, directory=None):
    if directory is not None:
        os.chdir(directory)
        # RLIMIT_NPROC alone cannot stop a child asking the forkserver to spawn.
        # Keep its connected policy channel; deny every new control connection.
        block_connections()
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with open(os.devnull, "r+b", buffering=0) as sink:
        for fd in (0, 1, 2):
            os.dup2(sink.fileno(), fd)
    with channel, channel.makefile("rwb") as stream:
        asyncio.run(candidate(stream))
