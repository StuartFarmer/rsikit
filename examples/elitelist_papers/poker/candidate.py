"""Private poker seat process; hidden-card isolation requires a separate player."""

import asyncio
import ctypes
import os
import resource

from poker.codec import decode_space, frame_size, pack, unpack
from rsikit.policy import load_policy


def block_connections(*, keep_process_group=False):
    """Block control connections and optionally escape from the episode process group."""
    arch, connect, setpgid, setsid = {
        "aarch64": (0xC00000B7, 203, 154, 157),
        "x86_64": (0xC000003E, 42, 109, 112),
    }[os.uname().machine]

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    denied = 0x00050001  # SECCOMP_RET_ERRNO | EPERM
    instructions = [
        Filter(0x20, 0, 0, 4),  # Load seccomp_data.arch.
        Filter(0x15, 1, 0, arch),
        Filter(0x06, 0, 0, denied),  # Reject alternate syscall architectures.
        Filter(0x20, 0, 0, 0),  # Load seccomp_data.nr.
        Filter(0x35, 0, 1, 0x40000000),  # Reject the x32 ABI too.
        Filter(0x06, 0, 0, denied),
    ]
    # io_uring_setup can bypass connect(). Descendants inherit this filter.
    syscalls = (connect, 425) + ((setpgid, setsid) if keep_process_group else ())
    for syscall in syscalls:
        instructions.extend((Filter(0x15, 0, 1, syscall), Filter(0x06, 0, 0, denied)))
    instructions.append(Filter(0x06, 0, 0, 0x7FFF0000))  # SECCOMP_RET_ALLOW
    rules = (Filter * len(instructions))(*instructions)
    libc = ctypes.CDLL(None, use_errno=True)
    program = Program(len(rules), rules)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


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
