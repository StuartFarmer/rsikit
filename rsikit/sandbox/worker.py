"""Container-only policy loading and process restrictions."""

import ctypes
import os

from rsikit.policy import Policy
from rsikit.sandbox.codec import MAX_SOURCE


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


def load_policy(source, observation_space, action_space, instructions):
    """Container-only policy loading, used by whole-episode and poker workers."""
    if len(source.encode()) > MAX_SOURCE:
        raise ValueError("Source exceeds 64 KiB")
    namespace = {"__name__": "candidate"}
    exec(compile(source, "candidate.py", "exec"), namespace)
    solution = namespace["Solution"]
    if not isinstance(solution, type) or not issubclass(solution, Policy):
        raise TypeError("Solution must subclass rsikit.Policy")
    return solution(observation_space, action_space, instructions=instructions)
