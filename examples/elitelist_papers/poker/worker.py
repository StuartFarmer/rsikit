"""Container-only table supervisor; each seat executes under a separate UID."""

import json
import multiprocessing as mp
import os
import socket
import tempfile
from time import monotonic

import gymnasium as gym

from poker.game import CandidateFailure, derived_seed, run_block
from rsikit.sandbox import worker as sandbox_worker
from rsikit.sandbox.codec import encode_space, frame_size, pack, unpack
from rsikit.sandbox.worker import run_candidate


def run_seat(channel, uid, directory, process_group):
    """Fork from a clean server which never receives deals, sources or observations."""
    if process_group is None:
        os.setsid()
    else:
        os.setpgid(0, process_group)
    private = tempfile.mkdtemp(prefix="seat-", dir=directory)
    os.chown(private, uid, uid)
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    # Compile generated code only in this private player process. Retaining the
    # class preserves Numba dispatchers; each hand still gets a fresh instance.
    original_load = sandbox_worker.load_policy
    source_seen, solution = None, None

    def load_once(source, observation_space, action_space, instructions):
        nonlocal source_seen, solution
        if source != source_seen:
            policy = original_load(source, observation_space, action_space, instructions)
            source_seen, solution = source, type(policy)
            return policy
        return solution(observation_space, action_space, instructions=instructions)

    sandbox_worker.load_policy = load_once
    sandbox_worker.block_connections(keep_process_group=True)
    channel.sendall(pack({"ready": True}))
    run_candidate(channel, directory=private)


def receive(channel, size, deadline):
    data = bytearray()
    while len(data) < size:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Policy response deadline exceeded")
        channel.settimeout(remaining)
        chunk = channel.recv(size - len(data))
        if not chunk:
            raise OSError("Policy exited without a response")
        data.extend(chunk)
    return bytes(data)


class Players:
    def __init__(
        self,
        sources,
        timeout,
        stack,
        instructions,
        *,
        policy_ms=5,
        warmup_timeout=15,
        uid_base=1000,
        directory=None,
        process_group=None,
        emit=None,
    ):
        self.sources, self.timeout = sources, timeout
        self.stack, self.instructions = stack, instructions
        self.processes, self.channels, self.occupants = [], [], []
        mp.set_forkserver_preload(["numba", "poker.worker"])
        self.context = mp.get_context("forkserver")
        self.policy_ms, self.warmup_timeout = policy_ms, warmup_timeout
        self.seconds = [0.0] * len(sources)
        self.calls = [0] * len(sources)
        self.warmup_seconds = 0.0
        self.started = self.last_progress = monotonic()
        self.hands, self.total = 0, 0
        self.phase = "starting"
        self.warmed = set()
        self.stream_progress = False
        self.uid_base, self.rotation = uid_base, 0
        self.directory, self.process_group, self.emit = directory, process_group, emit

    def progress(self, hands=None, total=None, *, force=False):
        if hands is not None:
            self.hands, self.total = hands, total
        if not self.stream_progress:
            return
        now = monotonic()
        if not force and now - self.last_progress < 5:
            return
        self.last_progress = now
        means = [1000 * seconds / max(calls, 1) for seconds, calls in zip(self.seconds, self.calls)]
        slow = max(range(len(means)), key=means.__getitem__)
        message = {
            "progress": dict(
                hands=self.hands,
                total=self.total,
                elapsed=now - self.started,
                phase=self.phase,
                slowest=slow,
                slowest_ms=means[slow],
                warmup_seconds=self.warmup_seconds,
            )
        }
        if self.emit:
            self.emit(message)
        else:
            print(json.dumps(message), flush=True)

    def close(self):
        for channel in self.channels:
            channel.close()
        for process in self.processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join()
            process.close()
        self.processes, self.channels = [], []

    def start_rotation(self, occupants):
        self.close()
        self.occupants = occupants
        self.warmed.clear()
        self.phase = "starting players / JIT warm-up"
        self.progress(force=True)
        for seat in range(len(occupants)):
            parent, child = socket.socketpair()
            parent.settimeout(self.timeout)
            uid = self.uid_base + self.rotation * 6 + seat
            process = self.context.Process(
                target=run_seat, args=(child, uid, self.directory, self.process_group)
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                process.close()
                raise
            finally:
                child.close()
            self.channels.append(parent)
            self.processes.append(process)
        self.rotation += 1
        # Startup imports have their own deadline, before any generated code runs.
        deadline = monotonic() + 30
        for channel in self.channels:
            size = frame_size(receive(channel, 4, deadline))
            if unpack(receive(channel, size, deadline)) != {"ready": True}:
                raise RuntimeError("Policy process failed to start")

    def exchange(self, seat, request):
        channel = self.channels[seat]
        key = seat, request["command"]
        warming = key not in self.warmed
        timeout = self.warmup_timeout if warming else self.timeout
        started = monotonic()
        deadline = started + timeout

        try:
            channel.settimeout(timeout)
            channel.sendall(pack(request))
            size = frame_size(receive(channel, 4, deadline))
            result = unpack(receive(channel, size, deadline))
            expected = "action" if request["command"] == "act" else "ok"
            if not isinstance(result, dict) or set(result) != {expected}:
                raise ValueError(str(result)[:2000])
            if expected == "ok" and result["ok"] is not True:
                raise ValueError("Malformed policy acknowledgment")
            elapsed = monotonic() - started
            player = self.occupants[seat]
            if warming:
                self.warmed.add(key)
                self.warmup_seconds += elapsed
            else:
                self.seconds[player] += elapsed
                self.calls[player] += 1
                # One second of grace absorbs scheduler jitter and short bursts.
                if self.seconds[player] > max(1.0, self.calls[player] * self.policy_ms / 1000):
                    raise CandidateFailure(
                        seat,
                        f"Speed budget exceeded: {1000 * self.seconds[player] / self.calls[player]:.2f} "
                        f"ms/call over {self.calls[player]} calls; target {self.policy_ms:g} ms. "
                        "Reduce simulation work or use numba.njit(cache=False) numeric kernels; "
                        "warm them up in reset. Compilation warm-up is excluded.",
                    )
            self.phase = "playing"
            self.progress()
            return result.get("action")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            phase = "warm-up" if warming else request["command"]
            raise CandidateFailure(
                seat, f"{phase}: {str(exc) or 'Policy response timed out'}"
            ) from exc

    def reset(self, seed):
        # Recreate instances, retaining module code and compiled kernels in each private process.
        for seat, player in enumerate(self.occupants):
            self.exchange(
                seat,
                {
                    "command": "start",
                    "source": self.sources[player],
                    "observation_space": encode_space(gym.spaces.Dict({})),
                    "action_space": encode_space(gym.spaces.Discrete(self.stack + 2, start=-1)),
                    "instructions": self.instructions,
                    "seed": derived_seed(seed, seat),
                },
            )

    def act(self, seat, observation):
        return self.exchange(seat, {"command": "act", "observation": observation})


def run_request(request, *, uid_base=1000, directory=None, process_group=None, emit=None):
    players = Players(
        request["sources"],
        request["call_timeout"],
        request["stack"],
        request["instructions"],
        policy_ms=request.get("policy_ms", 5),
        warmup_timeout=request.get("warmup_timeout", 15),
        uid_base=uid_base,
        directory=directory,
        process_group=process_group,
        emit=emit,
    )
    players.total = len(request["sources"]) * request["deals"]
    players.stream_progress = request.get("stream_progress", False)
    try:
        try:
            result = run_block(
                len(request["sources"]),
                request["deals"],
                request["seed"],
                players.act,
                stack=request["stack"],
                start_rotation=players.start_rotation,
                reset=players.reset,
                progress=players.progress,
            )
        except CandidateFailure as exc:
            player = players.occupants[exc.seat]
            result = {"failure": player, "error": str(exc)}
        players.progress(force=True)
        result["timings"] = dict(
            policy_seconds=players.seconds,
            policy_calls=players.calls,
            warmup_seconds=players.warmup_seconds,
        )
        result.update(uid_base=uid_base, worker_pid=os.getpid())
        return result
    finally:
        players.close()


def serve_table(channel, request, uid_base, directory):
    """Trusted table and all its players share a killable group, never the forkserver."""
    os.setpgid(0, 0)

    def emit(value):
        channel.sendall(json.dumps(value, allow_nan=False).encode() + b"\n")

    with channel:
        try:
            result = run_request(
                request,
                uid_base=uid_base,
                directory=directory,
                process_group=os.getpid(),
                emit=emit,
            )
            emit({"result": result})
        except Exception as exc:
            emit({"error": f"{type(exc).__name__}: {exc}"})


def probe(channel):
    with channel:
        channel.sendall(json.dumps({"forkserver_pid": os.getppid()}).encode() + b"\n")
