"""Pinned, headless Ocean 2048. Cooperative policies only; this is not a sandbox."""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import platform
import shlex
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

UPSTREAM = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
SOURCE = Path(__file__).with_suffix("")
FLAGS = ("-O3", "-std=c11", "-D_DEFAULT_SOURCE", "-fPIC", "-shared")
ENDINGS = (None, "game_over", "native_timeout", "external_cap")


@lru_cache(maxsize=1)
def build() -> Path:
    """Build once per source/compiler/platform in the local temporary directory."""
    compiler = shlex.split(os.environ.get("CC", "cc"))
    version = subprocess.check_output([*compiler, "--version"])
    digest = hashlib.sha256(version + repr((compiler, FLAGS, platform.platform())).encode())
    for source in sorted(SOURCE.glob("*")):
        if source.suffix in (".c", ".h"):
            digest.update(source.read_bytes())
    cache = Path(tempfile.gettempdir()) / f"rsikit-ocean-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    target = cache / (digest.hexdigest() + (".dylib" if platform.system() == "Darwin" else ".so"))
    if not target.exists():
        # Separate temporary outputs make simultaneous worker builds safe.
        fd, temporary = tempfile.mkstemp(dir=cache, suffix=target.suffix)
        os.close(fd)
        try:
            subprocess.run(
                [*compiler, *FLAGS, str(SOURCE / "adapter.c"), "-o", temporary],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


class Batch:
    """Independent native games; step takes one action for each active slot.

    Observations retain all original slots. Native endings expose upstream's
    auto-reset observation; completed results come from its log, never that board.
    """

    def __init__(self, seeds, max_steps=2000):
        self._handle = None
        self.seeds = [operator.index(seed) for seed in seeds]
        self.max_steps = operator.index(max_steps)
        if not self.seeds or any(seed < 0 or seed > 0xFFFFFFFF for seed in self.seeds):
            raise ValueError("seeds must be a nonempty sequence of uint32 integers")
        if not 1 <= self.max_steps <= 0x7FFFFFFF:
            raise ValueError("max_steps must be a positive int32 integer")
        self.observations = np.zeros((len(self.seeds), 16), dtype=np.float32)
        self._stats = np.zeros((len(self.seeds), 5), dtype=np.float64)
        self.active = np.arange(len(self.seeds), dtype=np.int64)
        self._lib = ctypes.CDLL(str(build()))
        self._lib.ocean_create.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.ocean_create.restype = ctypes.c_void_p
        self._lib.ocean_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ocean_step.restype = None
        self._lib.ocean_close.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.ocean_close.restype = None
        seeds_array = np.asarray(self.seeds, dtype=np.uint32)
        self._handle = self._lib.ocean_create(
            len(self.seeds), seeds_array.ctypes.data, self.observations.ctypes.data
        )
        if not self._handle:
            raise MemoryError("native Ocean allocation failed")

    def step(self, actions) -> np.ndarray:
        if self._handle is None:
            raise RuntimeError("batch is closed")
        actions = np.asarray(actions)
        if actions.shape != (len(self.active),) or actions.dtype.kind not in "iuf":
            raise ValueError("actions must be a numeric vector with one entry per active row")
        if (
            not np.isfinite(actions).all()
            or np.any(actions < 0)
            or np.any(actions > 3)
            or np.any(actions != np.floor(actions))
        ):
            raise ValueError("actions must be finite integers in [0, 3]")
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        rewards = np.empty(len(self.active), dtype=np.float32)
        self._lib.ocean_step(
            self._handle,
            len(self.active),
            self.active.ctypes.data,
            actions.ctypes.data,
            self.max_steps,
            rewards.ctypes.data,
            self._stats.ctypes.data,
        )
        self.active = np.flatnonzero(self._stats[:, 4] == 0).astype(np.int64, copy=False)
        return rewards

    @property
    def results(self) -> list[dict]:
        return [
            dict(
                seed=seed,
                score=int(row[0]),
                max_tile=int(row[1]),
                **{"return": float(row[2])},
                steps=int(row[3]),
                ending=ENDINGS[int(row[4])],
            )
            for seed, row in zip(self.seeds, self._stats)
            if row[4]
        ]

    def close(self):
        if self._handle is not None:
            self._lib.ocean_close(self._handle, len(self.seeds))
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class OceanEnv(gym.Env):
    """Scalar Gym adapter to the same batch policy contract, with lazy C state."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(0, 255, shape=(1, 16), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([4])
        self._batch = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32))
        self._batch = Batch([seed], self.max_steps)
        return self._batch.observations.copy(), {}

    def step(self, action):
        if self._batch is None or not len(self._batch.active):
            raise RuntimeError("reset is required before stepping")
        reward = float(self._batch.step(action)[0])
        results = self._batch.results
        info = (
            results[0]
            if results
            else {
                "score": int(self._batch._stats[0, 0]),
                "steps": int(self._batch._stats[0, 3]),
                "ending": None,
            }
        )
        ending = info["ending"]
        return (
            self._batch.observations.copy(),
            reward,
            ending == "game_over",
            ending in ("native_timeout", "external_cap"),
            info,
        )

    def close(self):
        if self._batch is not None:
            self._batch.close()
            self._batch = None
