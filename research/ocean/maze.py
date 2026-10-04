"""Headless pinned Ocean Maze with independent maps and frozen episode endings."""

import ctypes
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

UPSTREAM = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"


def metadata():
    library = Path(sys.prefix) / "share" / "rsikit-maze" / "maze.so"
    if not library.is_file():
        raise ImportError("Build Ocean Maze first: python scripts/install_maze.py")
    return dict(
        environment="maze",
        upstream=UPSTREAM,
        protocol="ocean-maze-episodic-v1",
        native_library=str(library),
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(library.with_name("maze.h").read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(library.with_name("headless.c").read_bytes()).hexdigest(),
        seed_semantics="one generated map per seed; map hashes checked across splits",
    )


@lru_cache(maxsize=1)
def native():
    library = ctypes.CDLL(metadata()["native_library"])
    pointer = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    for name, args, result in (
        ("maze_new", [pointer, ctypes.c_int], ctypes.c_void_p),
        ("maze_reset", [ctypes.c_void_p, ctypes.c_int, pointer], None),
        ("maze_step", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        ("maze_reward", [ctypes.c_void_p], ctypes.c_float),
        ("maze_close", [ctypes.c_void_p], None),
    ):
        function = getattr(library, name)
        function.argtypes, function.restype = args, result
    return library


class Maze:
    def __init__(self, num_envs=1, seed=0, log_interval=2001, map_size=15):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if type(map_size) is not int or not 5 <= map_size <= 47 or map_size % 2 != 1:
            raise ValueError("map_size must be odd and in [5, 47]")
        self.num_agents = num_envs
        self.observations = np.zeros((num_envs, 121), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(0, 4, self.observations.shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([5] * num_envs)
        self.terminals = np.zeros(num_envs, dtype=bool)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=int)
        self.scores = np.zeros(num_envs)
        self.maps = np.zeros((num_envs, 47 * 47), dtype=np.uint8)
        self.handles = []
        try:
            for observation in self.observations:
                handle = native().maze_new(observation, map_size)
                if not handle:
                    raise MemoryError("Maze allocation failed")
                self.handles.append(handle)
        except BaseException:
            self.close()
            raise

    def reset(self, seed=0):
        seeds = [seed] * self.num_agents if type(seed) is int else list(seed)
        if len(seeds) != self.num_agents or any(
            type(s) is not int or not 0 <= s <= 2147483647 for s in seeds
        ):
            raise ValueError("Expected one nonnegative int32 seed per game")
        for i, (handle, value) in enumerate(zip(self.handles, seeds)):
            native().maze_reset(handle, value, self.maps[i])
        self.terminals[:] = False
        self.rewards[:] = self.steps[:] = self.scores[:] = 0
        return self.observations, []

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.dtype.kind not in "iu" or not self.action_space.contains(actions):
            raise ValueError("Expected one discrete Maze action per game")
        self.rewards[:] = 0
        for i, (handle, action) in enumerate(zip(self.handles, actions)):
            if not self.terminals[i]:
                self.terminals[i] = native().maze_step(handle, int(action))
                self.rewards[i] = native().maze_reward(handle)
                self.scores[i] += self.rewards[i]
                self.steps[i] += 1
        return (
            self.observations,
            self.rewards,
            self.terminals,
            np.zeros(self.num_agents, dtype=bool),
            [],
        )

    def episode_stats(self):
        return [
            dict(
                score=float(self.scores[i]),
                perf=float(self.scores[i]),
                episode_return=float(self.scores[i]),
                episode_length=int(self.steps[i]),
                map_id=hashlib.sha256(self.maps[i].tobytes()).hexdigest(),
            )
            for i in range(self.num_agents)
        ]

    def close(self):
        for handle in self.handles:
            native().maze_close(handle)
        self.handles.clear()


def verify_splits(config):
    """Fail before model calls if any map repeats across calibration/search/audit phases."""
    panels = (
        {}
        if getattr(config, "optimizer", "gepa") == "elitetable"
        else {"calibration": range(2_000_000_000, 2_000_000_000 + config.calibration_cases)}
    )
    for phase in ("development", "validation", "test"):
        replicates = getattr(config, phase)
        cases = config.final_audit_cases if phase == "test" else config.audit_cases
        panels[f"{phase}/search"] = [r * 10000 + i for r in replicates for i in range(10)]
        panels[f"{phase}/audit"] = [r * 10000 + 1000 + i for r in replicates for i in range(cases)]
    seen, report = {}, {}
    env = Maze()
    try:
        for panel, seeds in panels.items():
            report[panel] = []
            for seed in seeds:
                env.reset(seed=[seed])
                identity = env.episode_stats()[0]["map_id"]
                if identity in seen:
                    raise ValueError(f"Duplicate Maze map: {panel}/{seed} and {seen[identity]}")
                seen[identity] = f"{panel}/{seed}"
                report[panel].append(dict(seed=seed, map_id=identity))
    finally:
        env.close()
    return report
