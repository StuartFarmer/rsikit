"""Benchmark daily BTC allocation steps, including policy decisions and episode resets.

Run: python -m examples.bitcoin --steps 1000000 --repeats 3
"""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import gymnasium
import numpy as np

from rsikit.envs import BitcoinEnv
from rsikit.policy import Policy


class Solution(Policy):
    """Buy-and-hold starting point, not an optimized investment strategy."""

    async def act(self, observation):
        return np.array([1.0], dtype=np.float64)


def benchmark(steps, repeats, fee_rate):
    samples = []
    # Exercise cash, partial allocations, buys, sells, and fully invested positions.
    actions = tuple(np.array([a], dtype=np.float64) for a in (0, 0.25, 1, 0.75, 0.5))
    env = BitcoinEnv(fee_rate=fee_rate)
    for _ in range(repeats):
        env.reset()
        resets = 0
        reward_sum = 0.0
        start = perf_counter()
        for step in range(steps):
            _, reward, done, _, _ = env.step(actions[step % len(actions)])
            reward_sum += reward
            if done:
                env.reset()
                resets += 1
        seconds = perf_counter() - start
        samples.append(
            dict(
                steps_per_second=steps / seconds,
                seconds=seconds,
                resets=resets,
                reward_sum=reward_sum,
            )
        )
    env.close()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "steps_per_repeat": steps,
        "fee_rate": fee_rate,
        "scope": "In-process env.step, action selection, reward accumulation, and resets; "
        "excludes imports, CSV load, async runner, Docker, and model inference.",
        "median_steps_per_second": median(s["steps_per_second"] for s in samples),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-steps-per-second", type=float, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    result = benchmark(args.steps, args.repeats, args.fee_rate)
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    if result["median_steps_per_second"] < args.min_steps_per_second:
        raise SystemExit("BTC environment throughput below the requested threshold")


if __name__ == "__main__":
    main()
