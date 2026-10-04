"""Split warm evaluation cost; python -m examples.benchmark_evaluator --output report.json."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import tempfile
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import gymnasium as gym

from research.rewards import mean_rewards
from research.rollouts import Rollouts
from rsikit import Episode, Executor, Run
from rsikit.envs import BitcoinEnv, BlackjackEnv, CirclePackingEnv
from rsikit.evaluation import PolicyError
from rsikit.policy import Policy

PACKING = """
import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
"""
CARTPOLE = Path(__file__).with_name("cartpole.py")


class TimedEnvironment(gym.Wrapper):
    def reset(self, **kwargs):
        self.started = perf_counter()
        self.seconds = 0.0
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        start = perf_counter()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.seconds += perf_counter() - start
        self.steps += 1
        if terminated or truncated:
            info = dict(info)
            info["artifacts"] = {
                **info.get("artifacts", {}),
                "environment-timing.json": json.dumps(
                    dict(seconds=self.seconds, steps=self.steps, started=self.started)
                ).encode(),
            }
        return observation, reward, terminated, truncated, info


TIMING = """
import json
from pathlib import Path
from time import perf_counter
Original = Solution
class Solution(Original):
    async def reset(self, **kwargs):
        self.seconds = 0.0
        self.steps = 0
        await super().reset(**kwargs)
    async def act(self, observation):
        start = perf_counter()
        action = await super().act(observation)
        self.seconds += perf_counter() - start
        self.steps += 1
        return action
    async def close(self):
        await super().close()
        Path("policy-timing.json").write_text(json.dumps(dict(seconds=self.seconds, steps=self.steps)))
"""


async def scheduling(samples):
    """Synthetic arrivals and uneven seeds, using real Run persistence and Executor."""
    report = {}
    for mode in ("batch_barrier", "continuous"):
        values = []
        for _ in range(samples):
            arrived, waits, durations = {}, [], []
            active = peak = 0

            class TimedExecutor(Executor):
                async def _evaluate(self, implementation, environment, seed):
                    nonlocal active, peak
                    start = perf_counter()
                    waits.append(start - arrived[implementation])
                    active += 1
                    peak = max(peak, active)
                    try:
                        await asyncio.sleep(0.08 if seed == 0 else 0.008)
                        return Episode([0, 1], [0], [0.5], [True], [False], [{}, {}])
                    finally:
                        active -= 1
                        durations.append(perf_counter() - start)

            with tempfile.TemporaryDirectory() as directory:
                async with (
                    TimedExecutor(concurrency=4) as executor,
                    Run.create(
                        name="benchmark",
                        export=False,
                        path=Path(directory) / "run",
                    ) as run,
                ):
                    rollouts = Rollouts(CirclePackingEnv(1), executor, run)
                    queue = asyncio.Queue()

                    async def produce():
                        for i in range(12):
                            await asyncio.sleep(0.005)
                            source = PACKING + f"\n# candidate {i}\n"
                            arrived[source] = perf_counter()
                            await queue.put(Policy.from_text(source, name=str(i)))
                        await queue.put(None)

                    async def consume():
                        tasks = []
                        try:
                            while (policy := await queue.get()) is not None:
                                if mode == "batch_barrier":
                                    batch = [policy]
                                    finished = False
                                    while not queue.empty():
                                        policy = queue.get_nowait()
                                        if policy is None:
                                            finished = True
                                            break
                                        batch.append(policy)
                                    await mean_rewards(rollouts, batch, seeds=(0, 1))
                                    if finished:
                                        break
                                else:
                                    tasks.append(
                                        asyncio.create_task(
                                            mean_rewards(rollouts, [policy], seeds=(0, 1))
                                        )
                                    )
                            await asyncio.gather(*tasks)
                        finally:
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)

                    start = perf_counter()
                    await asyncio.gather(produce(), consume())
                    elapsed = perf_counter() - start
                    assert len(durations) == 24 and peak <= 4
                    assert all(run.scores(p) == {0: 0.5, 1: 0.5} for p in run.policies())
                    values.append(
                        dict(
                            seconds=elapsed,
                            candidates_per_second=12 / elapsed,
                            mean_queue_seconds=statistics.mean(waits),
                            worker_utilization=sum(durations) / (4 * elapsed),
                            peak_workers=peak,
                            arrival_rate=11 / (max(arrived.values()) - min(arrived.values())),
                        )
                    )
        report[mode] = dict(
            samples=values,
            medians={key: statistics.median(row[key] for row in values) for key in values[0]},
        )
    return report


async def main(samples, output):
    report = {
        "backend": "local episode processes",
        "host": platform.platform(),
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("rsikit/execution.py"),
                Path("rsikit/run.py"),
                Path("Dockerfile"),
            )
        },
        "image": os.environ.get("RSIKIT_IMAGE_ID"),
        "scope": "First episode includes clean forkserver startup; subsequent episodes use fresh children. Residual includes startup, reset/close, validation and artifacts.",
        "workloads": {},
    }
    workloads = [
        ("packing", CirclePackingEnv(1), PACKING),
        ("cartpole", gym.make("CartPole-v1"), CARTPOLE.read_text()),
        (
            "blackjack",
            BlackjackEnv(),
            "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, o): return 0 if o[0] == 0 or o[1] >= 17 else 1\n",
        ),
        (
            "bitcoin",
            BitcoinEnv(),
            "import numpy as np\nfrom rsikit import Policy\nclass Solution(Policy):\n    async def act(self, o): return np.array([0.5], dtype=np.float64)\n",
        ),
    ]
    async with AsyncExitStack() as stack:
        executor = await stack.enter_async_context(Executor())
        for label, environment, source in workloads:
            env = TimedEnvironment(environment)
            values = []
            for repeat in range(samples + 1):
                start = perf_counter()
                results = [
                    r async for _, _, r in executor.evaluate([(label, source + TIMING, 1)], env)
                ]
                elapsed = perf_counter() - start
                result = results[0]
                if result.error is not None:
                    raise PolicyError(result.error)
                env_time = json.loads(result.artifacts["environment-timing.json"])
                policy_time = json.loads(result.artifacts["policy-timing.json"])
                assert env_time["steps"] == policy_time["steps"] > 0
                if values:
                    assert result.total_reward == values[0]["score"]
                if not repeat:
                    first_seconds = elapsed
                if repeat:
                    values.append(
                        dict(
                            seconds=elapsed,
                            startup_seconds=env_time["started"] - start,
                            environment_seconds=env_time["seconds"],
                            policy_seconds=policy_time["seconds"],
                            steps=env_time["steps"],
                            score=result.total_reward,
                        )
                    )
            medians = {key: statistics.median(row[key] for row in values) for key in values[0]}
            medians["other_seconds"] = (
                medians["seconds"] - medians["environment_seconds"] - medians["policy_seconds"]
            )
            report["workloads"][label] = dict(
                first_seconds=first_seconds, medians=medians, samples=values
            )
            print(label, json.dumps(medians), flush=True)
            env.close()
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["synthetic_scheduling"] = await scheduling(samples)
    print("synthetic_scheduling", json.dumps(report["synthetic_scheduling"]), flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluator-benchmark.json"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(main(args.samples, args.output))
