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
from time import perf_counter, sleep

import gymnasium as gym

from research.rewards import mean_rewards
from research.rollouts import Rollouts
from rsikit import Executor, Job, Run
from rsikit.envs import BitcoinEnv, BlackjackEnv, CirclePackingEnv
from rsikit.evaluation import PolicyError
from rsikit.policy import PolicyDefinition

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
            intervals = []

            class TimedEnvironment(CirclePackingEnv):
                def reset(self, *, seed=None, options=None):
                    self.started = perf_counter()
                    self.seed = seed
                    return super().reset(seed=seed, options=options)

                def step(self, action):
                    sleep(0.08 if self.seed == 0 else 0.008)
                    obs, _, terminated, truncated, info = super().step(action)
                    info["timing"] = (self.started, perf_counter())
                    return obs, 0.5, terminated, truncated, info

            class TimedExecutor(Executor):
                async def iterate(self, jobs):
                    from contextlib import aclosing

                    async with aclosing(super().iterate(jobs)) as completed:
                        async for job in completed:
                            start, end = job.result.infos[-1]["timing"]
                            waits.append(start - arrived[job.policy.source])
                            durations.append(end - start)
                            intervals.extend(((start, 1), (end, -1)))
                            yield job

            with tempfile.TemporaryDirectory() as directory:
                async with (
                    TimedExecutor(concurrency=4) as executor,
                    Run.create(
                        name="benchmark",
                        export=False,
                        path=Path(directory) / "run",
                    ) as run,
                ):
                    rollouts = Rollouts(TimedEnvironment(1), executor, run)
                    queue = asyncio.Queue()

                    async def produce():
                        for i in range(12):
                            await asyncio.sleep(0.005)
                            source = PACKING + f"\n# candidate {i}\n"
                            arrived[source] = perf_counter()
                            await queue.put(PolicyDefinition.from_text(source, name=str(i)))
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
                    active = peak = 0
                    for _, change in sorted(intervals):
                        active += change
                        peak = max(peak, active)
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
        "backend": "Huey process workers",
        "host": platform.platform(),
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                *sorted(Path("rsikit/execution").glob("*.py")),
                Path("rsikit/run.py"),
                Path("Dockerfile"),
            )
        },
        "image": os.environ.get("RSIKIT_IMAGE_ID"),
        "scope": "Huey reuses worker processes across episodes. Residual includes queue I/O, input copies, reset/close, validation and artifacts.",
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
                    job.result
                    async for job in executor.iterate(
                        [Job(PolicyDefinition(source=source + TIMING, name=label), env, seed=1)]
                    )
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
