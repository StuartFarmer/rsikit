"""Split warm evaluation cost; python -m examples.benchmark_evaluator --output report.json."""

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import tempfile
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import gymnasium as gym

from examples.benchmark_docker import CARTPOLE, PACKING, command
from rsikit import Executor, Run
from rsikit.envs import BitcoinEnv, BlackjackEnv, CirclePackingEnv
from rsikit.policy import _policy_class
from rsikit.sandbox.docker import DockerSandbox, InProcessDockerSandbox


class TimedEnvironment(gym.Wrapper):
    def reset(self, **kwargs):
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
                    dict(seconds=self.seconds, steps=self.steps)
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

            class Sandbox:
                async def start(self, workers):
                    pass

                async def close(self):
                    pass

                async def evaluate(self, implementation, environment, seed, call_timeout):
                    nonlocal active, peak
                    start = perf_counter()
                    waits.append(start - arrived[implementation])
                    active += 1
                    peak = max(peak, active)
                    try:
                        await asyncio.sleep(0.08 if seed == 0 else 0.008)
                        return 0.5, {}
                    finally:
                        active -= 1
                        durations.append(perf_counter() - start)

            with tempfile.TemporaryDirectory() as directory:
                async with Run.create(
                    name="benchmark",
                    environment=CirclePackingEnv(1),
                    export=False,
                    path=Path(directory) / "run",
                    executor=Executor(sandbox=Sandbox(), concurrency=4),
                ) as run:
                    queue = asyncio.Queue()

                    async def produce():
                        for i in range(12):
                            await asyncio.sleep(0.005)
                            source = PACKING + f"\n# candidate {i}\n"
                            arrived[source] = perf_counter()
                            await queue.put(_policy_class(str(i), source))
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
                                    await run.evaluate(batch, seeds=(0, 1))
                                    if finished:
                                        break
                                else:
                                    tasks.append(
                                        asyncio.create_task(run.evaluate([policy], seeds=(0, 1)))
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


async def main(samples, output, image, compare_image=None, in_process=False):
    backend = InProcessDockerSandbox if in_process else DockerSandbox
    report = {
        "backend": backend.__name__,
        "host": platform.platform(),
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("rsikit/execution.py"),
                Path("rsikit/run.py"),
                *sorted(Path("rsikit/sandbox").glob("*.py")),
            )
        },
        "image": (
            await command("docker", "image", "inspect", image, "--format", "{{.Id}}")
        ).strip(),
        "scope": "Warm sandbox; instrumented env.step and policy.act wall time. Residual includes IPC, validation, startup of children, reset/close and artifacts, not just transport.",
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
        executor = await stack.enter_async_context(Executor(sandbox=backend(image=image)))
        if compare_image:
            reference = await stack.enter_async_context(
                Executor(sandbox=DockerSandbox(image=compare_image))
            )
            report["compare_image"] = (
                await command("docker", "image", "inspect", compare_image, "--format", "{{.Id}}")
            ).strip()
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
                env_time = json.loads(result.artifacts["environment-timing.json"])
                policy_time = json.loads(result.artifacts["policy-timing.json"])
                assert env_time["steps"] == policy_time["steps"] > 0
                if values:
                    assert result.score == values[0]["score"]
                if repeat:
                    values.append(
                        dict(
                            seconds=elapsed,
                            environment_seconds=env_time["seconds"],
                            policy_seconds=policy_time["seconds"],
                            steps=env_time["steps"],
                            score=result.score,
                        )
                    )
            medians = {key: statistics.median(row[key] for row in values) for key in values[0]}
            medians["other_seconds"] = (
                medians["seconds"] - medians["environment_seconds"] - medians["policy_seconds"]
            )
            report["workloads"][label] = dict(medians=medians, samples=values)
            print(label, json.dumps(medians), flush=True)
            if compare_image:
                times = {"baseline": [], "optimized": []}
                backends = {"baseline": reference, "optimized": executor}
                for repeat in range(samples + 1):
                    for name in list(backends) if repeat % 2 else list(reversed(backends)):
                        start = perf_counter()
                        results = [
                            r
                            async for _, _, r in backends[name].evaluate(
                                [(label, source, 1)], environment
                            )
                        ]
                        elapsed = perf_counter() - start
                        assert len(results) == 1 and results[0].score == medians["score"]
                        assert not results[0].artifacts
                        if repeat:
                            times[name].append(elapsed)
                paired = dict(
                    samples=times,
                    median_seconds={name: statistics.median(v) for name, v in times.items()},
                )
                paired["speedup"] = (
                    paired["median_seconds"]["baseline"] / paired["median_seconds"]["optimized"]
                )
                report["workloads"][label]["paired_uninstrumented"] = paired
                print(label, "paired", json.dumps(paired), flush=True)
            env.close()
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["synthetic_scheduling"] = await scheduling(samples)
    print("synthetic_scheduling", json.dumps(report["synthetic_scheduling"]), flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evaluator-benchmark.json"))
    parser.add_argument("--image", default="rsikit-sandbox:local")
    parser.add_argument(
        "--in-process", action="store_true", help="Run policy and environment in one Docker process"
    )
    parser.add_argument(
        "--compare-image", help="Alternate uninstrumented episodes against this baseline image"
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(main(args.samples, args.output, args.image, args.compare_image, args.in_process))
