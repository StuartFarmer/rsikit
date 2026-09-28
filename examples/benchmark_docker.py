"""Measure cached-image Docker latency; run with python -m examples.benchmark_docker."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cloudpickle
import gymnasium as gym

from rsikit.envs import CirclePackingEnv
from rsikit.execution import Executor
from rsikit.sandbox import run_program
from rsikit.sandbox.docker import DockerSandbox

PACKING = """
import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
"""
CARTPOLE = Path(__file__).with_name("cartpole.py")


async def command(*args):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace"))
    return stdout.decode()


async def main(samples, output):
    report = {
        "date": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("gymnasium", "numpy", "cloudpickle")
        },
        "image": json.loads(await command("docker", "image", "inspect", "rsikit-sandbox:local"))[0][
            "Id"
        ],
        "docker": json.loads(
            await command(
                "docker",
                "info",
                "--format",
                '{"version":"{{.ServerVersion}}","cpus":{{.NCPU}},'
                '"memory":{{.MemTotal}},"running":{{.ContainersRunning}}}',
            )
        ),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("rsikit/execution.py"),
                Path("rsikit/episode.py"),
                Path("rsikit/evaluation.py"),
                Path("rsikit/policy.py"),
                *sorted(Path("rsikit/sandbox").glob("*.py")),
            ]
        },
        "measurements": {},
    }

    def record(name, values, jobs=1):
        result = {
            "seconds": values,
            "median_ms": statistics.median(values) * 1000,
            "min_ms": min(values) * 1000,
            "max_ms": max(values) * 1000,
            "jobs_per_sample": jobs,
            "jobs_per_second": jobs / statistics.median(values),
        }
        report["measurements"][name] = result
        print(
            f"{name}: {result['median_ms']:.1f} ms median; {result['jobs_per_second']:.2f} jobs/s",
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")

    async def measure(name, operation, count=samples, jobs=1):
        values = []
        for _ in range(count):
            start = perf_counter()
            await operation()
            values.append(perf_counter() - start)
        record(name, values, jobs)

    env = CirclePackingEnv(count=1)
    definition = cloudpickle.dumps(env)

    async def packing(sandbox):
        score, artifacts = await sandbox.evaluate(PACKING, definition, 1, 10)
        assert score == 0.5 and artifacts == {}, (score, artifacts)

    starts, evaluations, closes = [], [], []
    for _ in range(samples):
        sandbox = DockerSandbox()
        try:
            start = perf_counter()
            await sandbox.start(1)
            starts.append(perf_counter() - start)
            start = perf_counter()
            await packing(sandbox)
            evaluations.append(perf_counter() - start)
        finally:
            start = perf_counter()
            await sandbox.close()
            closes.append(perf_counter() - start)
    record("container_start", starts)
    record("first_packing_episode", evaluations)
    record("container_close", closes)
    record("cold_packing_total", [sum(parts) for parts in zip(starts, evaluations, closes)])

    sandbox = DockerSandbox()
    try:
        await sandbox.start(1)
        for label, args in (
            ("exec_true", ["true"]),
            ("exec_python", ["python", "-I", "-c", "pass"]),
            (
                "exec_import_evaluator",
                [
                    "python",
                    "-I",
                    "-c",
                    "import sys; sys.path.insert(0, '/opt/worker'); import rsikit.sandbox.evaluate",
                ],
            ),
        ):

            async def execute(args=args):
                await command("docker", "exec", sandbox.name, *args)

            await execute()
            await measure(label, execute)
        await packing(sandbox)
        await measure("warm_packing_episode", lambda: packing(sandbox))
    finally:
        await sandbox.close()

    workloads = [
        ("packing", PACKING, env, 0.5),
        ("cartpole500", CARTPOLE.read_text(), gym.make("CartPole-v1"), 500.0),
    ]
    for label, source, environment, expected in workloads:
        for concurrency in (1, 4):
            async with Executor(concurrency=concurrency) as executor:

                async def batch(size=16):
                    results = [
                        result
                        async for result in executor.evaluate(
                            [(str(i), source, 1) for i in range(size)], environment
                        )
                    ]
                    assert len(results) == size
                    assert all(
                        result.total_reward == expected and result.artifacts == {}
                        for _, _, result in results
                    )

                await batch(1)
                await measure(f"{label}_batch16_c{concurrency}", batch, count=3, jobs=16)
        environment.close()

    async def legacy():
        episode = await run_program(CARTPOLE, "CartPole-v1", env_seed=1, policy_seed=1)
        info = episode.infos[-1]
        assert info["episode"]["r"] == 500.0 and info["episode"]["l"] == 500

    await legacy()
    await measure("legacy_cartpole500_total", legacy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docker-benchmark.json"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(main(args.samples, args.output))
