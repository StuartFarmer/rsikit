"""Fixed-work Ocean sweep: OUTPUT [WORKERS ...] [--seeds SEED ...]."""

# Thread limits must be set before importing NumPy or the evaluator, including in spawned children.
# ruff: noqa: E402

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

from examples.benchmark_ocean import initialize, save, summarize
from research.ocean.baselines import policies
from research.ocean.environment import metadata
from research.ocean.evaluator import PanelEvaluator


async def main(output, worker_counts, seeds):
    if len(set(worker_counts)) != len(worker_counts) or any(w < 1 or w > 16 for w in worker_counts):
        raise ValueError("Worker counts must be unique integers from 1 to 16")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonempty, unique nonnegative integers")
    transitions_per_policy = len(seeds) * 2000
    output.mkdir(parents=True, exist_ok=False)
    candidates = policies()
    orders = [worker_counts, worker_counts[::-1], worker_counts[1::2] + worker_counts[::2]]
    report = dict(
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        platform=platform.platform(),
        python=sys.version,
        numpy=np.__version__,
        host_cpu_count=os.cpu_count(),
        resource_budget_enforced=False,
        thread_limits={
            k: os.environ[k]
            for k in os.environ
            if k
            in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        upstream=metadata("g2048"),
        workload=dict(
            mode="batch",
            batch_size=32,
            seeds=seeds,
            max_steps=2000,
            score_key="merge_score",
            jobs_per_trial=16,
            max_transitions_per_policy=transitions_per_policy,
            max_transitions_per_trial=16 * transitions_per_policy,
            cache_enabled=False,
            candidate_order="eight baselines in order, twice",
        ),
        orders=orders,
        policies=[
            dict(
                id=p.id,
                name=p.name,
                source_sha256=hashlib.sha256(p._implementation.encode()).hexdigest(),
            )
            for p in candidates
        ],
        trials=[],
    )
    save(output / "report.json", report)
    expected = {}
    for repeat, order in enumerate(orders, 1):
        for workers in order:
            directory = output / f"repeat-{repeat}-workers-{workers}"
            print(f"Starting repeat {repeat}, workers={workers}", flush=True)
            async with PanelEvaluator(
                directory, mode="batch", workers=workers, batch_size=32, max_steps=2000
            ) as panel:
                cold = await initialize(panel, candidates[0], seeds)
                wall_start, start = time.time(), time.perf_counter()
                rows = await asyncio.gather(
                    *(
                        panel.submit(candidates[j % 8], seeds=seeds, job_id=f"job-{j}")
                        for j in range(16)
                    )
                )
                elapsed, wall_elapsed = time.perf_counter() - start, time.time() - wall_start
                result = summarize(rows, elapsed)
                result.update(
                    repeat=repeat,
                    workers=workers,
                    cold=cold,
                    wall_clock_seconds=wall_elapsed,
                    wall_minus_monotonic_seconds=wall_elapsed - elapsed,
                )
                save(directory / "events.json", panel.events)
                save(directory / "result.json", result)
                assert result["valid_panels"] == 16 and result["failed_panels"] == 0, result
                for row in rows:
                    assert 0 < row["steps"] <= transitions_per_policy, row
                    assert [r["seed"] for r in row["results"]] == seeds, row
                    assert all(0 < r["steps"] <= 2000 for r in row["results"]), row
                    digest = hashlib.sha256(
                        json.dumps(row["results"], sort_keys=True).encode()
                    ).hexdigest()
                    assert expected.setdefault(row["policy_id"], digest) == digest, row
                result["all_result_rows_match"] = True
                report["trials"].append(result)
                save(output / "report.json", report)
                print(
                    f"Done: {elapsed:.3f}s, {16 / elapsed:.3f} policies/s; results match",
                    flush=True,
                )
    report["summary"] = []
    for workers in worker_counts:
        trials = [r for r in report["trials"] if r["workers"] == workers]
        rates = [r["valid_panels_per_second"] for r in trials]
        report["summary"].append(
            dict(
                workers=workers,
                median_policies_per_second=statistics.median(rates),
                min_policies_per_second=min(rates),
                max_policies_per_second=max(rates),
                median_seconds=statistics.median(r["elapsed_seconds"] for r in trials),
            )
        )
    report["validated_panels"] = len(report["trials"]) * 16
    report["all_measured_panels_valid_and_matching"] = True
    save(output / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("workers", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    asyncio.run(main(args.output, args.workers, args.seeds))
