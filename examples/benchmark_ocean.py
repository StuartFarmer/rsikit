"""Offline Ocean correctness, saturated capacity, and finite arrival replay.

Full pilot commands (no paid calls):
  python -m examples.benchmark_ocean correctness --output runs/ocean-correctness
  python -m examples.benchmark_ocean capacity --matrix --output runs/ocean-capacity
  python -m examples.benchmark_ocean replay --rate 0.1 --output runs/ocean-replay-01
  python -m examples.benchmark_ocean replay --rate 1 --output runs/ocean-replay-1
  python -m examples.benchmark_ocean replay --rate 10 --output runs/ocean-replay-10
Repeat replay with --mode reference, then the selected --workers/--batch-size.
For real arrivals use --trace trace.json; repeat with --speed 1.25 for headroom.
Saved traces must themselves span at least 300 seconds after acceleration to
qualify; --duration controls synthetic traces only and does not tile real traces.

Trace: [{"at": seconds, "policy_id": baseline ID OR "policy": source path}].
Paths are relative to the trace. Optional generation/attempt/llm_start/llm_finish
are retained. A real trace uses {"arrivals": [...], "model": ..., "provider_limits":
..., "generation_concurrency": ..., "active_generation_seconds": ..., optional
"llm_latencies": [...]}. Generation-active duration must exclude evaluation waits.
The runner bounds admitted work and delays the producer, never drops arrivals.
Cooperative candidate execution; this is not an adversarial-code security boundary.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import resource
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rsikit.policy import Policy


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(rows, elapsed):
    successful = [row for row in rows if row["status"] == "ok"]
    service = [row["service_seconds"] for row in successful]
    result = dict(
        elapsed_seconds=elapsed,
        valid_panels=len(successful),
        failed_panels=len(rows) - len(successful),
        valid_panels_per_second=len(successful) / elapsed,
        actual_steps_per_second=sum(row["steps"] for row in successful) / elapsed,
        median_service_seconds=statistics.median(service) if service else None,
        p95_service_seconds=percentile(service, 0.95),
        p95_queue_seconds=percentile([row["queue_seconds"] for row in rows], 0.95),
        per_policy={},
    )
    for policy_id in sorted({row["policy_id"] for row in rows}):
        subset = [row for row in rows if row["policy_id"] == policy_id]
        valid = [row for row in subset if row["status"] == "ok"]
        result["per_policy"][policy_id] = dict(
            valid_panels=len(valid),
            failed_panels=len(subset) - len(valid),
            median_service_seconds=statistics.median([row["service_seconds"] for row in valid])
            if valid
            else None,
            p95_service_seconds=percentile([row["service_seconds"] for row in valid], 0.95),
        )
    return result


def load_trace(path, policies):
    value = json.loads(path.read_text())
    metadata = value if isinstance(value, dict) else {}
    arrivals = metadata.get("arrivals", value)
    if not isinstance(arrivals, list) or not arrivals:
        raise ValueError("Trace must contain a nonempty arrivals list")
    result, previous = [], -1
    for row in arrivals:
        at = row.get("at")
        if (
            isinstance(at, bool)
            or not isinstance(at, (int, float))
            or not math.isfinite(at)
            or at < 0
            or at < previous
        ):
            raise ValueError("Trace arrival times must be finite, nonnegative and ordered")
        if "policy" not in row and "policy_id" not in row:
            raise ValueError("Each arrival needs a policy or policy_id")
        if "policy" in row:
            policy = Policy.from_file(path.parent / row["policy"])
            if "policy_id" in row and row["policy_id"] != policy.id:
                raise ValueError("Trace policy_id does not match saved policy source")
        else:
            policy = policies[row["policy_id"]]
        result.append((at, policy, row))
        previous = at
    return result, {key: value for key, value in metadata.items() if key != "arrivals"}


async def correctness(args, policies):
    from research.ocean.evaluator import rollout

    comparisons = []
    async with evaluator(args, args.output / "reference", "reference", 1, 1) as reference:
        for policy in policies:
            base = await rollout(
                policy._implementation, range(args.seeds), 1, args.max_steps, trace=True
            )
            expected = sorted(base["results"], key=lambda row: row["seed"])
            save(args.output / f"trace-{policy.id}.json", expected)
            original = await reference.submit(policy, seeds=range(args.seeds))
            comparisons.append(
                dict(
                    policy_id=policy.id,
                    mode="reference",
                    passed=original["status"] == "ok"
                    and original["results"]
                    == [
                        {key: value for key, value in row.items() if key != "actions"}
                        for row in expected
                    ],
                )
            )
            for width, reverse in ((1, False), (8, False), (32, False), (32, True)):
                seeds = list(range(args.seeds))
                if reverse:
                    seeds.reverse()
                actual = await rollout(
                    policy._implementation, seeds, width, args.max_steps, trace=True
                )
                comparisons.append(
                    dict(
                        policy_id=policy.id,
                        batch_size=width,
                        reversed=reverse,
                        passed=sorted(actual["results"], key=lambda row: row["seed"]) == expected,
                    )
                )
    return dict(
        kind="correctness",
        passed=all(row["passed"] for row in comparisons),
        full_workload=args.seeds == 32 and args.max_steps == 2000 and len(policies) >= 8,
        comparisons=comparisons,
    )


def evaluator(args, output, mode, workers, batch):
    from research.ocean.evaluator import PanelEvaluator

    return PanelEvaluator(
        output,
        mode=mode,
        workers=workers,
        batch_size=batch,
        max_steps=args.max_steps,
        timeout=args.timeout,
        diagnostics=args.diagnostics,
    )


async def initialize(panel, policy, seeds):
    first = await panel.submit(policy, seeds=seeds, job_id="cold")
    warm = await panel.submit(policy, seeds=seeds, job_id="warmup")
    if first["status"] != "ok" or warm["status"] != "ok":
        raise RuntimeError(f"Initialization failed: {first.get('error')} {warm.get('error')}")
    return dict(
        first_panel_seconds=first["response_seconds"], warmup_seconds=warm["response_seconds"]
    )


async def capacity(args, policies, config, output):
    mode, workers, batch = config
    async with evaluator(args, output, *config) as panel:
        cold = await initialize(panel, policies[0], range(args.seeds))
        rows, pending = [], set()
        start, submitted, valid = time.monotonic(), 0, 0
        # Complete whole policy cycles so each configuration measures the same mix.
        cycle = math.lcm(workers, len(policies))
        stop_arrivals = None
        try:
            while pending or stop_arrivals is None:
                enough = time.monotonic() - start >= args.duration and valid >= args.panels
                if stop_arrivals is None and enough and submitted % cycle == 0:
                    stop_arrivals = time.monotonic()
                while stop_arrivals is None and len(pending) < workers:
                    policy = policies[submitted % len(policies)]
                    pending.add(
                        asyncio.create_task(
                            panel.submit(policy, seeds=range(args.seeds), job_id=f"job-{submitted}")
                        )
                    )
                    submitted += 1
                    if enough and submitted % cycle == 0:
                        stop_arrivals = time.monotonic()
                if not pending:
                    break
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    row = task.result()
                    rows.append(row)
                    valid += row["status"] == "ok"
                # A broken policy must not create an endless success-count loop.
                if any(row["status"] != "ok" for row in rows):
                    stop_arrivals = stop_arrivals or time.monotonic()
            finish = time.monotonic()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        report = summarize(rows, finish - start)
        report.update(
            cold=cold,
            mode=mode,
            workers=workers,
            batch_size=batch,
            drain_seconds=finish - stop_arrivals,
            sufficient_duration=stop_arrivals - start >= 30 and valid >= 100,
            arrivals_seconds=stop_arrivals - start,
        )
        save(output / "events.json", panel.events)
        save(output / "result.json", report)
        return report


async def replay(args, policies, config, output, arrivals):
    mode, workers, batch = config
    async with evaluator(args, output, *config) as panel:
        cold = await initialize(panel, policies[0], range(args.seeds))
        rows, samples, admissions = [], [], []
        pending = set()
        start = time.monotonic()
        stopped = asyncio.Event()

        async def sample():
            while not stopped.is_set():
                samples.append(
                    dict(at=time.monotonic() - start, queued=panel.queued, running=panel.running)
                )
                try:
                    await asyncio.wait_for(stopped.wait(), 1)
                except asyncio.TimeoutError:
                    pass

        sampler = asyncio.create_task(sample())
        try:
            for index, (at, policy, metadata) in enumerate(arrivals):
                scheduled = start + at / args.speed
                await asyncio.sleep(max(0, scheduled - time.monotonic()))
                done = {task for task in pending if task.done()}
                pending.difference_update(done)
                rows.extend(task.result() for task in done)
                blocked = time.monotonic()
                was_blocked = False
                while len(pending) >= workers + args.queue_limit:
                    was_blocked = True
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    rows.extend(task.result() for task in done)
                admitted = time.monotonic()
                admissions.append(
                    dict(
                        job_id=f"job-{index}",
                        scheduled=scheduled,
                        admitted=admitted,
                        producer_lag_seconds=max(0, admitted - scheduled),
                        backpressure_seconds=admitted - blocked if was_blocked else 0,
                        trace=metadata,
                    )
                )
                pending.add(
                    asyncio.create_task(
                        panel.submit(policy, seeds=range(args.seeds), job_id=f"job-{index}")
                    )
                )
            final_arrival = time.monotonic()
            rows.extend(await asyncio.gather(*pending))
            pending.clear()
            duration = arrivals[-1][0] if args.trace else args.duration
            await asyncio.sleep(max(0, start + duration / args.speed - time.monotonic()))
            finish = time.monotonic()
        finally:
            stopped.set()
            await sampler
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        samples.append(dict(at=finish - start, queued=panel.queued, running=panel.running))
        arrival_duration = (arrivals[-1][0] if args.trace else args.duration) / args.speed
        # First minute after warmup excludes the first replay minute.
        first = [s["queued"] for s in samples if 60 <= s["at"] < 120]
        last = [
            s["queued"] for s in samples if arrival_duration - 60 <= s["at"] <= arrival_duration
        ]
        report = summarize(rows, finish - start)
        report.update(
            cold=cold,
            mode=mode,
            workers=workers,
            batch_size=batch,
            scheduled_jobs=len(arrivals),
            accounted_jobs=len(rows),
            dropped_jobs=len(arrivals) - len(rows),
            final_queue=panel.queued,
            arrival_duration_seconds=arrival_duration,
            drain_seconds=finish - final_arrival,
            scheduled_drain_seconds=max(0, finish - start - arrival_duration),
            backpressure_seconds=sum(a["backpressure_seconds"] for a in admissions),
            p95_producer_lag_seconds=percentile(
                [a["producer_lag_seconds"] for a in admissions], 0.95
            ),
            queue_growth=(statistics.mean(last) - statistics.mean(first))
            if first and last
            else None,
            backpressure_growth=None,
        )
        first_lag = [
            a["producer_lag_seconds"] for a in admissions if 60 <= a["scheduled"] - start < 120
        ]
        last_lag = [
            a["producer_lag_seconds"]
            for a in admissions
            if arrival_duration - 60 <= a["scheduled"] - start <= arrival_duration
        ]
        if first_lag and last_lag:
            report["backpressure_growth"] = statistics.mean(last_lag) - statistics.mean(first_lag)
        save(output / "events.json", panel.events)
        save(output / "admissions.json", admissions)
        save(output / "queue.json", samples)
        save(output / "result.json", report)
        # Standalone, dependency-free queue visualization retains every sampled burst.
        peak = max(s["queued"] for s in samples)
        maximum = max(1, peak)
        points = " ".join(
            f"{40 + 920 * s['at'] / max(1, finish - start):.2f},{260 - 220 * s['queued'] / maximum:.2f}"
            for s in samples
        )
        (output / "queue.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300">'
            '<rect width="1000" height="300" fill="white"/>'
            f'<text x="40" y="20">Queued jobs; peak {peak}; elapsed {finish - start:.1f}s</text>'
            f'<polyline points="{points}" fill="none" stroke="navy" stroke-width="2"/></svg>'
        )
        return report


def parser():
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("kind", choices=("correctness", "capacity", "replay"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--mode", choices=("reference", "summary", "batch"), default="batch")
    result.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--seeds", type=int, default=32, help="Seeds 0 through N-1")
    result.add_argument("--max-steps", type=int, default=2000)
    result.add_argument("--timeout", type=float, default=60)
    result.add_argument("--duration", type=float, help="Default: capacity 30s, replay 300s")
    result.add_argument("--panels", type=int, default=100)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument(
        "--matrix",
        action="store_true",
        help="Capacity: all 1/2/4 workers, modes and batch widths 1/8/32",
    )
    result.add_argument("--policy", action="append", type=Path, default=[])
    result.add_argument("--trace", type=Path)
    result.add_argument("--rate", type=float, default=1, help="Synthetic jobs/s")
    result.add_argument(
        "--speed", type=float, default=1, help="Replay acceleration; 1.25 checks headroom"
    )
    result.add_argument("--queue-limit", type=int, default=128)
    result.add_argument("--correctness-report", type=Path)
    result.add_argument("--capacity-report", type=Path)
    result.add_argument(
        "--diagnostics",
        action="store_true",
        help="Separate instrumented attribution run, excluded from capacity evidence",
    )
    return result


async def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    args.duration = (
        args.duration if args.duration is not None else (300 if args.kind == "replay" else 30)
    )
    for key in (
        "batch_size",
        "seeds",
        "max_steps",
        "timeout",
        "duration",
        "panels",
        "repeats",
        "rate",
        "speed",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            command.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.queue_limit < 0:
        command.error("--queue-limit must be nonnegative")
    if args.matrix and args.kind != "capacity":
        command.error("--matrix applies only to capacity")
    args.output.mkdir(parents=True, exist_ok=False)
    # Set before importing native/evaluator; fresh worker imports inherit these limits.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numpy as np

    from research.ocean.baselines import policies as baselines
    from research.ocean.native import FLAGS, SOURCE, UPSTREAM, build

    policies = [Policy.from_file(path) for path in args.policy] if args.policy else baselines()
    build_start = time.monotonic()
    library = build()
    build_seconds = time.monotonic() - build_start
    arrivals, metadata = [], {}
    if args.trace:
        arrivals, metadata = load_trace(args.trace, {p.id: p for p in policies})
        policies = [p for _, p, _ in arrivals]
    if args.kind == "replay":
        if not args.trace:
            arrivals = [
                (i / args.rate, policies[i % len(policies)], {})
                for i in range(max(1, math.ceil(args.duration * args.rate)))
            ]
    unique_policies = list({p.id: p for p in policies}.values())
    counts = Counter(p.id for p in policies)
    workload = dict(
        seeds=list(range(args.seeds)), max_steps=args.max_steps, policy_ids=sorted(counts)
    )
    mix = {key: value / len(policies) for key, value in sorted(counts.items())}
    hardware = dict(
        platform=platform.platform(),
        machine=platform.machine(),
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        python=sys.version,
        numpy=np.__version__,
        host=platform.node(),
        cpu_budget=4,
        memory_budget_bytes=8 * 1024**3,
    )
    manifest = dict(
        kind=args.kind,
        date=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        libc=platform.libc_ver(),
        machine=platform.machine(),
        python=sys.version,
        numpy=np.__version__,
        host_cpu_count=os.cpu_count(),
        allocated_cpu_budget=4,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        memory_budget_bytes=8 * 1024**3,
        resource_budget_enforced=False,
        native_library=str(library),
        upstream=UPSTREAM,
        compiler_flags=FLAGS,
        compiler_version=subprocess.check_output(
            [*shlex.split(os.environ.get("CC", "cc")), "--version"], text=True
        ).strip(),
        native_source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(SOURCE.iterdir())
            if path.suffix in (".c", ".h")
        },
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        build_seconds=build_seconds,
        cache_enabled=False,
        cooperative_execution=True,
        arguments={
            key: [str(p) for p in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        policy_provenance="external saved sources"
        if args.policy or args.trace
        else "reviewed baselines; not LLM-generated",
        policies=[
            dict(
                id=p.id, name=p.name, sha256=hashlib.sha256(p._implementation.encode()).hexdigest()
            )
            for p in unique_policies
        ],
        trace_metadata=metadata,
    )
    for policy in unique_policies:
        policy.to_file(args.output / f"policy-{policy.id}.py")
    save(args.output / "manifest.json", manifest)
    if args.kind == "correctness":
        report = await correctness(args, unique_policies)
    else:
        configs = [(args.mode, args.workers, args.batch_size)]
        if args.matrix:
            configs = [
                (mode, worker, width)
                for mode in ("reference", "summary", "batch")
                for worker in (1, 2, 4)
                for width in ((1, 8, 32) if mode == "batch" else (1,))
            ]
        runs = []
        for repeat in range(args.repeats):
            for config in configs if repeat % 2 == 0 else list(reversed(configs)):
                output = args.output / f"{repeat}-{config[0]}-{config[1]}-{config[2]}"
                output.mkdir()
                run = await (
                    capacity(args, policies, config, output)
                    if args.kind == "capacity"
                    else replay(args, policies, config, output, arrivals)
                )
                runs.append(dict(repeat=repeat, **run))
                save(args.output / "report.json", dict(kind=args.kind, runs=runs, pilot_pass=False))
        report = dict(
            kind=args.kind,
            runs=runs,
            pilot_pass=False,
            full_workload=args.seeds == 32
            and args.max_steps == 2000
            and len(unique_policies) >= 8
            and not args.diagnostics,
            notes=[
                "Peak RSS below is process high-water RSS, not simultaneous process-tree RAM.",
                "CPU/RAM budgets are declared, not OS-enforced; freeze allocation externally.",
                "Fresh candidate startup, result preparation and persistence are timed.",
                "Worker phase durations in events are diagnostic; worker wall times overlap.",
            ],
        )
        correctness_result = (
            json.loads(args.correctness_report.read_text()) if args.correctness_report else {}
        )
        gates = dict(
            correctness=bool(
                correctness_result.get("passed")
                and correctness_result.get("full_workload")
                and correctness_result.get("workload") == workload
                and correctness_result.get("hardware") == hardware
            )
        )
        if args.kind == "replay":
            active = metadata.get("active_generation_seconds", 0)
            latencies = metadata.get("llm_latencies", [])
            if not latencies:
                latencies = [
                    row[2]["llm_finish"] - row[2]["llm_start"]
                    for row in arrivals
                    if "llm_finish" in row[2] and "llm_start" in row[2]
                ]
            real = bool(
                args.trace
                and metadata.get("model")
                and metadata.get("generation_concurrency")
                and metadata.get("provider_limits")
                and math.isfinite(active)
                and active > 0
                and latencies
                and all(math.isfinite(x) and x > 0 for x in latencies)
            )
            rate = len(arrivals) / active * args.speed if real else args.rate * args.speed
            wait_limit = max(0.1, 0.1 * statistics.median(latencies)) if real else None
            capacity_result = (
                json.loads(args.capacity_report.read_text()) if args.capacity_report else {}
            )
            matching = [
                r
                for r in capacity_result.get("runs", [])
                if (r["mode"], r["workers"], r["batch_size"]) == configs[0]
            ]
            gates.update(
                real_llm_trace=real,
                full_replays=args.repeats >= 3
                and all(r["arrival_duration_seconds"] >= 300 for r in runs),
                capacity=bool(
                    capacity_result.get("full_workload")
                    and capacity_result.get("workload") == workload
                    and capacity_result.get("hardware") == hardware
                    and capacity_result.get("policy_mix") == mix
                    and len(matching) >= 3
                    and all(
                        r["sufficient_duration"]
                        and r["failed_panels"] == 0
                        and r["valid_panels_per_second"] >= 1.25 * rate
                        for r in matching
                    )
                ),
                stable_queue=all(
                    r["queue_growth"] is not None and r["queue_growth"] <= 1 for r in runs
                ),
                queue_wait=real and all(r["p95_queue_seconds"] <= wait_limit for r in runs),
                accounted=all(r["dropped_jobs"] == 0 and r["failed_panels"] == 0 for r in runs),
                no_growing_backpressure=all(
                    r["backpressure_growth"] is not None and r["backpressure_growth"] <= 0.1
                    for r in runs
                ),
            )
            report.update(
                arrival_rate=rate,
                queue_wait_limit_seconds=wait_limit,
                pilot_pass=report["full_workload"] and all(gates.values()),
            )
        else:
            gates["capacity_sample"] = args.repeats >= 3 and all(
                r["sufficient_duration"] for r in runs
            )
        report["gates"] = gates
    report.update(workload=workload, hardware=hardware, policy_mix=mix)
    rss_factor = 1 if sys.platform == "darwin" else 1024
    report["parent_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_factor
    )
    report["max_reaped_child_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * rss_factor
    )
    save(args.output / "report.json", report)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                kind=args.kind,
                passed=report.get("passed"),
                pilot_pass=report.get("pilot_pass", False),
            )
        )
    )
    return report


if __name__ == "__main__":
    result = asyncio.run(main())
    if result.get("passed") is False:
        raise SystemExit(1)
