"""Candidate-panel processes and compact Ocean measurements for research.

This is cooperative execution, not a security boundary against hostile policies.
"""

import asyncio
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import cloudpickle
import gymnasium as gym
import numpy as np

from research.rewards import Measurement
from rsikit import Executor
from rsikit.evaluation import PolicyError, PolicyTimeout
from rsikit.policy import load_policy, validate_policy

from .native import Batch, OceanEnv, build

MAX_RESULT = 64 * 1024 * 1024


def _inputs(seeds, batch_size, max_steps):
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be nonempty unsigned 32-bit integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique within a panel")
    if any(type(x) is not int or x < 1 for x in (batch_size, max_steps)):
        raise ValueError("Batch size and max_steps must be positive integers")
    return seeds


async def rollout(source, seeds, batch_size=32, max_steps=2000, trace=False, diagnostics=False):
    """Run one source over a seed panel directly; the caller supplies isolation."""
    seeds = _inputs(seeds, batch_size, max_steps)
    results = []
    timing = dict(reset=0.0, policy=0.0, environment=0.0, validation=0.0)
    for offset in range(0, len(seeds), batch_size):
        start = perf_counter() if diagnostics else 0
        env = Batch(seeds[offset : offset + batch_size], max_steps=max_steps)
        policy = None
        actions_by_slot = [[] for _ in seeds[offset : offset + batch_size]] if trace else None
        try:
            size = len(env.active)
            policy = load_policy(
                source,
                gym.spaces.Box(0, 255, (size, 16), dtype=np.float32),
                gym.spaces.MultiDiscrete([4] * size),
                "Stateless independent rows of 2048 tile exponents; return one action per row.",
            )
            await policy.reset(seed=0)
            if diagnostics:
                timing["reset"] += perf_counter() - start
            while len(env.active):
                slots = env.active.copy()
                if len(slots) != size:
                    size = len(slots)
                    policy.observation_space = gym.spaces.Box(0, 255, (size, 16), dtype=np.float32)
                    policy.action_space = gym.spaces.MultiDiscrete([4] * size)
                observation = env.observations[slots].copy()
                start = perf_counter() if diagnostics else 0
                action = await policy.act(observation)
                if diagnostics:
                    timing["policy"] += perf_counter() - start
                start = perf_counter() if diagnostics else 0
                action = np.asarray(action)
                if (
                    action.shape != (size,)
                    or action.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(action))
                    or not np.all((action >= 0) & (action < 4) & (action == np.floor(action)))
                ):
                    raise PolicyError(
                        "Expected finite integer actions of shape (active_rows,) in [0, 3]"
                    )
                action = np.ascontiguousarray(action, dtype=np.int64)
                if diagnostics:
                    timing["validation"] += perf_counter() - start
                if trace:
                    for slot, value in zip(slots, action):
                        actions_by_slot[slot].append(int(value))
                start = perf_counter() if diagnostics else 0
                env.step(action)
                if diagnostics:
                    timing["environment"] += perf_counter() - start
            rows = [dict(row) for row in env.results]
            if trace:
                for row, actions in zip(rows, actions_by_slot):
                    row["actions"] = actions
            results.extend(rows)
        finally:
            try:
                if policy is not None:
                    await policy.close()
            finally:
                env.close()
    return dict(
        results=results,
        steps=sum(row["steps"] for row in results),
        timings=timing if diagnostics else {},
    )


def _child(channel, directory, source, seeds, batch_size, max_steps, diagnostics):
    os.setsid()
    os.chdir(directory)
    with channel, open("policy.log", "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            result = asyncio.run(
                rollout(source, seeds, batch_size, max_steps, diagnostics=diagnostics)
            )
            result["worker_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024
            )
            data = json.dumps(result, allow_nan=False).encode()
            if len(data) > MAX_RESULT:
                raise ValueError("Panel result exceeds 64 MiB")
        except BaseException:
            data = json.dumps({"error": traceback.format_exc()[-8000:]}).encode()
        channel.send_bytes(data)


class PanelEvaluator:
    """Bound fresh candidate processes; persist every result before completing jobs."""

    def __init__(
        self,
        output,
        *,
        mode="batch",
        workers=1,
        batch_size=32,
        max_steps=2000,
        timeout=60,
        diagnostics=False,
    ):
        if mode not in ("reference", "summary", "batch"):
            raise ValueError("Mode must be reference, summary or batch")
        if type(workers) is not int or workers < 1:
            raise ValueError("Workers must be a positive integer")
        _inputs([0], batch_size, max_steps)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("policies", "measurements"):
            (self.output / name).mkdir(exist_ok=True)
        self.mode, self.workers, self.batch_size = mode, workers, batch_size
        self.max_steps, self.timeout = max_steps, timeout
        self.diagnostics = diagnostics
        self.events = []
        self.queued = self.running = 0
        self._slots = asyncio.Semaphore(workers)
        self._tasks = set()
        self._job_ids = set()
        self._closed = False
        method = "forkserver" if sys.platform == "linux" else "spawn"
        self._context = mp.get_context(method)
        if method == "forkserver":
            mp.set_forkserver_preload(["research.ocean.evaluator"])
        self.library = build()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _panel(self, source, seeds):
        with TemporaryDirectory(prefix="ocean-panel-") as directory:
            reader, writer = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_child,
                args=(
                    writer,
                    directory,
                    source,
                    seeds,
                    1 if self.mode == "summary" else self.batch_size,
                    self.max_steps,
                    self.diagnostics,
                ),
            )
            reading = None
            try:
                process.start()
                writer.close()
                reading = asyncio.create_task(asyncio.to_thread(reader.recv_bytes, MAX_RESULT))
                done, _ = await asyncio.wait([reading], timeout=self.timeout)
                if not done:
                    raise PolicyTimeout(f"Candidate panel exceeded {self.timeout:g}s")
                result = json.loads(reading.result())
                if "error" in result:
                    raise PolicyError(result["error"])
                return result
            finally:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        if sys.platform != "darwin" or process.is_alive():
                            raise
                    if process.is_alive():
                        process.kill()
                    process.join()
                process.close()
                writer.close()
                if reading is not None:
                    await asyncio.gather(reading, return_exceptions=True)
                reader.close()

    async def _reference(self, source, seeds):
        # One episode child at a time per admitted panel: no nested worker pools.
        rows = []
        async with Executor(episode_timeout=self.timeout) as executor:
            for seed in seeds:
                episode = await executor._evaluate(
                    source,
                    cloudpickle.dumps(OceanEnv(max_steps=self.max_steps)),
                    seed,
                    policy_seed=0,
                )
                info = episode.infos[-1]
                rows.append(
                    {
                        "seed": seed,
                        "score": info["score"],
                        "max_tile": info["max_tile"],
                        "return": info["return"],
                        "steps": len(episode),
                        "ending": info["ending"],
                    }
                )
        return dict(results=rows, steps=sum(row["steps"] for row in rows), timings={})

    async def submit(self, policy, seeds=range(32), job_id=None):
        if self._closed:
            raise RuntimeError("Evaluator is closed")
        seeds = _inputs(seeds, self.batch_size, self.max_steps)
        job_id = uuid4().hex if job_id is None else str(job_id)
        if (
            not job_id
            or len(job_id) > 128
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in job_id
            )
        ):
            raise ValueError("Job ID must be 1–128 letters, digits, underscores or hyphens")
        if job_id in self._job_ids or (self.output / "measurements" / f"{job_id}.json").exists():
            raise ValueError("Job ID already used")
        self._job_ids.add(job_id)
        task = asyncio.current_task()
        self._tasks.add(task)
        event = dict(
            job_id=job_id,
            policy_id=policy.id,
            policy_name=policy.name,
            mode=self.mode,
            batch_size=self.batch_size,
            workers=self.workers,
            diagnostics=self.diagnostics,
            seeds=list(seeds),
            max_steps=self.max_steps,
            submitted=perf_counter(),
            started=None,
            status="failed",
            error=None,
            results=[],
            steps=0,
        )
        acquired = False
        self.queued += 1
        try:
            try:
                await self._slots.acquire()
                acquired = True
            finally:
                self.queued -= 1
            self.running += 1
            event["started"] = perf_counter()
            try:
                validate_policy(policy)
                source_path = self.output / "policies" / f"{policy.id}.py"
                if not source_path.exists():
                    policy.to_file(source_path)
                operation = self._reference if self.mode == "reference" else self._panel
                result = await asyncio.wait_for(
                    operation(policy._implementation, seeds), self.timeout
                )
                rows = result["results"]
                if [row["seed"] for row in rows] != list(seeds) or any(
                    not math.isfinite(row["score"]) for row in rows
                ):
                    raise PolicyError("Incomplete or nonfinite candidate panel")
                event.update(result, status="ok")
            except asyncio.TimeoutError:
                event["error"] = f"Candidate panel exceeded {self.timeout:g}s"
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
            return event
        except asyncio.CancelledError:
            event["status"], event["error"] = "cancelled", "Evaluation cancelled"
            raise
        finally:
            try:
                event["received"] = perf_counter()
                destination = self.output / "measurements" / f"{job_id}.json"
                destination.write_text(json.dumps(event, allow_nan=False) + "\n")
                event["persisted"] = perf_counter()
                event["queue_seconds"] = (
                    event["received"] if event["started"] is None else event["started"]
                ) - event["submitted"]
                event["service_seconds"] = (
                    0 if event["started"] is None else event["persisted"] - event["started"]
                )
                event["response_seconds"] = event["persisted"] - event["submitted"]
                with (self.output / "evaluations.jsonl").open("a") as log:
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                self.events.append(event)
            finally:
                if acquired:
                    self.running -= 1
                    self._slots.release()
                self._tasks.discard(task)

    async def evaluate(self, policies, seeds=range(32)):
        policies = list(policies)
        seeds = _inputs(seeds, self.batch_size, self.max_steps)
        if len({p.id for p in policies}) != len(policies):
            raise ValueError("Duplicate policies in evaluation request")
        results = await asyncio.gather(*(self.submit(policy, seeds) for policy in policies))
        return {
            result["policy_id"]: Measurement(
                scores={row["seed"]: row["score"] for row in result["results"]},
                failure=result["error"],
            )
            for result in results
        }
