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

import numpy as np

from research.rewards import Measurement
from rsikit.evaluation import InfrastructureError, PolicyError, PolicyTimeout
from rsikit.policy import load_policy, validate_policy

from .environment import factory, metadata

MAX_RESULT = 64 * 1024 * 1024


def _inputs(seeds, batch_size, max_steps):
    if any(type(x) is not int or not 1 <= x <= 0x7FFFFFFF for x in (batch_size, max_steps)):
        raise ValueError("Batch size and max_steps must be positive int32 integers")
    seeds = tuple(seeds)
    largest_seed = (0x7FFFFFFF - batch_size + 1) // batch_size
    if not seeds or any(type(s) is not int or not 0 <= s <= largest_seed for s in seeds):
        raise ValueError(
            "Seeds must be nonempty nonnegative integers with (seed + 1) * batch_size <= 2**31"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique within a panel")

    return seeds


async def rollout(
    source,
    seeds,
    batch_size=32,
    max_steps=2000,
    trace=False,
    diagnostics=False,
    *,
    env_name="g2048",
    score_key="merge_score",
    env_kwargs=None,
):
    """Each seed is a fresh upstream batch run for a fixed number of vector steps.

    Native autoresets and logging stay upstream. A logged score is the mean over
    completed episodes (zero if none completed); `return` includes all rewards,
    including unfinished episodes, averaged over the batch lanes.
    """
    seeds = _inputs(seeds, batch_size, max_steps)
    constructor = factory(env_name)
    kwargs = dict(env_kwargs or {})
    if set(kwargs) & {"num_envs", "seed", "log_interval", "buf"}:
        raise ValueError("num_envs, seed, log_interval and buf are controlled by the evaluator")
    results = []
    timing = dict(reset=0.0, policy=0.0, environment=0.0, validation=0.0)
    for seed in seeds:
        start = perf_counter() if diagnostics else 0
        env = constructor(num_envs=batch_size, seed=seed, log_interval=max_steps, **kwargs)
        policy = None
        try:
            observation, _ = env.reset(seed=seed)
            if env.num_agents != batch_size:
                raise ValueError("This evaluator requires one agent per environment")
            policy = load_policy(
                source,
                env.observation_space,
                env.action_space,
                "Independent rows of upstream Ocean observations; one action per row.",
            )
            await policy.reset(seed=seed)
            if diagnostics:
                timing["reset"] += perf_counter() - start
            actions = [] if trace else None
            returns = np.zeros(batch_size, dtype=np.float64)
            infos = []
            for _ in range(max_steps):
                start = perf_counter() if diagnostics else 0
                action = np.asarray(await policy.act(observation.copy()))
                if diagnostics:
                    timing["policy"] += perf_counter() - start
                start = perf_counter() if diagnostics else 0
                if not np.isfinite(action).all() or not env.action_space.contains(action):
                    raise PolicyError(
                        f"Expected actions in {env.action_space}; got shape {action.shape}"
                    )
                if diagnostics:
                    timing["validation"] += perf_counter() - start
                if trace:
                    actions.append(action.tolist())
                start = perf_counter() if diagnostics else 0
                observation, reward, _, _, infos = env.step(action)
                returns += reward
                if diagnostics:
                    timing["environment"] += perf_counter() - start
            metrics = dict(infos[0]) if infos else {}
            count = int(metrics.get("n", 0))
            if score_key != "return" and count and score_key not in metrics:
                raise ValueError(
                    f"Unknown score key {score_key!r}; upstream logged {sorted(metrics)}"
                )
            score = (
                float(returns.mean()) if score_key == "return" else float(metrics.get(score_key, 0))
            )
            row = dict(
                seed=seed,
                batch_size=batch_size,
                vector_steps=max_steps,
                steps=max_steps * batch_size,
                score=score,
                score_key=score_key,
                episodes=count,
                metrics=metrics,
                ending="horizon",
            )
            row["return"] = float(returns.mean())
            if trace:
                row["actions"] = actions
            results.append(row)
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


def _child(channel, directory, source, seeds, batch_size, max_steps, diagnostics, options):
    os.setsid()
    os.chdir(directory)
    with channel, open("policy.log", "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            result = asyncio.run(
                rollout(source, seeds, batch_size, max_steps, diagnostics=diagnostics, **options)
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
        env_name="g2048",
        score_key="merge_score",
        env_kwargs=None,
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
            mp.set_forkserver_preload(["pufferlib.ocean", "research.ocean.evaluator"])
        self.options = dict(env_name=env_name, score_key=score_key, env_kwargs=env_kwargs)
        self.upstream = metadata(env_name)

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
                    self.batch_size,
                    self.max_steps,
                    self.diagnostics,
                    self.options,
                ),
            )
            reading = None
            try:
                process.start()
                writer.close()
                reading = asyncio.create_task(asyncio.to_thread(reader.recv_bytes, MAX_RESULT))
                done, _ = await asyncio.wait([reading], timeout=self.timeout * len(seeds))
                if not done:
                    raise PolicyTimeout(f"Candidate panel exceeded {self.timeout * len(seeds):g}s")
                try:
                    result = json.loads(reading.result())
                except (EOFError, OSError, ValueError) as exc:
                    raise InfrastructureError("Ocean worker exited without a valid result") from exc
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
        # Same upstream workload, but a fresh process for each batch seed.
        rows = []
        for seed in seeds:
            result = await self._panel(source, [seed])
            rows.extend(result["results"])
        return dict(results=rows, steps=sum(row["steps"] for row in rows))

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
            protocol=self.upstream["protocol"],
            environment=self.options["env_name"],
            score_key=self.options["score_key"],
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
                    operation(policy._implementation, seeds), self.timeout * len(seeds)
                )
                rows = result["results"]
                if [row["seed"] for row in rows] != list(seeds) or any(
                    not math.isfinite(row["score"]) for row in rows
                ):
                    raise PolicyError("Incomplete or nonfinite candidate panel")
                event.update(result, status="ok")
            except asyncio.TimeoutError:
                event["error"] = f"Candidate panel exceeded {self.timeout * len(seeds):g}s"
            except (InfrastructureError, OSError) as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise InfrastructureError(str(exc)) from exc
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
