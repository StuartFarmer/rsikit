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

from rsikit import Episode
from rsikit.evaluation import InfrastructureError, PolicyError, PolicyTimeout, _policy_boundary
from rsikit.policy import InvalidPolicy, load_policy

from .environment import factory, metadata

MAX_RESULT = 64 * 1024 * 1024


def _inputs(seeds, batch_size, max_steps):
    if any(type(x) is not int or not 1 <= x <= 0x7FFFFFFF for x in (batch_size, max_steps)):
        raise ValueError("Batch size and max_steps must be positive int32 integers")
    seeds = tuple(seeds)
    largest_seed = 0x7FFFFFFF
    if not seeds or any(type(s) is not int or not 0 <= s <= largest_seed for s in seeds):
        raise ValueError("Seeds must be nonempty nonnegative int32 integers")
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
    record=False,
    env_name="g2048",
    score_key="merge_score",
    env_kwargs=None,
    _policy_factory=load_policy,
    _on_batch=None,
):
    """Run one episode per seed, batching at most batch_size independent games."""
    seeds = _inputs(seeds, batch_size, max_steps)
    metadata(env_name)  # Fail clearly if the installed binding lacks the episode fix.
    constructor = factory(env_name)
    kwargs = dict(env_kwargs or {})
    if set(kwargs) & {"num_envs", "seed", "log_interval", "buf"}:
        raise ValueError("num_envs, seed, log_interval and buf are controlled by the evaluator")
    results = []
    timing = dict(reset=0.0, policy=0.0, environment=0.0, validation=0.0)
    for offset in range(0, len(seeds), batch_size):
        panel = seeds[offset : offset + batch_size]
        width = len(panel)
        start = perf_counter() if diagnostics else 0
        env = constructor(num_envs=width, seed=0, log_interval=max_steps + 1, **kwargs)
        policy = None
        episodes = {seed: Episode() for seed in panel}
        try:
            observation, _ = env.reset(seed=list(panel))
            if record:
                for lane, seed in enumerate(panel):
                    episodes[seed].observations.append(observation[lane].copy())
                    episodes[seed].infos.append({})
            if env.num_agents != width:
                raise ValueError("This evaluator requires one agent per environment")
            with _policy_boundary():
                policy = _policy_factory(
                    source,
                    env.observation_space,
                    env.action_space,
                    "Independent rows of upstream Ocean observations; one action per row.",
                )
                # Deterministic row-independent policies; episode memory resets per batch.
                await policy.reset(seed=0)
            if diagnostics:
                timing["reset"] += perf_counter() - start
            actions = [[] for _ in panel] if trace else None
            returns = np.zeros(width, dtype=np.float64)
            steps = np.zeros(width, dtype=np.int64)
            # ponytail: frozen lanes keep fixed buffers; compact only if inference dominates.
            active = np.ones(width, dtype=bool)
            terminated = np.zeros(width, dtype=bool)
            for step in range(max_steps):
                start = perf_counter() if diagnostics else 0
                with _policy_boundary():
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
                    for lane in np.flatnonzero(active):
                        actions[lane].append(action[lane].tolist())
                start = perf_counter() if diagnostics else 0
                observation, reward, done, truncated, _ = env.step(action)
                if record:
                    for lane in np.flatnonzero(active):
                        episode = episodes[panel[lane]]
                        episode.actions.append(action[lane].copy())
                        episode.rewards.append(float(reward[lane]))
                        episode.observations.append(observation[lane].copy())
                        episode.terminations.append(bool(done[lane]))
                        episode.truncations.append(bool(truncated[lane] or step == max_steps - 1))
                        episode.infos.append({})
                returns[active] += reward[active]
                steps[active] += 1
                terminated |= active & done
                active &= ~(done | truncated)
                if diagnostics:
                    timing["environment"] += perf_counter() - start
                if not active.any():
                    break
            for lane, metrics in enumerate(env.episode_stats()):
                if score_key != "return" and score_key not in metrics:
                    raise ValueError(
                        f"Unknown score key {score_key!r}; available: {sorted(metrics)}"
                    )
                row = dict(
                    seed=panel[lane],
                    vector_steps=int(steps[lane]),
                    steps=int(steps[lane]),
                    score=float(returns[lane] if score_key == "return" else metrics[score_key]),
                    score_key=score_key,
                    episodes=1,
                    metrics=metrics,
                    ending="terminated" if terminated[lane] else "truncated",
                )
                row["return"] = float(returns[lane])
                if trace:
                    row["actions"] = actions[lane]
                if record:
                    episode = episodes[panel[lane]]
                    episode.infos[-1].update(metrics=metrics, fitness=row["score"])
                    row["episode"] = episode.encode()
                results.append(row)
        except PolicyError as exc:
            if not record:
                raise
            statistics = env.episode_stats()
            for lane, (seed, episode) in enumerate(episodes.items()):
                if episode.rewards and (episode.terminations[-1] or episode.truncations[-1]):
                    metrics = statistics[lane]
                    score = (
                        episode.total_reward if score_key == "return" else float(metrics[score_key])
                    )
                    episode.infos[-1].update(metrics=metrics, fitness=score)
                else:
                    episode.error = str(exc)
                    score = None
                results.append(
                    dict(seed=seed, score=score, steps=len(episode), episode=episode.encode())
                )
        finally:
            try:
                if policy is not None:
                    await policy.close()
            finally:
                env.close()
        if _on_batch is not None:
            _on_batch(results)
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
        except BaseException as exc:
            data = json.dumps(
                {
                    "error": traceback.format_exc()[-8000:],
                    "kind": "policy" if isinstance(exc, PolicyError) else "infrastructure",
                }
            ).encode()
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

    async def _panel(self, source, seeds, *, record=False):
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
                    {**self.options, "record": record},
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
                    error = PolicyError if result.get("kind") == "policy" else InfrastructureError
                    raise error(result["error"])
                return result
            finally:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        if sys.platform != "darwin":
                            raise
                        # Darwin can report EPERM before an exiting child is reapable.
                        await asyncio.to_thread(process.join, 1.0)
                        if process.is_alive():
                            raise
                    if process.is_alive():
                        process.kill()
                    process.join()
                process.close()
                writer.close()
                if reading is not None:
                    await asyncio.gather(reading, return_exceptions=True)
                reader.close()

    async def _reference(self, source, seeds, *, record=False):
        # Same episodes, but a fresh process for each seed.
        rows = []
        for seed in seeds:
            result = await self._panel(source, [seed], record=record)
            rows.extend(result["results"])
        return dict(results=rows, steps=sum(row["steps"] for row in rows))

    async def submit(self, policy, seeds=range(32), job_id=None, *, record=False):
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
                policy.validate()
                source_path = self.output / "policies" / f"{policy.id}.py"
                if not source_path.exists():
                    policy.to_file(source_path)
                operation = self._reference if self.mode == "reference" else self._panel
                result = await asyncio.wait_for(
                    operation(policy.source, seeds, record=record),
                    self.timeout * len(seeds),
                )
                rows = result["results"]
                if [row["seed"] for row in rows] != list(seeds) or any(
                    row.get("episode", {}).get("error") is None and not math.isfinite(row["score"])
                    for row in rows
                ):
                    raise PolicyError("Incomplete or nonfinite candidate panel")
                event.update(result, status="ok")
                if any(row.get("episode", {}).get("error") for row in rows):
                    event.update(status="failed", error="Candidate episode failed")
            except asyncio.TimeoutError:
                event["error"] = f"Candidate panel exceeded {self.timeout * len(seeds):g}s"
            except (InfrastructureError, OSError) as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise InfrastructureError(str(exc)) from exc
            except (PolicyError, InvalidPolicy) as exc:
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
        results = await asyncio.gather(
            *(self.submit(policy, seeds, record=True) for policy in policies)
        )
        return {result["policy_id"]: episode_panel(result, seeds) for result in results}


def episode_panel(event, seeds):
    """Decode recorded trajectories; process failures have no recoverable trajectory."""
    episodes = {row["seed"]: Episode.from_data(row["episode"]) for row in event["results"]}
    for seed in seeds:
        if seed not in episodes:
            episodes[seed] = Episode(error=event["error"] or "Episode did not return")
    return episodes
