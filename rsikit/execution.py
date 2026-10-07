"""Local episode processes; run this application inside Docker for host isolation."""

import asyncio
import logging
import math
import multiprocessing as mp
import os
import pickle
import signal
import sys
import traceback
from collections.abc import AsyncIterator, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cloudpickle
import gymnasium as gym

from .episode import Episode
from .evaluation import (
    Evaluator,
    InfrastructureError,
    PolicyError,
    PolicyTimeout,
    _policy_boundary,
    _validate_seed,
)
from .policy import Policy, PolicyDefinition, load_policy

MAX_RESULT = 64 * 1024 * 1024


@dataclass(eq=False)
class Job:
    """One episode on worker copies of the inputs; submit each job only once.

    Inputs are copied when a worker slot opens. Keep them unchanged until execution
    finishes. PolicyDefinition inputs are loaded inside the worker deadline.
    """

    policy: Policy | PolicyDefinition
    environment: gym.Env
    seed: int | None = None
    max_steps: int | None = field(default=None, kw_only=True)
    instructions: str | None = field(default=None, kw_only=True)
    result: Episode | None = field(default=None, init=False)
    _submitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.policy, (Policy, PolicyDefinition)):
            raise TypeError("policy must be a Policy instance or PolicyDefinition")
        if not isinstance(self.environment, gym.Env):
            raise TypeError("environment must be a Gymnasium environment")
        _validate_seed(self.seed)
        Evaluator(max_steps=self.max_steps)
        if self.instructions is not None:
            if not isinstance(self.instructions, str):
                raise TypeError("instructions must be text or None")
            if isinstance(self.policy, Policy):
                raise ValueError("Set instructions on the policy instance before creating a Job")

    @property
    def done(self) -> bool:
        """Whether an episode result is available, including policy failures."""
        return self.result is not None


def _run_child(channel, definition, seed, max_steps, instructions, directory):
    """One process owns the environment, actual Solution, logs and artifacts."""
    os.setsid()
    os.chdir(directory)
    with channel, open("episode.log", "w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
        try:
            candidate, env = cloudpickle.loads(definition)
            wrapper, index = env, 0
            while isinstance(wrapper, gym.Wrapper):
                if isinstance(wrapper, gym.wrappers.RecordVideo):
                    wrapper.video_folder = str(Path(directory) / "videos" / str(index))
                    Path(wrapper.video_folder).mkdir(parents=True)
                    index += 1
                wrapper = wrapper.env

            async def evaluate():
                policy = candidate if isinstance(candidate, Policy) else None
                episode = Episode()
                try:
                    if policy is None:
                        text = instructions
                        if text is None:
                            text = (
                                env.get_wrapper_attr("instructions")
                                if env.has_wrapper_attr("instructions")
                                else ""
                            )
                        with _policy_boundary():
                            policy = load_policy(
                                candidate.source,
                                deepcopy(env.observation_space),
                                deepcopy(env.action_space),
                                text,
                            )
                    episode = await Evaluator(max_steps=max_steps).evaluate(policy, env, seed=seed)
                    return episode
                except PolicyError as exc:
                    episode.error = "".join(traceback.format_exception(exc))
                    return episode
                finally:
                    primary = sys.exc_info()[1]
                    try:
                        try:
                            if policy is not None:
                                try:
                                    with _policy_boundary():
                                        await policy.close()
                                except PolicyError as exc:
                                    if primary is None and episode.error is None:
                                        episode.error = "".join(traceback.format_exception(exc))
                                    else:
                                        logging.getLogger(__name__).exception(
                                            "Cleanup failed while handling an episode error"
                                        )
                        finally:
                            env.close()
                    except BaseException:
                        if primary is None and episode.error is None:
                            raise
                        # Keep the original failure/cancellation; expose secondary cleanup errors.
                        logging.getLogger(__name__).exception(
                            "Cleanup failed while handling an episode error"
                        )

            episode = asyncio.run(evaluate())
            sys.stdout.flush()
            sys.stderr.flush()
            if episode.infos:
                episode.artifacts.update(episode.infos[-1].get("artifacts", {}))
            size = sum(len(data) for data in episode.artifacts.values())
            for path in Path(directory).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    size += path.stat().st_size
                    if size > MAX_RESULT:
                        raise InfrastructureError("Evaluation artifacts exceed 64 MiB")
                    if path.stat().st_size:
                        episode.artifacts[str(path.relative_to(directory))] = path.read_bytes()
            data = pickle.dumps(episode.encode(), protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            data = pickle.dumps(
                {
                    "kind": "policy" if isinstance(exc, PolicyError) else "infrastructure",
                    "error": "".join(traceback.format_exception(exc))[-8000:],
                }
            )
        if len(data) > MAX_RESULT:
            data = pickle.dumps({"kind": "infrastructure", "error": "Episode exceeds 64 MiB"})
        channel.send_bytes(data)


class Executor:
    """Bound concurrency and deadlines for fresh local episode processes."""

    def __init__(self, *, concurrency: int = 1, episode_timeout: float = 60.0):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not math.isfinite(episode_timeout) or episode_timeout <= 0:
            raise ValueError("episode_timeout must be positive and finite")
        self.concurrency = concurrency
        self.episode_timeout = episode_timeout
        self._slots = asyncio.Semaphore(concurrency)
        self._tasks = set()
        self._queued = self._running = 0
        method = "forkserver" if sys.platform == "linux" else "spawn"
        self._context = mp.get_context(method)
        if method == "forkserver":
            mp.set_forkserver_preload(["rsikit.execution"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        """Cancel outstanding episodes and reap their processes."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate(self, job: Job) -> Episode:
        try:
            definition = cloudpickle.dumps((job.policy, job.environment))
        except Exception as exc:
            raise InfrastructureError(
                f"Cannot serialize {type(job.policy).__qualname__} job inputs: {exc}"
            ) from exc
        with TemporaryDirectory(prefix="rsikit-episode-") as directory:
            receiving, sending = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_run_child,
                args=(sending, definition, job.seed, job.max_steps, job.instructions, directory),
            )
            reading = None
            try:
                process.start()
                sending.close()
                # Drain while the child runs: joining first deadlocks on large trajectories.
                reading = asyncio.create_task(asyncio.to_thread(receiving.recv_bytes, MAX_RESULT))
                try:
                    done, _ = await asyncio.wait([reading], timeout=self.episode_timeout)
                    if not done:
                        raise asyncio.TimeoutError
                    data = reading.result()
                except asyncio.TimeoutError as exc:
                    raise PolicyTimeout(f"Episode exceeded {self.episode_timeout:g}s") from exc
                except (EOFError, OSError) as exc:
                    raise InfrastructureError("Episode process exited without a result") from exc
                try:
                    result = pickle.loads(
                        data
                    )  # Workers run trusted local Python, like the parent.
                    if isinstance(result, dict) and set(result) == {"kind", "error"}:
                        if (
                            result["kind"] not in ("policy", "infrastructure")
                            or not isinstance(result["error"], str)
                            or not result["error"].strip()
                        ):
                            raise ValueError("Invalid worker error")
                    else:
                        return Episode.from_data(result)
                except Exception as exc:
                    raise InfrastructureError(f"Invalid episode result: {exc}") from exc
                raise (PolicyError if result["kind"] == "policy" else InfrastructureError)(
                    result["error"]
                )
            except (PolicyError, InfrastructureError) as exc:
                path = Path(directory) / "episode.log"
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 4096))
                        tail = stream.read(4096).decode(errors="replace")
                    if tail:
                        exc.args = (f"{exc}\nEpisode log (tail):\n{tail}",)
                if isinstance(exc, PolicyError):
                    return Episode(error=str(exc))
                raise
            finally:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Child may not yet have reached setsid().
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
                sending.close()
                if reading is not None:
                    await asyncio.gather(reading, return_exceptions=True)
                receiving.close()

    async def execute(self, jobs: Iterable[Job]) -> AsyncIterator[Job]:
        """Yield original jobs in completion order, with their result attached.

        Close a partially consumed iterator with contextlib.aclosing. Infrastructure
        failures are raised after successful siblings finish; cancellation reaps workers.
        """
        jobs = list(jobs)
        if any(not isinstance(job, Job) for job in jobs):
            raise TypeError("execute expects Job instances")
        if len({id(job) for job in jobs}) != len(jobs) or any(job._submitted for job in jobs):
            raise ValueError("Each Job can only be submitted once")
        for job in jobs:
            job._submitted = True

        async def evaluate(job):
            policy_id = (
                job.policy.id
                if isinstance(job.policy, PolicyDefinition)
                else type(job.policy).__qualname__
            )
            seed = job.seed
            queued_at = perf_counter()
            self._queued += 1
            try:
                await self._slots.acquire()
            finally:
                self._queued -= 1
            started_at = perf_counter()
            self._running += 1
            try:
                try:
                    episode = await self._evaluate(job)
                    job.result = episode
                    if episode.error is not None:
                        logging.getLogger(__name__).error(
                            "Policy %s failed (seed=%s): %s",
                            policy_id[:12],
                            seed,
                            episode.error,
                            extra={
                                "event": "evaluation_failed",
                                "policy_id": policy_id,
                                "seed": seed,
                            },
                        )
                    return job
                except Exception as exc:
                    logging.getLogger(__name__).error(
                        "Policy %s failed (seed=%s): %s",
                        policy_id[:12],
                        seed,
                        exc,
                        extra={
                            "event": "evaluation_failed",
                            "policy_id": policy_id,
                            "seed": seed,
                        },
                    )
                    raise
            finally:
                self._running -= 1
                self._slots.release()
                logging.getLogger(__name__).debug(
                    "Episode %s seed=%s: queued %.4fs, evaluation %.4fs; %s active, %s queued",
                    policy_id[:12],
                    seed,
                    started_at - queued_at,
                    perf_counter() - started_at,
                    self._running,
                    self._queued,
                    extra={
                        "event": "episode_timing",
                        "policy_id": policy_id,
                        "seed": seed,
                        "queue_seconds": started_at - queued_at,
                        "evaluation_seconds": perf_counter() - started_at,
                        "active": self._running,
                        "queued": self._queued,
                    },
                )

        tasks = []
        error = None
        try:
            tasks = [asyncio.create_task(evaluate(job)) for job in jobs]
            self._tasks.update(tasks)
            for task in asyncio.as_completed(tasks):
                try:
                    completed = await task
                except Exception as exc:
                    if error is None:
                        error = exc
                else:
                    yield completed
            if error is not None:
                raise error
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)


async def execute(
    jobs: Iterable[Job],
    *,
    concurrency: int = 1,
    episode_timeout: float = 60.0,
) -> list[Job]:
    """Collect completed jobs; use Executor.execute to persist results as they arrive."""
    async with Executor(concurrency=concurrency, episode_timeout=episode_timeout) as executor:
        return [job async for job in executor.execute(jobs)]
