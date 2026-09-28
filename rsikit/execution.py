"""Local episode processes; run this application inside Docker for host isolation."""

import asyncio
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import signal
import sys
import traceback
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cloudpickle
import gymnasium as gym

from .episode import Episode, encode_episode
from .evaluation import InfrastructureError, PolicyError, PolicyTimeout, _run_episode
from .policy import MAX_SOURCE, load_policy

MAX_RESULT = 64 * 1024 * 1024


def _run_child(channel, source, definition, seed, options, directory):
    """One process owns the environment, actual Solution, logs and artifacts."""
    os.setsid()
    os.chdir(directory)
    with channel, open("episode.log", "w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
        try:
            env = cloudpickle.loads(definition)
            wrapper, index = env, 0
            while isinstance(wrapper, gym.Wrapper):
                if isinstance(wrapper, gym.wrappers.RecordVideo):
                    wrapper.video_folder = str(Path(directory) / "videos" / str(index))
                    Path(wrapper.video_folder).mkdir(parents=True)
                    index += 1
                wrapper = wrapper.env

            def make_policy(observation_space, action_space, *, instructions):
                return load_policy(source, observation_space, action_space, instructions)

            episode = asyncio.run(
                _run_episode(
                    lambda: env, make_policy, env_seed=seed, _policy_errors=True, **options
                )
            )
            sys.stdout.flush()
            sys.stderr.flush()
            episode.artifacts.update(episode.infos[-1].get("artifacts", {}))
            size = sum(len(data) for data in episode.artifacts.values())
            for path in Path(directory).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    size += path.stat().st_size
                    if size > MAX_RESULT:
                        raise InfrastructureError("Evaluation artifacts exceed 64 MiB")
                    if path.stat().st_size:
                        episode.artifacts[str(path.relative_to(directory))] = path.read_bytes()
            if len(json.dumps(encode_episode(episode)).encode()) > MAX_RESULT:
                raise InfrastructureError("Saved episode exceeds 64 MiB")
            result = episode
        except BaseException as exc:
            result = (
                "policy" if isinstance(exc, PolicyError) else "infrastructure",
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        if len(data) > MAX_RESULT:
            data = pickle.dumps(("infrastructure", "Episode exceeds 64 MiB"))
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

    async def _evaluate(self, source, definition, seed, **options):
        options.setdefault("policy_seed", seed)
        with TemporaryDirectory(prefix="rsikit-episode-") as directory:
            receiving, sending = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_run_child, args=(sending, source, definition, seed, options, directory)
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
                result = pickle.loads(data)
                if isinstance(result, Episode):
                    return result
                kind, message = result
                raise (PolicyError if kind == "policy" else InfrastructureError)(message)
            except (PolicyError, InfrastructureError) as exc:
                path = Path(directory) / "episode.log"
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 4096))
                        tail = stream.read(4096).decode(errors="replace")
                    if tail:
                        exc.args = (f"{exc}\nEpisode log (tail):\n{tail}",)
                raise
            finally:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Child may not yet have reached setsid().
                    except PermissionError:
                        # Darwin returns EPERM for an already-exited, empty process group.
                        if sys.platform != "darwin" or process.is_alive():
                            raise
                    if process.is_alive():
                        process.kill()
                    process.join()
                process.close()
                sending.close()
                if reading is not None:
                    await asyncio.gather(reading, return_exceptions=True)
                receiving.close()

    async def evaluate(
        self,
        jobs: Iterable[tuple[str, str, int]],
        environment: gym.Env,
    ) -> AsyncIterator[tuple[str, int, Episode]]:
        """Yield complete episodes, retaining successful siblings when a policy fails."""
        jobs = list(jobs)
        if not jobs:
            return
        # Only caller-provided environments are serialized, never returned worker objects.
        definition = cloudpickle.dumps(environment)

        async def evaluate(policy_id, implementation, seed):
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
                    episode = await self._evaluate(
                        implementation,
                        definition,
                        seed,
                    )
                    return policy_id, seed, episode
                except asyncio.CancelledError:
                    raise
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
                    if isinstance(exc, PolicyError):
                        return policy_id, seed, exc
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
        policy_error = None
        failures = {}
        try:
            tasks = [asyncio.create_task(evaluate(*job)) for job in jobs]
            self._tasks.update(tasks)
            for task in asyncio.as_completed(tasks):
                try:
                    completed = await task
                except Exception as exc:
                    if error is None:
                        error = exc
                else:
                    policy_id, seed, result = completed
                    if isinstance(result, PolicyError):
                        policy_error = policy_error or result
                        diagnostic = f"seed={seed}: {type(result).__name__}: {result}"
                        failures[policy_id] = "\n".join(
                            filter(None, [failures.get(policy_id), diagnostic])
                        )
                    else:
                        yield completed
            if error is not None:
                raise error
            if policy_error is not None:
                policy_error.failures = failures
                raise policy_error
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)


async def run_program(
    program: Path,
    make_env,
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
    episode_timeout: float = 60.0,
) -> Episode:
    """Run a saved policy in a fresh child of the current application."""
    executor = Executor(episode_timeout=episode_timeout)
    try:
        with Path(program).open(encoding="utf-8") as stream:
            source = stream.read(MAX_SOURCE + 1)
    except (OSError, UnicodeError) as exc:
        raise InfrastructureError(f"Cannot read policy source: {exc}") from exc
    if len(source.encode()) > MAX_SOURCE:
        raise PolicyError("Source exceeds 64 KiB")
    env = gym.make(make_env) if isinstance(make_env, str) else make_env()
    try:
        return await executor._evaluate(
            source,
            cloudpickle.dumps(env),
            env_seed,
            policy_seed=policy_seed,
            max_steps=max_steps,
            instructions=instructions,
        )
    finally:
        primary = sys.exc_info()[1]
        try:
            env.close()
        except BaseException:
            if primary is None:
                raise
            logging.getLogger(__name__).exception("Environment cleanup failed during evaluation")
