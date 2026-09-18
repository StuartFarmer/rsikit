"""Evaluate groups inside one sandbox; Run only persists returned results."""

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

import cloudpickle
import gymnasium as gym
from pydantic import BaseModel, Field, FiniteFloat

from .episode import PolicyError
from .sandbox.docker import DockerSandbox


class Sandbox(Protocol):
    async def start(self, workers: int) -> None: ...
    async def evaluate(
        self, implementation: str, environment: bytes, seed: int, call_timeout: float
    ) -> tuple[float, dict[str, bytes]]: ...
    async def close(self) -> None: ...


class Result(BaseModel):
    score: FiniteFloat
    artifacts: dict[str, bytes] = Field(default_factory=dict)


class Executor:
    """Own worker concurrency, policy call timeouts, and sandbox lifecycle."""

    def __init__(
        self, *, sandbox: Sandbox | None = None, concurrency: int = 1, call_timeout: float = 10.0
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.sandbox = DockerSandbox() if sandbox is None else sandbox
        self.concurrency = concurrency
        self.call_timeout = call_timeout
        self._busy = asyncio.Lock()

    async def evaluate(
        self,
        jobs: Iterable[tuple[str, str, int]],
        environment: gym.Env,
    ) -> AsyncIterator[tuple[str, int, Result]]:
        """Yield completed (policy ID, seed, result) tuples from one sandbox per batch."""
        jobs = list(jobs)
        if not jobs:
            return
        # Only caller-provided environments are serialized, never returned worker objects.
        definition = cloudpickle.dumps(environment)
        async with self._busy:
            slots = asyncio.Semaphore(self.concurrency)

            async def evaluate(policy_id, implementation, seed):
                async with slots:
                    try:
                        score, artifacts = await self.sandbox.evaluate(
                            implementation,
                            definition,
                            seed,
                            self.call_timeout,
                        )
                        return policy_id, seed, Result(score=score, artifacts=artifacts)
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

            tasks = []
            error = None
            policy_error = None
            failures = {}
            try:
                logging.getLogger(__name__).info("Starting %s", type(self.sandbox).__name__)
                starting = asyncio.create_task(self.sandbox.start(self.concurrency))
                try:
                    await asyncio.shield(starting)
                except asyncio.CancelledError:
                    await starting
                    raise
                tasks = [asyncio.create_task(evaluate(*job)) for job in jobs]
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
                primary = sys.exc_info()[1]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await self.sandbox.close()
                    logging.getLogger(__name__).info("Sandbox closed")
                except Exception:
                    if primary is None:
                        raise
                    logging.getLogger(__name__).exception(
                        "Sandbox cleanup failed while handling an evaluation error"
                    )
