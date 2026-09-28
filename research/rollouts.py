"""Collect seed panels for experiments, explicitly recording raw episodes in a Run."""

import asyncio
import logging
from contextlib import AsyncExitStack, aclosing

from rsikit import Executor, Run


class Rollouts:
    """Research orchestration; the caller owns the environment, executor and storage.

    Cached episodes are reusable within this experiment's fixed environment/configuration.
    No scoring, screening, optimizer updates, or resource lifecycle is performed here.
    """

    def __init__(self, environment, executor: Executor, run: Run):
        self.environment, self.executor, self.run = environment, executor, run
        self._locks = {}
        name = getattr(getattr(environment, "spec", None), "id", None) or type(environment).__name__
        logging.getLogger(__name__).info(
            "Environment: %s", name, extra={"progress": dict(kind="environment", name=name)}
        )

    async def collect(self, policies, *, seeds=(0,)):
        seeds = tuple(dict.fromkeys(seeds))
        if not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("Evaluation requires at least one seed; seed IDs must be integers")
        policies = {policy.id: policy for policy in policies}
        async with AsyncExitStack() as stack:
            for policy_id in sorted(policies):
                await stack.enter_async_context(self._locks.setdefault(policy_id, asyncio.Lock()))
            jobs = []
            for policy in policies.values():
                self.run.save_policy(policy)
                for seed in seeds:
                    episode = self.run.load_episode(policy, seed)
                    if episode is None:
                        jobs.append((policy.id, policy._implementation, seed))
                    else:
                        yield policy.id, seed, episode
            requested = {(policy_id, seed) for policy_id, _, seed in jobs}
            async with aclosing(self.executor.evaluate(jobs, self.environment)) as results:
                async for policy_id, seed, episode in results:
                    if (policy_id, seed) not in requested:
                        raise ValueError("Executor returned an unexpected or duplicate result")
                    self.run.save_episode(policies[policy_id], seed, episode)
                    requested.remove((policy_id, seed))
                    yield policy_id, seed, episode
            if requested:
                raise RuntimeError("Executor finished without returning all requested results")
