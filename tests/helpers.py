"""Explicit storage/execution fixtures for optimizer integration tests."""

import asyncio
from copy import deepcopy

import cloudpickle
import gymnasium as gym

from research.rewards import mean_rewards
from research.rollouts import Rollouts
from rsikit import Episode, Evaluator, Executor, Run
from rsikit.evaluation import PolicyError


async def run_episode(make_env, make_policy, *, seed=None, max_steps=None, instructions=None):
    """Own trusted test instances and expose the final transition for legacy assertions."""
    with gym.make(make_env) if isinstance(make_env, str) else make_env() as env:
        if instructions is None:
            instructions = (
                env.get_wrapper_attr("instructions") if env.has_wrapper_attr("instructions") else ""
            )
        policy = make_policy(
            deepcopy(env.observation_space), deepcopy(env.action_space), instructions=instructions
        )
        try:
            episode = await Evaluator(max_steps=max_steps).evaluate(policy, env, seed=seed)
            return episode.final_step
        finally:
            await policy.close()


class recorded_run:
    def __init__(self, path=None, *, name=None, environment, executor=None, **kwargs):
        self.executor = executor if executor is not None else Executor()
        self.run = (
            Run.create(name=name, path=path, **kwargs)
            if name is not None
            else Run.open(path, **kwargs)
        )
        self.rollouts = Rollouts(environment, self.executor, self.run)

    def __enter__(self):
        self.run.__enter__()
        return self.run, self.rollouts

    def __exit__(self, *exc):
        self.run.close()

    async def __aenter__(self):
        await self.executor.__aenter__()
        await self.run.__aenter__()
        return self.run, self.rollouts

    async def __aexit__(self, *exc):
        try:
            await self.executor.__aexit__(*exc)
        finally:
            self.run.close()


async def finish_pending(rollouts):
    """A test controller chooses unfinished work; Run never executes on reopen."""
    for policy in rollouts.run.policies():
        seeds = [seed for seed, score in rollouts.run.scores(policy).items() if score is None]
        if seeds:
            await mean_rewards(rollouts, [policy], seeds=seeds)


class fake_executor:
    """In-process research test double; no Huey resources or production hooks."""

    execute = Executor.execute

    def __init__(self, *, evaluation, concurrency=1, **kwargs):
        self.evaluation = evaluation
        self.concurrency = concurrency
        self._slots = asyncio.Semaphore(concurrency)
        self._queued = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def iterate(self, jobs):
        async def evaluate(job):
            self._queued += 1
            try:
                await self._slots.acquire()
            finally:
                self._queued -= 1
            try:
                policy, environment = cloudpickle.loads(
                    cloudpickle.dumps((job.policy, job.environment))
                )
                try:
                    episode = await self.evaluation.evaluate(policy.source, environment, job.seed)
                except PolicyError as exc:
                    episode = Episode(error=str(exc))
                try:
                    job.result = Episode.from_data(episode.encode())
                except ValueError as exc:
                    from rsikit.evaluation import InfrastructureError

                    raise InfrastructureError(str(exc)) from exc
                return job
            finally:
                self._slots.release()

        tasks = [asyncio.create_task(evaluate(job)) for job in jobs]
        error = None
        try:
            for task in asyncio.as_completed(tasks):
                try:
                    job = await task
                except Exception as exc:
                    error = error or exc
                else:
                    yield job
            if error:
                raise error
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def episodes(scores=None, feedback="", failure=None, accepted=True, *, metrics=None, features=None):
    from rsikit import Episode
    from tests.test_episode_storage import trajectory

    if failure is not None:
        result = {seed: trajectory(score) for seed, score in (scores or {}).items()}
        seed = next((s for s in range(len(result) + 1) if s not in result))
        result[seed] = Episode(error=failure)
        return result
    if not accepted:
        return {}
    result = {seed: trajectory(score) for seed, score in (scores or {}).items()}
    for episode in result.values():
        episode.infos[-1].update(metrics=metrics or {}, features=features or {}, feedback=feedback)
    return result
