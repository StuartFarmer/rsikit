"""Explicit storage/execution fixtures for optimizer integration tests."""

from research.rewards import mean_rewards
from research.rollouts import Rollouts
from rsikit import Executor, Run
from rsikit.sandbox import _run_episode


async def run_episode(*args, **kwargs):
    """Exercise worker lifecycle preparation while retaining concise legacy assertions."""
    return (await _run_episode(*args, **kwargs)).final_step


class recorded_run:
    def __init__(self, path=None, *, name=None, environment, executor=None, **kwargs):
        self.executor = executor if executor is not None else Executor()
        self.run = (
            Run.create(name=name, path=path, **kwargs) if name is not None else Run.open(path)
        )
        self.rollouts = Rollouts(environment, self.executor, self.run)

    def __enter__(self):
        return self.run, self.rollouts

    def __exit__(self, *exc):
        self.run.close()

    async def __aenter__(self):
        await self.executor.__aenter__()
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
