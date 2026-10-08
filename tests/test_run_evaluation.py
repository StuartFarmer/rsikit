"""Run owns evaluation reuse, partial results, and streaming cleanup."""

import asyncio
import tempfile
import unittest
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym

from rsikit import Episode, PolicyDefinition, Run
from rsikit.evaluation import InfrastructureError, PolicyError, episode_error, episode_scores
from tests.helpers import fake_executor
from tests.test_episode_storage import trajectory
from tests.test_evaluation import SOURCE
from tests.test_run import FakeEvaluation


class RunEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Run.create(name="evaluate", path=Path(directory.name) / "run")
        self.addCleanup(self.run.close)
        self.environment = gym.make("CartPole-v1")
        self.addCleanup(self.environment.close)
        self.backend = FakeEvaluation()
        self.executor = fake_executor(evaluation=self.backend, concurrency=2)
        self.options = dict(environment=self.environment, executor=self.executor)
        self.policy = PolicyDefinition.from_text(SOURCE)
        self.other = PolicyDefinition.from_text(SOURCE + "# other\n")

    async def test_requested_scores_reuse_episodes_not_legacy_scalars(self):
        self.run.save_policy(self.policy, scores={0: 999, 99: 999})
        self.backend.evaluate.side_effect = lambda source, env, seed: trajectory(2 + 2 * seed)
        measured = await self.run.evaluate(
            [self.policy, self.policy], seeds=iter([0, 1, 0]), **self.options
        )
        self.assertEqual(episode_scores(measured[self.policy.id]), {0: 2, 1: 4})
        self.assertEqual(
            await self.run.mean_scores([self.policy], seeds=[0, 1], **self.options),
            {self.policy.id: 3},
        )
        self.assertEqual(self.backend.evaluate.await_count, 2)
        self.assertEqual(self.run.scores(self.policy), {0: 2, 1: 4, 99: 999})
        self.assertEqual(await self.run.evaluate([], **self.options), {})
        for seeds in ([], [True], [1, True]):
            with self.assertRaises(ValueError):
                await self.run.evaluate([self.policy], seeds=seeds, **self.options)

    async def test_failures_retain_partial_panels_and_aggregate_diagnostics(self):
        async def evaluate(source, env, seed):
            if seed == 1:
                raise PolicyError("bad action")
            return trajectory(2)

        self.backend.evaluate.side_effect = evaluate
        policies = [self.policy, self.other]
        results = await self.run.evaluate(policies, seeds=[0, 1], **self.options)
        for policy in policies:
            self.assertEqual(episode_scores(results[policy.id]), {0: 2})
            self.assertIsNotNone(episode_error(results[policy.id]))
            self.assertEqual(self.run.scores(policy), {0: 2, 1: None})
        with self.assertRaises(PolicyError) as raised:
            await self.run.mean_scores(policies, seeds=[0, 1], **self.options)
        self.assertEqual(set(raised.exception.failures), {p.id for p in policies})
        self.backend.evaluate.side_effect = None
        scores = await self.run.mean_scores(policies, seeds=[0, 1], **self.options)
        self.assertEqual(set(scores.values()), {4.5})
        self.backend.evaluate.side_effect = InfrastructureError("offline")
        with self.assertRaises(InfrastructureError):
            await self.run.evaluate(policies, seeds=[2], **self.options)

    async def test_explicit_fitness_is_persisted_without_changing_raw_rewards(self):
        episode = trajectory(2)
        episode.infos[-1]["fitness"] = 42.0
        self.backend.evaluate.return_value = episode
        results = await self.run.evaluate([self.policy], **self.options)
        self.assertEqual(results[self.policy.id][0].total_reward, 2)
        self.assertEqual(self.run.scores(self.policy), {0: 42})
        self.assertEqual(
            await self.run.mean_scores([self.policy], **self.options), {self.policy.id: 42}
        )
        self.assertEqual(self.backend.evaluate.await_count, 1)

    async def test_overlapping_work_is_reused_and_distinct_policies_overlap(self):
        started, both, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active = 0

        async def evaluate(source, env, seed):
            nonlocal active
            active += 1
            started.set()
            if active == 2:
                both.set()
            try:
                await release.wait()
                return trajectory(3)
            finally:
                active -= 1

        self.backend.evaluate.side_effect = evaluate
        first = asyncio.create_task(self.run.evaluate([self.policy], **self.options))
        await asyncio.wait_for(started.wait(), 1)
        duplicate = asyncio.create_task(self.run.evaluate([self.policy], **self.options))
        other = asyncio.create_task(self.run.evaluate([self.other], **self.options))
        try:
            await asyncio.wait_for(both.wait(), 1)
        finally:
            release.set()
            await asyncio.gather(first, duplicate, other)
        self.assertEqual(self.backend.evaluate.await_count, 2)

    async def test_early_stream_close_cancels_owned_jobs_and_releases_lock(self):
        cancelled, waiting = asyncio.Event(), asyncio.Event()

        async def evaluate(source, env, seed):
            if seed == 0:
                await waiting.wait()
                return trajectory(3)
            waiting.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.backend.evaluate.side_effect = evaluate
        async with aclosing(
            self.run.collect([self.policy], seeds=[0, 1], **self.options)
        ) as stream:
            self.assertEqual((await anext(stream))[1], 0)
        self.assertTrue(cancelled.is_set())
        self.backend.evaluate.side_effect = None
        self.assertEqual(
            await asyncio.wait_for(self.run.mean_scores([self.policy], **self.options), 1),
            {self.policy.id: 3},
        )

    async def test_invalid_executor_results_close_the_stream_and_release_locks(self):
        for mode in ("unexpected", "duplicate", "missing", "invalid_episode", "boolean_seed"):
            with self.subTest(mode=mode):
                closed = False

                async def iterate(jobs):
                    nonlocal closed
                    try:
                        job = jobs[0]
                        if mode == "missing":
                            return
                        job.result = Episode() if mode == "invalid_episode" else trajectory(3)
                        if mode == "unexpected":
                            job.seed = 99
                        elif mode == "boolean_seed":
                            job.seed = True
                        yield job
                        if mode == "duplicate":
                            yield job
                    finally:
                        closed = True

                # Use fresh seeds so the preceding case's valid result cannot satisfy this one.
                seed = 1 if mode == "boolean_seed" else 10 + len(mode)
                with self.assertRaises((ValueError, RuntimeError)):
                    await self.run.evaluate(
                        [self.policy],
                        seeds=[seed],
                        environment=self.environment,
                        executor=SimpleNamespace(iterate=iterate, concurrency=1),
                    )
                self.assertTrue(closed)
                await asyncio.wait_for(
                    self.run.evaluate([self.policy], seeds=[seed], **self.options), 1
                )


if __name__ == "__main__":
    unittest.main()
