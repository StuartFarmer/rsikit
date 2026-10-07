"""Run persistence and executor scheduling without a Docker dependency."""

import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import gymnasium as gym
from slick import prompts
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, SQLModel, select

import rsikit.generation as generation
from research.rewards import mean_rewards
from rsikit import PolicyDefinition, generate
from rsikit.evaluation import InfrastructureError, PolicyError
from rsikit.policy import InvalidPolicy
from tests.helpers import fake_executor, finish_pending, recorded_run
from tests.providers import ScriptedProvider
from tests.test_episode_storage import trajectory

RESPONSE = {
    "name": "Model chose this name",
    "description": "Always push left to establish a baseline.",
    "implementation": "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n",
}


class FakeEvaluation:
    def __init__(self):
        self.evaluate = AsyncMock(return_value=trajectory(7.0, {}))


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run"
        self.env = gym.make("CartPole-v1", max_episode_steps=3)
        self.addCleanup(self.env.close)
        root = patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts")
        root.start()
        self.addCleanup(root.stop)
        self.provider = ScriptedProvider([json.dumps(RESPONSE)])
        self.policy = await generate("test task", provider=self.provider)
        self.evaluation = FakeEvaluation()
        self.executor = fake_executor(evaluation=self.evaluation)

    def create(self, **kwargs):
        return recorded_run(
            name="test", environment=self.env, path=self.path, executor=self.executor, **kwargs
        )

    def reopen(self, path=None):
        return recorded_run(path or self.path, environment=self.env, executor=self.executor)

    async def test_new_candidate_uses_free_worker_before_previous_candidate_finishes(self):
        other = await generate(
            "task", provider=ScriptedProvider([json.dumps({**RESPONSE, "name": "Other"})])
        )
        first_started, second_started, release = (asyncio.Event() for _ in range(3))
        active = peak = 0

        async def evaluate(implementation, environment, seed):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                if seed == 0:
                    first_started.set()
                    await release.wait()
                else:
                    second_started.set()
                    await release.wait()
                return trajectory(float(seed), {})
            finally:
                active -= 1

        self.evaluation.evaluate.side_effect = evaluate
        self.executor = fake_executor(evaluation=self.evaluation, concurrency=2)
        async with self.create() as (run, rollouts):
            first = asyncio.create_task(mean_rewards(rollouts, [self.policy], seeds=[0]))
            await asyncio.wait_for(first_started.wait(), 1)
            second = asyncio.create_task(mean_rewards(rollouts, [other], seeds=[1, 2]))
            duplicate = asyncio.create_task(mean_rewards(rollouts, [other], seeds=[1, 2]))
            try:
                await asyncio.wait_for(second_started.wait(), 1)
                self.assertFalse(first.done())
            finally:
                release.set()
                await asyncio.gather(first, second, duplicate)
            self.assertEqual(run.scores(other), {1: 1, 2: 2})
        self.assertEqual(peak, 2)
        self.assertEqual(self.evaluation.evaluate.await_count, 3)

    async def test_async_run_repairs_and_resumes_batches(self):
        async with self.create() as (run, rollouts):
            await mean_rewards(rollouts, [self.policy], seeds=[0])
            self.evaluation.evaluate.side_effect = PolicyError("bad policy")
            with self.assertRaises(PolicyError):
                await mean_rewards(rollouts, [self.policy], seeds=[1])
            self.evaluation.evaluate.side_effect = None
            await finish_pending(rollouts)
            await mean_rewards(rollouts, [self.policy], seeds=[100])
            self.assertEqual(run.scores(self.policy), {0: 7, 1: 7, 100: 7})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            run.database()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await finish_pending(rollouts)

    async def test_cancel_queued_submission_does_not_interrupt_active_batch(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def evaluate(implementation, environment, seed):
            if seed == 0:
                started.set()
                await release.wait()
            return trajectory(float(seed), {})

        self.evaluation.evaluate.side_effect = evaluate

        async def batch(seeds):
            return [
                item
                async for item in self.executor.evaluate(
                    [(str(seed), RESPONSE["implementation"], seed) for seed in seeds], self.env
                )
            ]

        async with self.executor:
            first = asyncio.create_task(batch([0, 1]))
            await started.wait()
            queued = asyncio.create_task(batch([2]))
            try:
                for _ in range(20):
                    if self.executor._queued == 2:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(self.executor._queued, 2)
                queued.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await queued
                release.set()
                self.assertEqual([seed for _, seed, _ in await first], [0, 1])
            finally:
                release.set()
                queued.cancel()
                await asyncio.gather(first, queued, return_exceptions=True)
        self.assertEqual([call.args[2] for call in self.evaluation.evaluate.call_args_list], [0, 1])

    async def test_resume_only_dispatches_policies_it_locked(self):
        other = await generate(
            "task", provider=ScriptedProvider([json.dumps({**RESPONSE, "name": "Other"})])
        )
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def evaluate(implementation, environment, seed):
            started[seed].set()
            await release[seed].wait()
            return trajectory(float(seed), {})

        self.evaluation.evaluate.side_effect = evaluate
        self.executor = fake_executor(evaluation=self.evaluation, concurrency=3)
        async with self.create() as (run, rollouts):
            first = asyncio.create_task(mean_rewards(rollouts, [self.policy], seeds=[0]))
            await started[0].wait()
            resume = asyncio.create_task(finish_pending(rollouts))
            await asyncio.sleep(0)  # resume has snapshotted A and is waiting on its lock.
            second = asyncio.create_task(mean_rewards(rollouts, [other], seeds=[1]))
            await started[1].wait()
            try:
                release[0].set()
                await first
                await asyncio.wait_for(asyncio.shield(resume), 0.5)
                self.assertFalse(second.done())
            finally:
                release[1].set()
                await asyncio.gather(first, second, resume, return_exceptions=True)
        self.assertEqual([call.args[2] for call in self.evaluation.evaluate.call_args_list], [0, 1])

    async def test_async_run_retries_failed_episode(self):
        async with self.create() as (run, rollouts):
            self.evaluation.evaluate.side_effect = InfrastructureError("worker died")
            with self.assertRaises(InfrastructureError):
                await mean_rewards(rollouts, [self.policy])
            self.evaluation.evaluate.side_effect = None
            await finish_pending(rollouts)
            self.assertEqual(run.scores(self.policy), {0: 7})

    async def test_save_infers_tables_upserts_and_commits_atomically(self):
        class CustomEvaluation(SQLModel, table=True):
            __tablename__ = "test_custom_evaluation"
            id: int | None = Field(default=None, primary_key=True)
            policy_id: str
            score: float

        class Marker(SQLModel, table=True):
            __tablename__ = "test_marker"
            name: str = Field(primary_key=True)
            label: str

        evaluation = CustomEvaluation(policy_id=self.policy.id, score=7)
        with self.create() as (run, rollouts):
            with closing(sqlite3.connect(self.path / "run.sqlite")) as db:
                names = {
                    row[0]
                    for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                self.assertNotIn("test_custom_evaluation", names)
                self.assertNotIn("test_marker", names)
            run.save(evaluation, Marker(name="one", label="baseline"))
            self.assertIsNotNone(evaluation.id)
            evaluation.score = 8
            run.save(evaluation)
            run.save(CustomEvaluation(id=evaluation.id, policy_id=self.policy.id, score=9))
            with self.assertRaises(IntegrityError):
                run.save(
                    CustomEvaluation(id=evaluation.id, policy_id=self.policy.id, score=99),
                    Marker(name="invalid", label=None),
                )
        with self.reopen() as (run, rollouts), run.database() as db:
            rows = db.exec(select(CustomEvaluation)).all()
            self.assertEqual([(row.policy_id, row.score) for row in rows], [(self.policy.id, 9)])
            self.assertEqual([row.label for row in db.exec(select(Marker))], ["baseline"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            run.save(evaluation)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            run.database()

    async def test_policy_scores_exports_and_reuse(self):
        self.assertIn("scipy.linalg.solve_discrete_are", self.provider.calls[0])
        self.assertIn("CPU-only PyTorch", self.provider.calls[0])
        self.assertTrue(isinstance(self.policy, PolicyDefinition))
        self.assertEqual(self.policy.name, RESPONSE["name"])
        self.assertEqual(self.policy.description, RESPONSE["description"])
        with self.create() as (run, rollouts):
            self.assertIs(rollouts.environment, self.env)
            self.assertIs(rollouts.executor, self.executor)
            self.assertEqual(
                await mean_rewards(rollouts, [self.policy], seeds=iter([0, 1, 0])),
                {self.policy.id: 7.0},
            )
            await mean_rewards(rollouts, [self.policy], seeds=[0, 1])
            self.assertEqual(self.evaluation.evaluate.await_count, 2)
            (export,) = (self.path / "exports").glob("*.py")
            restored = PolicyDefinition.from_file(export)
            self.assertEqual(restored.id, self.policy.id)
            self.assertEqual(restored.name, self.policy.name)
            self.assertEqual(restored.description, self.policy.description)
            self.assertEqual(restored.source, RESPONSE["implementation"])
        export.unlink()
        with self.reopen() as (run, rollouts):
            (restored,) = run.policies()
            await finish_pending(rollouts)
            self.assertEqual(restored.id, self.policy.id)
            self.assertEqual(restored.description, RESPONSE["description"])
            self.assertEqual(run.scores(restored), {0: 7.0, 1: 7.0})
            self.assertTrue(export.exists())
        with closing(sqlite3.connect(self.path / "run.sqlite")) as db:
            self.assertEqual(
                [r[1] for r in db.execute("PRAGMA table_info(settings)")], ["name", "export"]
            )
            self.assertEqual(
                [r[1] for r in db.execute("PRAGMA table_info(policy)")],
                ["id", "name", "description", "implementation", "scores"],
            )
        self.assertEqual(len(self.provider.calls), 1)

    async def test_open_upgrades_existing_runs_without_descriptions(self):
        self.path.mkdir()
        with closing(sqlite3.connect(self.path / "run.sqlite")) as db:
            db.executescript("""
                CREATE TABLE settings (name TEXT PRIMARY KEY, export BOOLEAN NOT NULL);
                INSERT INTO settings VALUES ('old run', 0);
                CREATE TABLE policy (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    implementation TEXT NOT NULL, scores JSON NOT NULL);
            """)
            db.execute(
                "INSERT INTO policy VALUES (?, ?, ?, ?)",
                (self.policy.id, self.policy.name, RESPONSE["implementation"], '{"0": 7.0}'),
            )
            db.commit()
        with self.reopen() as (run, rollouts):
            restored = run.policies()[0]
            self.assertEqual(restored.description, "")
            self.assertEqual(run.scores(restored), {0: 7.0})
            self.assertEqual(await mean_rewards(rollouts, [restored]), {restored.id: 7.0})
        with self.reopen() as (run, rollouts):
            self.assertEqual(len(run.policies()), 1)

    async def test_default_seed_means_and_empty_batch(self):
        async def evaluate(implementation, environment, seed):
            return trajectory(seed * 2.0, {})

        self.evaluation.evaluate.side_effect = evaluate
        with self.create() as (run, rollouts):
            self.assertEqual(await mean_rewards(rollouts, []), {})
            self.assertEqual(await mean_rewards(rollouts, [self.policy]), {self.policy.id: 0.0})
            self.assertEqual(
                await mean_rewards(rollouts, [self.policy], seeds=(0, 1, 2)), {self.policy.id: 2.0}
            )
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: 2.0, 2: 4.0})
            self.assertEqual(self.evaluation.evaluate.await_count, 3)
            with self.assertRaisesRegex(ValueError, "at least one seed"):
                await mean_rewards(rollouts, [self.policy], seeds=())

    async def test_interruption_and_resume_after_move(self):
        started = asyncio.Event()

        async def blocked(implementation, environment, seed):
            if seed == 1:
                started.set()
                await asyncio.Event().wait()
            return trajectory(7.0, {})

        self.evaluation.evaluate.side_effect = blocked
        with self.create() as (run, rollouts):
            task = asyncio.create_task(mean_rewards(rollouts, [self.policy], seeds=[0, 1, 2]))
            await started.wait()
            # Allow the completed seed's result to reach the persistence consumer.
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: None, 2: None})
        moved = self.path.with_name("moved")
        shutil.move(self.path, moved)
        self.evaluation.evaluate = AsyncMock(return_value=trajectory(9.0, {}))
        with self.reopen(moved) as (run, rollouts):
            await finish_pending(rollouts)
            self.assertEqual(run.scores(self.policy), {0: 7.0, 1: 9.0, 2: 9.0})
            self.assertEqual([c.args[2] for c in self.evaluation.evaluate.call_args_list], [1, 2])

    async def test_group_failure_preserves_other_scores_and_artifacts(self):
        async def evaluate(implementation, environment, seed):
            if seed == 1:
                raise PolicyError("bad action")
            return trajectory(seed, {"nested/result.txt": str(seed).encode()})

        self.evaluation.evaluate.side_effect = evaluate
        self.executor = fake_executor(evaluation=self.evaluation, concurrency=2)
        with self.create() as (run, rollouts):
            with self.assertRaisesRegex(PolicyError, "bad action"):
                await mean_rewards(rollouts, [self.policy], seeds=[0, 1, 2])
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: None, 2: 2.0})
            self.assertEqual(
                (run.path / "artifacts" / self.policy.id / "2/nested/result.txt").read_bytes(), b"2"
            )
        self.evaluation.evaluate = AsyncMock(return_value=trajectory(3.0, {}))
        with self.reopen() as (run, rollouts):
            await finish_pending(rollouts)
            self.assertEqual(run.scores(self.policy), {0: 0.0, 1: 3.0, 2: 2.0})
            self.assertEqual(self.evaluation.evaluate.await_count, 1)

    async def test_executor_bounds_workers_and_closes_once(self):
        other = await generate(
            "task", provider=ScriptedProvider([json.dumps({**RESPONSE, "name": "Other"})])
        )
        started, release = asyncio.Event(), asyncio.Event()
        active = peak = calls = 0

        async def evaluate(*args):
            nonlocal active, peak, calls
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
            try:
                await release.wait()
                return trajectory(7.0, {})
            finally:
                active -= 1

        self.evaluation.evaluate.side_effect = evaluate
        self.executor = fake_executor(evaluation=self.evaluation, concurrency=2)
        with self.create() as (run, rollouts):
            task = asyncio.create_task(mean_rewards(rollouts, [self.policy, other], seeds=[0, 1]))
            await asyncio.wait_for(started.wait(), 2)
            self.assertEqual(calls, 2)
            release.set()
            result = await task
            self.assertEqual(result, {self.policy.id: 7.0, other.id: 7.0})
        self.assertEqual(peak, 2)

    async def test_cancellation_waits_for_workers(self):
        started = asyncio.Event()
        active = 0

        async def evaluate(*args):
            nonlocal active
            active += 1
            if active == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                active -= 1

        self.evaluation.evaluate.side_effect = evaluate
        self.executor = fake_executor(evaluation=self.evaluation, concurrency=2)
        async with self.create() as (run, rollouts):
            task = asyncio.create_task(mean_rewards(rollouts, [self.policy], seeds=[0, 1, 2]))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(run.scores(self.policy), {0: None, 1: None, 2: None})
        self.assertEqual(active, 0)

    async def test_invalid_results_artifact_paths_and_startup_failure_remain_pending(self):
        mutated = trajectory()
        mutated.rewards[0] = float("nan")
        with self.create() as (run, rollouts):
            for outcome, diagnostic in [
                (mutated, "finite"),
                (trajectory(1.0, {"../../../../outside": b"bad"}), "Artifact path"),
            ]:
                self.evaluation.evaluate.return_value = outcome
                with self.assertRaisesRegex(ValueError, diagnostic):
                    await mean_rewards(rollouts, [self.policy])
                self.assertEqual(run.scores(self.policy), {0: None})
            self.evaluation.evaluate.side_effect = InfrastructureError("start failed")
            with self.assertRaisesRegex(InfrastructureError, "start failed"):
                await finish_pending(rollouts)
        self.assertFalse((self.path.parent / "outside").exists())

    async def test_run_ownership_identity_and_export_opt_out(self):
        with self.create(export=False) as (run, rollouts):
            with self.assertRaises(BlockingIOError):
                self.reopen()
            await mean_rewards(rollouts, [self.policy])
            from rsikit.run import _StoredPolicy

            with run.database() as db:
                stored = db.get(_StoredPolicy, self.policy.id)
                stored.name = "changed"
                db.add(stored)
                db.commit()
            with self.assertRaisesRegex(ValueError, "cannot change"):
                await mean_rewards(rollouts, [self.policy])
            self.assertFalse((self.path / "exports").exists())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await finish_pending(rollouts)

    async def test_generation_never_executes_code_and_export_name_is_safe(self):
        response = {
            **RESPONSE,
            "name": "../../outside",
            "implementation": "raise AssertionError('host execution')\n"
            + RESPONSE["implementation"],
        }
        policy = await generate("task", provider=ScriptedProvider([json.dumps(response)]))
        with self.create() as (run, rollouts):
            await mean_rewards(rollouts, [policy])
            (export,) = (self.path / "exports").glob("*.py")
            self.assertEqual(export.parent, self.path / "exports")

        candidate = await generate(
            "task",
            provider=ScriptedProvider([json.dumps({**RESPONSE, "implementation": "pass"})]),
        )
        self.assertEqual(candidate.source, "pass")
        with self.assertRaisesRegex(InvalidPolicy, "Solution"):
            candidate.validate()


if __name__ == "__main__":
    unittest.main()
