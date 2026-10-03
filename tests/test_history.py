"""Persist the data needed for island curves, ancestry, and failed-run analysis."""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from slick import prompts
from slick.providers import ProviderError
from sqlmodel import select

from examples.alphaevolve import run_search
from research import alphaevolve
from research.alphaevolve import improved, original
from research.alphaevolve.history import Evaluation, Generation
from tests.helpers import fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeEvaluation


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_variants_preserve_islands_parents_resets_and_append_generations(self):
        for variant in (original, improved):
            with (
                self.subTest(variant=variant.__name__),
                tempfile.TemporaryDirectory() as directory,
                gym.make("CartPole-v1") as env,
                patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            ):
                path = Path(directory) / "run"
                evaluation = FakeEvaluation()

                async def evaluate(implementation, environment, seed):
                    return trajectory(
                        float(implementation.split("return ")[1].split()[0]) + seed, {}
                    )

                evaluation.evaluate.side_effect = evaluate
                agent = variant.AlphaEvolve(
                    "task",
                    ScriptedProvider([program(i) for i in range(5)]),
                    config=variant.Config(islands=2, mode="rewrite", reset_interval=3),
                )
                with recorded_run(
                    name="history",
                    path=path,
                    environment=env,
                    executor=fake_executor(evaluation=evaluation),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts):
                    await run_search(
                        agent,
                        run,
                        rollouts,
                        generations=2,
                        batch_size=2,
                        seeds=(0, 1),
                    )
                    await run_search(
                        agent,
                        run,
                        rollouts,
                        generations=1,
                        batch_size=1,
                        seeds=(0, 1),
                    )
                with recorded_run(path, environment=env) as (run, rollouts), run.database() as db:
                    generations = db.exec(select(Generation).order_by(Generation.number)).all()
                    evaluations = db.exec(
                        select(Evaluation).order_by(Evaluation.generation, Evaluation.attempt)
                    ).all()
                    self.assertEqual([row.number for row in generations], [1, 2, 3])
                    self.assertTrue(all(row.complete for row in generations))
                    self.assertTrue(all(row.seeds == [0, 1] for row in generations))
                    self.assertTrue(
                        all(row.optimizer == variant.AlphaEvolve.__module__ for row in generations)
                    )
                    expected = [1.5, 1.5] if variant is original else [0.5, 1.5]
                    self.assertEqual([item["score"] for item in generations[0].islands], expected)
                    self.assertEqual(len(evaluations), 5)
                    self.assertEqual([row.score for row in evaluations], [0.5, 1.5, 2.5, 3.5, 4.5])
                    self.assertTrue(all(row.status == "evaluated" for row in evaluations))
                    parents = {row.policy_id for row in evaluations[:2]}
                    self.assertTrue(all(row.parent_id in parents for row in evaluations[2:4]))
                    (reset,) = generations[1].resets
                    self.assertEqual(reset["completed"], 3)
                    self.assertNotEqual(reset["reset"], reset["donor"])
                    self.assertIn(reset["founder"], {row.policy_id for row in evaluations})
                    self.assertEqual(run.scores(run.policies()[0]), {0: 0.0, 1: 1.0})

    async def test_provider_failure_and_cancellation_keep_incomplete_history(self):
        for cancelled in (False, True):
            with (
                self.subTest(cancelled=cancelled),
                tempfile.TemporaryDirectory() as directory,
                gym.make("CartPole-v1") as env,
                patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            ):
                path = Path(directory) / "run"
                provider = ScriptedProvider(
                    [program(0)] if cancelled else [program(0), ProviderError("offline")]
                )
                agent = improved.AlphaEvolve("task", provider, config=improved.Config(islands=1))
                evaluation = FakeEvaluation()
                started = asyncio.Event()

                async def blocked(*args):
                    started.set()
                    await asyncio.Event().wait()

                if cancelled:
                    evaluation.evaluate.side_effect = blocked
                with recorded_run(
                    name="interrupted",
                    path=path,
                    environment=env,
                    executor=fake_executor(evaluation=evaluation),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts):
                    task = asyncio.create_task(
                        run_search(
                            agent,
                            run,
                            rollouts,
                            generations=2,
                            batch_size=1,
                        )
                    )
                    if cancelled:
                        await asyncio.wait_for(started.wait(), 2)
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancelled else ProviderError):
                        await task
                with recorded_run(path, environment=env) as (run, rollouts), run.database() as db:
                    rows = db.exec(select(Generation).order_by(Generation.number)).all()
                    self.assertFalse(rows[-1].complete)
                    if not cancelled:
                        self.assertTrue(rows[0].complete)
                    evaluation = db.exec(
                        select(Evaluation).where(Evaluation.generation == rows[-1].number)
                    ).one()
                    self.assertEqual(evaluation.status, "generated" if cancelled else "rejected")
                    self.assertEqual(evaluation.island, 0)
                    if not cancelled:
                        self.assertIn("offline", evaluation.error)
