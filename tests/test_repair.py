"""Diagnostic-driven repair before generation returns and after sandbox failures."""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from pydantic import ValidationError
from rich.console import Console
from slick import prompts
from slick.providers import ProviderError
from sqlmodel import select

from examples.alphaevolve import run_search
from research import alphaevolve
from research.alphaevolve import improved, original, paper
from research.alphaevolve.history import Evaluation, Generation
from research.alphaevolve.improved import AlphaEvolve, Config
from rsikit import Executor
from rsikit.evaluation import InfrastructureError, PolicyError
from rsikit.generation.edits import Program
from tests.helpers import recorded_run
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import SOURCE, program
from tests.test_episode_storage import trajectory
from tests.test_run import FakeSandbox

ROOT = Path(alphaevolve.__file__).parent


class RepairTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_runtime_repairs_overlap_within_generation_limit_for_all_variants(self):
        for variant in (paper, original, improved):
            with self.subTest(variant=variant.__name__), tempfile.TemporaryDirectory() as directory:
                provider = ScriptedProvider([program(i) for i in (9, 8, 7, 9, 0, 1, 2)])
                acall = provider.acall
                started, release = asyncio.Event(), asyncio.Event()
                active = peak = 0

                async def respond(*args, **kwargs):
                    nonlocal active, peak
                    index = len(provider.calls)
                    response = await acall(*args, **kwargs)
                    if index < 4:
                        return response
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        started.set()
                    try:
                        await release.wait()
                        return response
                    finally:
                        active -= 1

                provider.acall = respond
                agent = variant.AlphaEvolve("task", provider, config=variant.Config(islands=1))
                if variant is paper:
                    self.addCleanup(agent.close)
                sandbox = FakeSandbox()

                async def evaluate(source, environment, seed, timeout):
                    if any(f"return {i}" in source for i in (9, 8, 7)):
                        raise PolicyError("bad action")
                    return trajectory(7.0, {})

                sandbox.evaluate.side_effect = evaluate
                with (
                    gym.make("CartPole-v1") as env,
                    recorded_run(
                        name="concurrent-repairs",
                        path=Path(directory) / "run",
                        environment=env,
                        executor=Executor(sandbox=sandbox, concurrency=3),
                    ) as (run, rollouts),
                ):
                    task = asyncio.create_task(
                        run_search(
                            agent,
                            run,
                            rollouts,
                            generations=1,
                            batch_size=4,
                            generation_concurrency=2,
                            console=Console(file=io.StringIO()),
                        )
                    )
                    try:
                        await asyncio.wait_for(started.wait(), 1)
                        self.assertEqual(len(provider.calls), 6)
                        self.assertEqual(active, 2)
                        release.set()
                        await asyncio.wait_for(task, 2)
                    finally:
                        release.set()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    self.assertEqual((peak, active), (2, 0))
                    self.assertEqual((agent.completed, agent.repair_calls), (4, 3))
                    self.assertEqual(agent._pending, {})
                    with run.database() as db:
                        rows = db.exec(select(Evaluation)).all()
                        self.assertEqual(
                            sorted(row.status for row in rows), ["evaluated"] * 4 + ["failed"] * 4
                        )

    async def test_shadowed_solution_is_repaired_and_rechecked(self):
        duplicate = program(0).model_copy(
            update={"implementation": SOURCE + "\nclass Solution(Policy):\n    pass\n"}
        )
        still_duplicate = duplicate.model_copy(
            update={"implementation": duplicate.implementation.replace("return 0", "return 1")}
        )
        provider = ScriptedProvider([duplicate, still_duplicate, program(2)])
        agent = AlphaEvolve("task", provider, config=Config(max_repairs=2))
        policies = await agent.generate()
        self.assertEqual([p._implementation for p in policies], [program(2).implementation])
        self.assertEqual(agent.repair_calls, 2)
        self.assertIn("exactly one top-level Solution class", provider.calls[1])
        self.assertIn("exactly one top-level Solution class", provider.calls[2])

    async def test_exhausted_proposal_does_not_cancel_concurrent_survivor(self):
        broken = program(0).model_copy(update={"implementation": SOURCE + "}"})
        provider = ScriptedProvider([broken, program(1), broken])
        original = provider.acall
        repaired, release = asyncio.Event(), asyncio.Event()
        sibling_started = asyncio.Event()

        async def delayed(*args, **kwargs):
            response = await original(*args, **kwargs)
            if len(provider.calls) == 1:
                await sibling_started.wait()
            elif len(provider.calls) == 2:
                sibling_started.set()
                await release.wait()
            elif len(provider.calls) == 3:
                repaired.set()
            return response

        provider.acall = delayed
        agent = AlphaEvolve("task", provider, config=Config(max_repairs=1))
        async with asyncio.timeout(2):
            task = asyncio.create_task(agent.generate(n=2, concurrency=2))
            try:
                await repaired.wait()
                await asyncio.sleep(0)
                release.set()
                policies = await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual([p.name for p in policies], ["Policy 1"])
        self.assertEqual((agent.generation_calls, agent.repair_calls), (2, 1))
        self.assertEqual(agent.attempts[0]["status"], "discarded")
        agent.update({policies[0].id: 7})
        self.assertEqual(agent.completed, 1)
        self.assertEqual(agent._pending, {})

    async def test_search_continues_after_empty_generation_and_exhausted_runtime_repairs(self):
        broken = program(0).model_copy(update={"implementation": SOURCE + "}"})
        provider = ScriptedProvider(
            [
                broken,
                broken,
                broken,
                broken,  # Both first-generation proposals die.
                program(0),
                program(9),
                program(8),  # One survivor; runtime repair also fails.
                program(1),
                broken,
                broken,  # Later generation still improves the survivor.
            ]
        )
        agent = AlphaEvolve(
            "task", provider, config=Config(islands=1, max_repairs=1, mode="rewrite")
        )
        sandbox = FakeSandbox()
        checked = []

        async def evaluate(implementation, environment, seed, call_timeout):
            checked.append((implementation, seed))
            if "return 9" in implementation or "return 8" in implementation:
                if seed == 1:
                    raise PolicyError("Action outside action_space")
                return trajectory(100.0, {})  # A partial success must never enter selection.
            return trajectory(8.0 if "return 1" in implementation else 7.0, {})

        sandbox.evaluate.side_effect = evaluate
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=3) as env,
            recorded_run(
                name="discard",
                path=Path(directory) / "run",
                environment=env,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as (run, rollouts),
        ):
            await run_search(
                agent,
                run,
                rollouts,
                generations=3,
                batch_size=2,
                generation_concurrency=1,
                seeds=[0, 1],
                console=Console(file=output, width=140),
            )
            self.assertEqual(agent.completed, 2)
            self.assertEqual(agent.best.name, "Policy 1")
            self.assertEqual(agent.islands[0].seed_scores, {0: 8.0, 1: 8.0})
            self.assertIn('Per-seed rewards: {"0": 7.0, "1": 7.0}', provider.calls[-2])
            self.assertEqual(agent._pending, {})
            self.assertEqual((agent.generation_calls, agent.repair_calls), (6, 4))
            self.assertEqual(len(checked), 8)
            self.assertEqual(len(run.policies()), 4)
            self.assertEqual(sum(row["status"] == "discarded" for row in agent.attempts), 4)
            self.assertIn("No surviving policies", output.getvalue())
            self.assertIn("Generation 3/3", (run.path / "run.log").read_text())
            with run.database() as session:
                history = session.exec(
                    select(Evaluation).order_by(Evaluation.attempt, Evaluation.revision)
                ).all()
                generations = session.exec(select(Generation).order_by(Generation.number)).all()
                self.assertEqual(len(history), 7)  # Six proposals; one runtime replacement.
                self.assertEqual(
                    [row.status for row in history],
                    [
                        "discarded",
                        "discarded",
                        "evaluated",
                        "failed",
                        "discarded",
                        "evaluated",
                        "discarded",
                    ],
                )
                failed, repaired = history[3:5]
                self.assertEqual((failed.attempt, failed.revision, repaired.revision), (4, 0, 1))
                self.assertNotEqual(failed.policy_id, repaired.policy_id)
                self.assertIn("Action outside action_space", failed.error)
                self.assertEqual(repaired.repairs, 1)
                self.assertIsNone(repaired.score)  # No aggregate from incomplete episodes.
                self.assertTrue(all(row.complete for row in generations))
                self.assertIsNone(generations[0].islands[0]["score"])
                self.assertEqual([row.islands[0]["score"] for row in generations[1:]], [7.0, 8.0])

    async def test_repairs_syntax_and_schema_with_exact_budget(self):
        broken = Program(name="Broken", description="A baseline.", implementation=SOURCE + "}\n")
        provider = ScriptedProvider([broken, "not json", program(0), program(1)])
        agent = AlphaEvolve("task", provider)
        policies = await agent.generate(n=2, concurrency=1)
        self.assertEqual([p.name for p in policies], ["Policy 0", "Policy 1"])
        self.assertEqual((agent.generation_calls, agent.repair_calls), (2, 2))
        self.assertIn("unmatched", provider.calls[1])
        self.assertIn("}\n", provider.calls[1])
        self.assertIn("not json", provider.calls[2])
        self.assertEqual(len(agent.attempts[0]["repairs"]), 2)
        self.assertIsNone(agent.best)
        agent.update({p.id: 1 for p in policies})
        self.assertEqual(agent.completed, 2)

    async def test_repair_preserves_original_parent_boundaries(self):
        escaped = program(1).model_copy(
            update={
                "implementation": program(1).implementation.replace("from rsikit", "from elsewhere")
            }
        )
        provider = ScriptedProvider(
            [
                program(0),
                escaped,
                escaped.model_copy(
                    update={
                        "implementation": escaped.implementation.replace("return 1", "return 2")
                    }
                ),
                program(2),
            ]
        )
        agent = AlphaEvolve("task", provider, config=Config(islands=1, mode="rewrite"))
        initial = (await agent.generate())[0]
        agent.update({initial.id: 0})
        child = (await agent.generate())[0]
        self.assertEqual(child._implementation, program(2).implementation)
        self.assertIn(SOURCE, provider.calls[-1])
        self.assertIn("immutable", provider.calls[-1])
        self.assertEqual(agent.repair_calls, 2)

    async def test_exhaustion_and_infrastructure_never_create_pending_policy(self):
        broken = Program(name="Broken", description="Broken Python.", implementation=SOURCE + "}")
        agent = AlphaEvolve(
            "task", ScriptedProvider([broken, broken]), config=Config(max_repairs=1)
        )
        self.assertEqual(await agent.generate(), [])
        self.assertEqual(agent.repair_calls, 1)
        self.assertEqual(agent._pending, {})
        self.assertEqual(agent.attempts[-1]["status"], "discarded")
        self.assertIn("exhausted after 1", agent.attempts[-1]["error"])
        for error in [
            ProviderError("offline"),
            ValidationError.from_exception_data(
                "Provider response", [{"type": "missing", "loc": ("response",), "input": {}}]
            ),
            asyncio.CancelledError(),
        ]:
            provider = ScriptedProvider([broken, error])
            # ScriptedProvider raises Exception; cancellation is injected at the call boundary.
            if isinstance(error, asyncio.CancelledError):
                original = provider.acall

                async def cancel(context, **kwargs):
                    if provider.calls:
                        raise asyncio.CancelledError()
                    return await original(context, **kwargs)

                provider.acall = cancel
            agent = AlphaEvolve("task", provider)
            with self.assertRaises(type(error)):
                await agent.generate()
            self.assertEqual(agent._pending, {})
            self.assertEqual(agent.repair_calls, 1)

    async def test_runtime_repair_reuses_successful_scores_and_original_budget(self):
        provider = ScriptedProvider([program(0), program(9), program(1)])
        agent = AlphaEvolve("task", provider)
        sandbox = FakeSandbox()
        checked = []

        async def evaluate(implementation, environment, seed, call_timeout):
            checked.append(implementation)
            if "return 9" in implementation:
                raise PolicyError("Action outside action_space")
            return trajectory(7.0, {})

        sandbox.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=3) as env,
            recorded_run(
                name="repair",
                path=Path(directory) / "run",
                environment=env,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as (run, rollouts),
        ):
            output = io.StringIO()
            await run_search(
                agent,
                run,
                rollouts,
                generations=1,
                batch_size=2,
                console=Console(file=output, force_terminal=False, width=140),
            )
            self.assertEqual(checked.count(SOURCE), 1)
            self.assertEqual(len(checked), 3)
            self.assertEqual(len(run.policies()), 3)
            self.assertEqual(len(list((run.path / "exports").glob("*.py"))), 3)
            self.assertEqual(agent.completed, 2)
            self.assertEqual(agent.repair_calls, 1)
            self.assertIn("Action outside action_space", provider.calls[-1])
            self.assertIn("Repairing", output.getvalue())
            self.assertEqual(agent._pending, {})

    async def test_syntax_and_runtime_share_budget_and_infrastructure_takes_priority(self):
        broken = Program(name="Broken", description="Broken Python.", implementation=SOURCE + "}")
        provider = ScriptedProvider([broken, program(9)])
        agent = AlphaEvolve("task", provider, config=Config(max_repairs=1))
        policy = (await agent.generate())[0]
        self.assertIsNone(await agent.repair(policy, "Action outside action_space"))
        self.assertEqual(agent._pending, {})
        self.assertIn("exhausted after 1", agent.attempts[-1]["error"])
        self.assertEqual(len(provider.calls), 2)
        sandbox = FakeSandbox()

        async def evaluate(implementation, environment, seed, call_timeout):
            if seed == 0:
                raise PolicyError("bad policy")
            await asyncio.sleep(0)
            raise InfrastructureError("Docker stopped")

        sandbox.evaluate.side_effect = evaluate
        with gym.make("CartPole-v1") as env:
            with self.assertRaisesRegex(InfrastructureError, "Docker stopped"):
                async for _ in Executor(sandbox=sandbox, concurrency=2).evaluate(
                    [(policy.id, policy._implementation, seed) for seed in (0, 1)], env
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
