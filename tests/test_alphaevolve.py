"""Exercise the generate/evaluate/update boundary and AlphaEvolve selection."""

import asyncio
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from jinja2 import Environment, nodes
from pydantic import ValidationError
from slick import prompts, render
from slick.providers import ProviderError

from research import alphaevolve
from research.alphaevolve.improved import AlphaEvolve, Config, InvalidCandidate
from research.alphaevolve.original.agent import Guidance
from research.rewards import mean_rewards
from rsikit import Executor, Policy
from rsikit.generation.edits import Edit, Mutation, Program, apply_edits, check_rewrite
from tests.helpers import recorded_run
from tests.providers import ScriptedProvider
from tests.test_episode_storage import trajectory
from tests.test_run import FakeSandbox

ROOT = Path(alphaevolve.__file__).parent
SOURCE = """from rsikit import Policy
# EVOLVE-BLOCK-START
class Solution(Policy):
    async def act(self, observation):
        return 0
# EVOLVE-BLOCK-END
"""


def program(number):
    return Program(
        description="Test policy approach.",
        name=f"Policy {number}",
        implementation=SOURCE.replace("return 0", f"return {number}"),
    )


class AlphaEvolveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_independent_founders_and_seed_feedback_reach_prompts(self):
        provider = ScriptedProvider([program(i) for i in range(9)])
        agent = AlphaEvolve("task", provider, config=Config(mode="rewrite", reset_interval=0))
        policies = await agent.generate(n=8)
        scores = {p.id: float(i) for i, p in enumerate(policies)}
        details = {p.id: {0: 100.0 + i, 1: -100.0 + i} for i, p in enumerate(policies)}
        invalid = {**details, policies[-1].id: {0: math.nan}}
        with self.assertRaises(ValueError):
            agent.update_scores(scores, seed_scores=invalid)
        self.assertEqual(agent.completed, 0)
        self.assertTrue(all(island is None for island in agent.islands))
        agent.update_scores(scores, seed_scores=details)
        self.assertEqual(
            [p.policy.name for p in agent.islands], [f"Policy {i}" for i in range(4, 8)]
        )
        self.assertEqual(agent.best, policies[7])
        details[policies[4].id][0] = 999  # Retained evidence must be a snapshot.
        parent, inspiration = agent.islands[:2]
        self.assertEqual(parent.seed_scores, {0: 104.0, 1: -96.0})
        for operation, output in (("mutate", Mutation), ("rewrite", Program)):
            prompt = render(
                f"improved/prompts/{operation}.j2",
                instance=agent,
                schema=output.model_json_schema(),
                parent=parent,
                inspirations=[inspiration],
                guidance="",
                failures=[],
            )
            self.assertIn('"1": -96.0', prompt)
            self.assertIn('"1": -95.0', prompt)
        prompt = render(
            "improved/prompts/evolve_prompt.j2",
            instance=agent,
            schema=Guidance.model_json_schema(),
            parent=parent,
            ideas=[],
            failures=[],
        )
        self.assertIn('"1": -96.0', prompt)
        await agent.generate()
        self.assertIn("Per-seed rewards", provider.calls[-1])

    async def test_failed_and_duplicate_founders_leave_islands_open_for_new_programs(self):
        broken = program(0).model_copy(update={"implementation": SOURCE + "}"})
        renamed = program(0).model_copy(update={"name": "Same program, different name"})
        provider = ScriptedProvider([program(0), renamed, broken, program(1), program(2)])
        agent = AlphaEvolve(
            "task", provider, config=Config(islands=3, max_repairs=0, reset_interval=0)
        )
        initial = await agent.generate(n=3)
        agent.update_scores({p.id: 0 for p in initial})
        self.assertEqual(sum(island is not None for island in agent.islands), 1)
        agent.reset_islands()
        self.assertEqual(sum(island is not None for island in agent.islands), 1)
        island, parent, _ = agent.sample()
        self.assertIs(parent, agent.islands[island])
        remaining = await agent.generate(n=2)
        self.assertTrue(all("Initial proposal" in call for call in provider.calls))
        agent.update_scores({p.id: i + 1 for i, p in enumerate(remaining)})
        self.assertEqual(len({island.policy.id for island in agent.islands}), 3)
        agent.reset_islands()
        self.assertEqual(len(agent.events), 1)
        self.assertTrue(all(island is not None for island in agent.islands))

    async def test_two_generations_persist_only_at_evaluation_and_update_selection(self):
        provider = ScriptedProvider([program(i) for i in range(20)])
        agent = AlphaEvolve("Improve score", provider, config=Config(mode="rewrite"))
        sandbox = FakeSandbox()

        async def evaluate(implementation, environment, seed, call_timeout):
            score = float(implementation.split("return ")[1].split()[0])
            return trajectory(score, {"result.txt": str(score).encode()})

        sandbox.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as folder,
            gym.make("CartPole-v1", max_episode_steps=2) as environment,
            recorded_run(
                name="loop",
                path=Path(folder) / "run",
                environment=environment,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as (run, rollouts),
        ):
            for generation in range(2):
                policies = await agent.generate(n=10)
                self.assertEqual(len(policies), 10)
                self.assertTrue(all(issubclass(p, Policy) for p in policies))
                self.assertEqual(len(run.policies()), generation * 10)
                self.assertEqual(len(list(run.path.rglob("*.py"))), generation * 10)
                scores = await mean_rewards(rollouts, policies)
                self.assertEqual(len(scores), 10)
                self.assertEqual(len(run.policies()), (generation + 1) * 10)
                if generation == 0:
                    self.assertIsNone(agent.best)
                else:
                    self.assertEqual(agent.best.name, "Policy 9")
                agent.update_scores(scores)
                self.assertEqual(agent.best.id, policies[-1].id)
                self.assertEqual(len(list(run.path.rglob("*.py"))), (generation + 1) * 10)
                self.assertEqual(len(list(run.path.rglob("result.txt"))), (generation + 1) * 10)
        self.assertEqual((agent.generation_calls, agent.completed), (20, 20))
        self.assertEqual(sandbox.start.await_count, 2)
        self.assertGreater(len({row["parent"].policy.id for row in agent.attempts[10:]}), 1)
        with self.assertRaises(KeyError):
            agent.update_scores(scores)

    async def test_invalid_generation_is_recorded_without_retry_or_execution(self):
        provider = ScriptedProvider(
            [
                program(0),
                "not json",
                Mutation(
                    description="Test policy improvement.",
                    name="Invalid",
                    edits=[Edit(search="from rsikit", replacement="from other")],
                ),
                Program(
                    description="Test policy approach.",
                    name="Wrong interface",
                    implementation="class Other: pass",
                ),
                program(1),
            ]
        )
        agent = AlphaEvolve("task", provider, config=Config(islands=1, max_repairs=0))
        initial = await agent.generate()
        agent.update_scores({initial[0].id: 0})
        self.assertEqual(await agent.generate(), [])
        self.assertEqual(await agent.generate(), [])
        # Switch operation to test interface validation after a full rewrite.
        agent.config = Config(islands=1, mode="rewrite", max_repairs=0)
        self.assertEqual(await agent.generate(), [])
        child = (await agent.generate())[0]
        self.assertEqual(agent.best, initial[0])
        self.assertEqual(agent.completed, 1)
        self.assertEqual(agent.attempts[1]["raw"], "not json")
        self.assertIn("not json", provider.calls[-1])
        self.assertEqual(agent.generation_calls, 5)
        with self.assertRaises(ValueError):
            agent.update_scores({child.id: math.nan})
        with self.assertRaises(KeyError):
            agent.update_scores({child.id: 1, "unknown": 2})
        self.assertEqual(agent.completed, 1)
        agent.update_scores({child.id: 1})
        self.assertEqual(agent.best, child)
        self.assertEqual(await agent.generate(n=0), [])

    async def test_ensemble_guidance_and_ties(self):
        unused = ScriptedProvider([])
        selected = ScriptedProvider(
            [
                program(0),
                Guidance(instruction="Try a new representation"),
                program(1),
                program(2),
                '{"instruction": " "}',
                program(3),
            ]
        )
        agent = AlphaEvolve(
            "task",
            unused,
            ensemble=((unused, 0), (selected, 1)),
            config=Config(islands=1, mode="rewrite", meta_interval=2, reset_interval=2),
        )
        for score in [0, 2, 2, 3]:
            policy = (await agent.generate())[0]
            prior = agent.best
            agent.update_scores({policy.id: score})
            if policy.name == "Policy 2":
                self.assertEqual(agent.best, prior)  # Preserve incumbent on a tie.
        self.assertEqual(unused.calls, [])
        self.assertEqual((agent.generation_calls, agent.meta_calls), (4, 2))
        self.assertIn("blank", agent.attempts[3]["meta_error"])
        self.assertIn("Try a new representation", selected.calls[2])
        self.assertGreater(agent.prompt_ideas[1].reward, 0)
        self.assertEqual(len(agent.events), 0)
        self.assertEqual(agent.best.name, "Policy 3")

    async def test_generation_errors_cancel_and_failed_batch_never_becomes_pending(self):
        provider = ScriptedProvider([program(0), ProviderError("offline")])
        agent = AlphaEvolve("task", provider)
        with self.assertRaisesRegex(ProviderError, "offline"):
            await agent.generate(n=2)
        self.assertEqual(agent._pending, {})
        self.assertIsNone(agent.best)
        self.assertEqual(len(provider.calls), 2)

        started = asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        with patch.object(provider, "acall", side_effect=blocked):
            task = asyncio.create_task(agent.generate())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(agent.attempts[-1]["status"], "cancelled")
            agent.config = Config(generation_timeout=0.01)
            with self.assertRaises(TimeoutError):
                await agent.generate()
        self.assertEqual(agent.attempts[-1]["status"], "rejected")
        self.assertEqual(agent._pending, {})

    async def test_concurrent_generation_bounds_calls_and_keeps_repairs_with_proposal(self):
        broken = program(0).model_copy(update={"implementation": SOURCE + "}"})
        provider = ScriptedProvider([broken, program(1), program(2), program(0)])
        acall = provider.acall
        started = [asyncio.Event() for _ in range(4)]
        release = [asyncio.Event() for _ in range(4)]
        active = peak = calls = 0

        async def delayed(*args, **kwargs):
            nonlocal active, peak, calls
            index = calls
            calls += 1
            response = await acall(*args, **kwargs)
            active += 1
            peak = max(peak, active)
            started[index].set()
            try:
                await release[index].wait()
                return response
            finally:
                active -= 1

        provider.acall = delayed
        agent = AlphaEvolve("task", provider)
        with self.assertLogs("research.alphaevolve", level="INFO") as logs:
            async with asyncio.timeout(2):
                task = asyncio.create_task(agent.generate(n=3, concurrency=2))
                try:
                    await started[1].wait()
                    self.assertFalse(started[2].is_set())
                    release[1].set()
                    await started[2].wait()
                    self.assertTrue(any("Policy 1" in line for line in logs.output))
                    self.assertEqual(agent._pending, {})
                    release[0].set()
                    await started[3].wait()  # First proposal repairs while third generates.
                    release[3].set()
                    release[2].set()
                    policies = await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertEqual([p.name for p in policies], ["Policy 0", "Policy 1", "Policy 2"])
        self.assertEqual((peak, active, agent.generation_calls, agent.repair_calls), (2, 0, 3, 1))
        self.assertEqual([len(row.get("repairs", [])) for row in agent.attempts], [1, 0, 0])
        self.assertEqual(len(agent._pending), 3)

    async def test_concurrent_failure_and_cancellation_reap_siblings(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                provider = ScriptedProvider([])
                started, fail = asyncio.Event(), asyncio.Event()
                calls = active = 0

                async def blocked(*args, **kwargs):
                    nonlocal active, calls
                    calls += 1
                    index = calls
                    active += 1
                    if calls == 2:
                        started.set()
                    try:
                        if index == 1:
                            await fail.wait()
                            raise ProviderError("offline")
                        await asyncio.Event().wait()
                    finally:
                        active -= 1

                provider.acall = blocked
                agent = AlphaEvolve("task", provider)
                async with asyncio.timeout(2):
                    task = asyncio.create_task(agent.generate(n=5, concurrency=2))
                    try:
                        await started.wait()
                        if cancel:
                            task.cancel()
                        else:
                            fail.set()
                        with self.assertRaises(asyncio.CancelledError if cancel else ProviderError):
                            await task
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(active, 0)
                self.assertEqual(agent._pending, {})
                self.assertTrue(all(row["status"] != "generating" for row in agent.attempts))
                finished_calls = calls
                await asyncio.sleep(0)
                self.assertEqual(calls, finished_calls)

    async def test_initial_generation_rejects_unbalanced_evolution_markers(self):
        invalid = Program(
            description="Test policy approach.",
            name="Unclosed block",
            implementation=SOURCE.replace("# EVOLVE-BLOCK-END", ""),
        )
        agent = AlphaEvolve(
            "task",
            ScriptedProvider(
                [
                    invalid,
                    program(0),
                    Mutation(
                        description="Test policy improvement.",
                        name="Improved",
                        edits=[Edit(search="return 0", replacement="return 1")],
                    ),
                ]
            ),
            config=Config(islands=1, max_repairs=0),
        )
        self.assertEqual(await agent.generate(), [])
        self.assertIn("Unclosed evolution block", agent.attempts[-1]["error"])
        self.assertIsNone(agent.best)
        self.assertEqual(agent._pending, {})
        self.assertEqual(agent.attempts[-1]["status"], "discarded")
        initial = (await agent.generate())[0]
        agent.update_scores({initial.id: 0})
        child = (await agent.generate())[0]
        agent.update_scores({child.id: 1})
        self.assertEqual(agent.best, child)

    async def test_typed_edit_boundaries_and_prompt_contracts(self):
        edits = [
            Edit(search="return 0", replacement="return 1"),
            Edit(search="return 1", replacement="return 2"),
        ]
        self.assertEqual(apply_edits(SOURCE, edits), program(2).implementation)
        for source, edit in (
            (SOURCE, Edit(search="absent", replacement="x")),
            (SOURCE, Edit(search="return 0", replacement="return 0")),
            ("aaa", Edit(search="aa", replacement="b")),
            (SOURCE, Edit(search="from rsikit", replacement="from elsewhere")),
        ):
            with self.assertRaises(InvalidCandidate):
                apply_edits(source, [edit])
        with self.assertRaises(InvalidCandidate):
            check_rewrite(SOURCE, program(1).implementation.replace("EVOLVE-BLOCK-END", "changed"))
        with self.assertRaises(ValidationError):
            Mutation.model_validate({"name": "Test", "edits": [{"search": "", "replacement": "x"}]})
        with self.assertRaises(ValidationError):
            Program(description="Test policy approach.", name=" ", implementation=SOURCE)
        agent = AlphaEvolve("task", ScriptedProvider([program(0)]))
        policy = (await agent.generate())[0]
        agent.update_scores({policy.id: 1})
        parent = agent.islands[0]
        for operation, output in (("mutate", Mutation), ("rewrite", Program)):
            text = render(
                f"improved/prompts/{operation}.j2",
                instance=agent,
                schema=output.model_json_schema(),
                parent=parent,
                inspirations=[],
                guidance="guidance",
                failures=[],
            )
            self.assertIn("guidance", text)
            self.assertIn('"properties"', text)
            self.assertIn("Solution", text)
        self.assertIn(
            '"properties"',
            render(
                "original/prompts/initialize.j2",
                instance=agent,
                schema=Program.model_json_schema(),
                proposal=1,
            ),
        )
        self.assertIn(
            '"properties"',
            render(
                "improved/prompts/evolve_prompt.j2",
                instance=agent,
                schema=Guidance.model_json_schema(),
                parent=parent,
                ideas=[],
                failures=[],
            ),
        )
        for path in ROOT.rglob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )


if __name__ == "__main__":
    unittest.main()
