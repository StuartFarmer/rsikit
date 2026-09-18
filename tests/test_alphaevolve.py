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
from slick import prompts
from slick.providers import ProviderError

import rsikit.alphaevolve as alphaevolve
from rsikit import Executor, Policy, Run
from rsikit.alphaevolve import AlphaEvolve, Config, InvalidCandidate
from rsikit.alphaevolve.agent import Guidance
from rsikit.alphaevolve.edits import Edit, Mutation, Program, apply_edits, check_rewrite
from tests.providers import ScriptedProvider
from tests.test_run import FakeSandbox

ROOT = Path(alphaevolve.__file__).parent / "prompts"
SOURCE = """from rsikit import Policy
# EVOLVE-BLOCK-START
class Solution(Policy):
    async def act(self, observation):
        return 0
# EVOLVE-BLOCK-END
"""


def program(number):
    return Program(
        name=f"Policy {number}", implementation=SOURCE.replace("return 0", f"return {number}")
    )


class AlphaEvolveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_two_generations_persist_only_at_evaluation_and_update_selection(self):
        provider = ScriptedProvider(
            [
                *(program(i) for i in range(10)),
                *(
                    Mutation(
                        name=f"Child {i}",
                        edits=[Edit(search="return 9", replacement=f"return {i}")],
                    )
                    for i in range(10, 20)
                ),
            ]
        )
        agent = AlphaEvolve("Improve score", provider)
        sandbox = FakeSandbox()

        async def evaluate(implementation, environment, seed, call_timeout):
            score = float(implementation.split("return ")[1].split()[0])
            return score, {"result.txt": str(score).encode()}

        sandbox.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as folder,
            gym.make("CartPole-v1", max_episode_steps=2) as environment,
            Run.create(
                name="loop",
                path=Path(folder) / "run",
                environment=environment,
                executor=Executor(sandbox=sandbox, concurrency=2),
            ) as run,
        ):
            for generation in range(2):
                policies = await agent.generate(n=10)
                self.assertEqual(len(policies), 10)
                self.assertTrue(all(issubclass(p, Policy) for p in policies))
                self.assertEqual(len(run.policies()), generation * 10)
                self.assertEqual(len(list(run.path.rglob("*.py"))), generation * 10)
                scores = await run.evaluate(policies)
                self.assertEqual(len(scores), 10)
                self.assertEqual(len(run.policies()), (generation + 1) * 10)
                if generation == 0:
                    self.assertIsNone(agent.best)
                else:
                    self.assertEqual(agent.best.name, "Policy 9")
                agent.update(scores)
                self.assertEqual(agent.best.id, policies[-1].id)
                self.assertEqual(len(list(run.path.rglob("*.py"))), (generation + 1) * 10)
                self.assertEqual(len(list(run.path.rglob("result.txt"))), (generation + 1) * 10)
        self.assertEqual((agent.generation_calls, agent.completed), (20, 20))
        self.assertEqual(sandbox.start.await_count, 2)
        self.assertTrue(all("Score: 9.0" in call for call in provider.calls[10:]))
        with self.assertRaises(KeyError):
            agent.update(scores)

    async def test_invalid_generation_is_recorded_without_retry_or_execution(self):
        provider = ScriptedProvider(
            [
                program(0),
                "not json",
                Mutation(
                    name="Invalid", edits=[Edit(search="from rsikit", replacement="from other")]
                ),
                Program(name="Wrong interface", implementation="class Other: pass"),
                program(1),
            ]
        )
        agent = AlphaEvolve("task", provider, config=Config(islands=1))
        initial = await agent.generate()
        agent.update({initial[0].id: 0})
        with self.assertRaises(ValidationError):
            await agent.generate()
        with self.assertRaises(InvalidCandidate):
            await agent.generate()
        # Switch operation to test interface validation after a full rewrite.
        agent.config = Config(islands=1, mode="rewrite")
        with self.assertRaises(InvalidCandidate):
            await agent.generate()
        child = (await agent.generate())[0]
        self.assertEqual(agent.best, initial[0])
        self.assertEqual(agent.completed, 1)
        self.assertEqual(agent.attempts[1]["raw"], "not json")
        self.assertIn("not json", provider.calls[-1])
        self.assertEqual(agent.generation_calls, 5)
        with self.assertRaises(ValueError):
            agent.update({child.id: math.nan})
        with self.assertRaises(KeyError):
            agent.update({child.id: 1, "unknown": 2})
        self.assertEqual(agent.completed, 1)
        agent.update({child.id: 1})
        self.assertEqual(agent.best, child)
        self.assertEqual(await agent.generate(n=0), [])

    async def test_ensemble_guidance_islands_and_ties(self):
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
            config=Config(mode="rewrite", meta_interval=2, reset_interval=2),
        )
        for score in [0, 2, 2, 3]:
            policy = (await agent.generate())[0]
            prior = agent.best
            agent.update({policy.id: score})
            if policy.name == "Policy 2":
                self.assertEqual(agent.best, prior)  # Preserve incumbent on a tie.
        self.assertEqual(unused.calls, [])
        self.assertEqual((agent.generation_calls, agent.meta_calls), (4, 2))
        self.assertIn("blank", agent.attempts[3]["meta_error"])
        self.assertIn("Try a new representation", selected.calls[2])
        self.assertGreater(agent.prompt_ideas[1].reward, 0)
        self.assertEqual(len(agent.events), 4)
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

    async def test_initial_generation_rejects_unbalanced_evolution_markers(self):
        invalid = Program(
            name="Unclosed block", implementation=SOURCE.replace("# EVOLVE-BLOCK-END", "")
        )
        agent = AlphaEvolve(
            "task",
            ScriptedProvider(
                [
                    invalid,
                    program(0),
                    Mutation(
                        name="Improved", edits=[Edit(search="return 0", replacement="return 1")]
                    ),
                ]
            ),
        )
        with self.assertRaisesRegex(InvalidCandidate, "Unclosed evolution block"):
            await agent.generate()
        self.assertIsNone(agent.best)
        self.assertEqual(agent._pending, {})
        self.assertEqual(agent.attempts[-1]["status"], "rejected")
        initial = (await agent.generate())[0]
        agent.update({initial.id: 0})
        child = (await agent.generate())[0]
        agent.update({child.id: 1})
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
            Program(name=" ", implementation=SOURCE)
        agent = AlphaEvolve("task", ScriptedProvider([program(0)]))
        policy = (await agent.generate())[0]
        agent.update({policy.id: 1})
        parent = agent.islands[0]
        for method in (AlphaEvolve.mutate, AlphaEvolve.rewrite):
            text = await method.render(agent, parent, [], "guidance", [])
            self.assertIn("guidance", text)
            self.assertIn('"properties"', text)
            self.assertIn("Solution", text)
        self.assertIn('"properties"', await AlphaEvolve.initialize.render(agent, 1))
        self.assertIn('"properties"', await AlphaEvolve.evolve_prompt.render(agent, parent, [], []))
        for path in ROOT.glob("*.j2"):
            self.assertEqual(
                list(Environment().parse(path.read_text()).find_all((nodes.If, nodes.CondExpr))), []
            )


if __name__ == "__main__":
    unittest.main()
