"""Prompt boundaries, strategy composition and evidence-scoped memory."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from slick import prompts

from rsikit import Candidate, EoH, Evaluation, EvaluationError, HillClimb, ProposalRejected
from rsikit.proposer import Draft, PromptProposer
from rsikit.tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[2]


def candidate(identifier, score, *, valid=True, feedback=""):
    return Candidate(
        id=identifier,
        source=f"def f(): return {identifier}",
        parent_id=0 if identifier else None,
        evaluation=Evaluation(valid=valid, metrics={"score": score}, feedback=feedback),
    )


class PromptComponentTests(unittest.IsolatedAsyncioTestCase):
    async def test_crossover_records_all_parents_and_keeps_primary_edit_boundary(self):
        class Cross(HillClimb):
            def select_parent(self):
                self.context = {"operation": "crossover", "parents": tuple(self.history[:2])}
                return self.history[0]

        initial = "fixed\n# EVOLVE-BLOCK-START\nold\n# EVOLVE-BLOCK-END\n"
        source = initial.replace("old", "combined")
        provider = ScriptedProvider(
            [
                Draft(description="combine", source=source),
                Draft(description="illegal", source=source.replace("fixed", "changed")),
            ]
        )
        operations = PromptProposer("Improve", provider)

        async def propose(parent, history):
            return await operations(parent, history, context=strategy.context)

        strategy = Cross(initial, Evaluation(valid=True, metrics={"score": 1}), propose)
        strategy.history.append(candidate(1, 0.5))
        batch = await strategy.generate()
        self.assertEqual(batch[0].source, source)
        self.assertEqual(batch[0].parent_ids, (0, 1))
        self.assertEqual(batch[0].parent_id, 0)
        await strategy.update(batch, [Evaluation(valid=True, metrics={"score": 2})])
        self.assertEqual(await strategy.generate(), [])
        self.assertEqual(strategy.history[-1].parent_ids, (0, 1))
        self.assertFalse(strategy.history[-1].evaluation.valid)
        self.assertEqual(operations.records[0]["operation"], "crossover")
        with self.assertRaisesRegex(ValueError, "distinct parents"):
            await operations(
                strategy.best, tuple(strategy.history), context={"operation": "crossover"}
            )
        self.assertEqual(len(provider.calls), 2)

    async def test_provider_schema_failure_does_not_become_a_candidate_rejection(self):
        with self.assertRaises(ValidationError) as failure:
            Draft.model_validate({})
        provider = ScriptedProvider([failure.exception])
        proposer = PromptProposer("Improve f", provider)
        parent = candidate(0, 0)
        strategy = HillClimb(parent.source, parent.evaluation, proposer)
        with self.assertRaises(ValidationError):
            await strategy.generate()
        self.assertEqual(len(strategy.history), 1)
        self.assertIsNone(strategy.pending)

    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_operations_preserve_context_and_literal_instructions(self):
        parent, other = candidate(0, 0), candidate(1, 1)
        draft = json.dumps({"description": "combine ideas", "source": "def f(): return 2\n"})
        operations = ("mutate", "INIT", "E1", "E2", "M1", "M2", "M3", "alphaevolve")
        provider = ScriptedProvider(
            [draft] * len(operations) + ["Compare the loops", draft, "fixed"]
        )
        proposer = PromptProposer("Improve f", provider, instructions={"E2": "{{ literal }}"})
        for operation in (*operations, "dgm-archive"):
            context = {"operation": operation, "parents": (parent, other), "inspirations": (other,)}
            source = await proposer(parent, (parent, other), context=context)
            self.assertEqual(source, "def f(): return 2\n")
            self.assertEqual(context["parents"], (parent, other))
            self.assertEqual(proposer.records[-1]["parent_ids"], [0, 1])
        self.assertIn("{{ literal }}", provider.calls[3])
        self.assertIn(other.source, provider.calls[3])
        self.assertIn("Compare the loops", provider.calls[-1])
        self.assertEqual(proposer.records[-1]["draft"]["description"], "combine ideas")
        self.assertEqual(
            await proposer.repair("broken", Evaluation(valid=False, feedback="syntax")), "fixed"
        )
        self.assertIn("syntax", provider.calls[-1])

    async def test_rejection_and_provider_failure_keep_distinct_attempt_records(self):
        provider = ScriptedProvider(
            [
                "not JSON",
                '{"description":"idea","source":"  "}',
                RuntimeError("transport"),
            ]
        )
        proposer = PromptProposer("Improve f", provider)
        parent = candidate(0, 0)
        for _ in range(2):
            with self.assertRaises(ProposalRejected):
                await proposer(parent, (parent,))
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await proposer(parent, (parent,))
        with (
            patch.object(provider, "acall", side_effect=asyncio.CancelledError()),
            self.assertRaises(asyncio.CancelledError),
        ):
            await proposer(parent, (parent,))
        self.assertEqual([r["attempt"] for r in proposer.records], [0, 1, 2, 3])
        self.assertEqual([r["candidate_id"] for r in proposer.records], [1] * 4)
        self.assertEqual(proposer.records[0]["calls"][0]["response"], "not JSON")
        self.assertTrue(all("error" in r for r in proposer.records))

    async def test_same_proposer_supports_hillclimb_and_frozen_eoh_cycle(self):
        for kind in (HillClimb, EoH):
            with self.subTest(kind=kind):
                provider = ScriptedProvider(
                    [
                        json.dumps({"description": "step", "source": f"def f(): return {i}"})
                        for i in range(1, 6)
                    ]
                )
                operations = PromptProposer("Improve f", provider)

                async def propose(parent, history):
                    return await operations(parent, history, context=strategy.context)

                options = {"population_size": 1} if kind is EoH else {}
                strategy = kind(
                    candidate(0, 0).source, candidate(0, 0).evaluation, propose, **options
                )
                for score in range(1, 6):
                    batch = await strategy.generate()
                    await strategy.update(batch, [Evaluation(valid=True, metrics={"score": score})])
                self.assertEqual(strategy.best.id, 5)
                parents = [r["parent_ids"] for r in operations.records]
                self.assertEqual(
                    parents,
                    [[0], [1], [2], [3], [4]]
                    if kind is HillClimb
                    else [[0, 0, 0], [0, 0, 0], [0], [0], [0]],
                )


class ReflectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_measured_order_memory_bound_and_next_prompt(self):
        from rsikit.reflection import ReflectionMemory

        provider = ScriptedProvider(
            ["prefer one", "prefer zero", "repair syntax", "", RuntimeError("offline")]
        )
        memory = ReflectionMemory("Improve f", provider, max_items=1)
        parent, child = candidate(0, 0), candidate(1, 1)
        await memory.observe(parent, child)
        self.assertEqual(memory.records[-1]["better_id"], 1)
        await memory.observe(parent, child, maximize=False)
        self.assertEqual(memory.records[-1]["better_id"], 0)
        self.assertEqual(memory.texts, ("prefer zero",))
        await memory.observe(parent, candidate(2, 0))  # tie: no critique
        self.assertEqual(len(provider.calls), 2)
        failed = candidate(3, 0, valid=False, feedback="invalid syntax")
        await memory.observe(parent, failed)
        self.assertEqual(memory.texts, ("repair syntax",))
        for exception in (ValueError, RuntimeError):
            with self.assertRaises(exception):
                await memory.observe(parent, child)
            self.assertEqual(memory.texts, ("repair syntax",))
        missing = child.model_copy(update={"evaluation": Evaluation(valid=True)})
        with self.assertRaises(EvaluationError):
            await memory.observe(parent, missing)
        next_provider = ScriptedProvider(['{"description":"repair","source":"new source"}'])
        proposer = PromptProposer("Improve f", next_provider)
        await proposer(parent, (parent,), context={"guidance": memory.texts})
        self.assertIn("repair syntax", next_provider.calls[0])
