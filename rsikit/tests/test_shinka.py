"""ShinkaEvolve through RSIKit's generate/evaluate/update lifecycle."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, nodes
from slick import prompts

from rsikit import Candidate, Evaluation, EvaluationError, ShinkaEvolve, ShinkaProposer
from rsikit.edits import InvalidCandidate, apply_diff
from rsikit.tests.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "fixed\n# EVOLVE-BLOCK-START\nseed\n# EVOLVE-BLOCK-END\n"


def measured(score):
    return Evaluation(valid=True, metrics={"score": score})


class ShinkaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(prompts, "TEMPLATE_ROOT", ROOT)
        root.start()
        self.addCleanup(root.stop)

    async def test_contract_minimization_and_model_rewards(self):
        async def propose(parent, history):
            return f"source {len(history)}"

        strategy = ShinkaEvolve("seed", measured(5), propose, maximize=False, models=2)
        for score in (8, 2):
            batch = await strategy.generate()
            with self.assertRaises(RuntimeError):
                await strategy.generate()
            with self.assertRaises(EvaluationError):
                await strategy.update(batch, [Evaluation(valid=True)])
            self.assertEqual(strategy.pending, batch[0])
            await strategy.update(batch, [measured(score)])
        self.assertEqual(strategy.best.evaluation.metrics["score"], 2)
        self.assertEqual({row["model"] for row in strategy.selections}, {0, 1})
        self.assertEqual(sorted(float(g[0]) for g in strategy.model_gains), [0, 3])
        self.assertEqual(sum(strategy.offspring.values()), 2)

    async def test_novelty_retries_precede_evaluation_and_embed_only_mutable_code(self):
        embedded = []

        async def embed(source):
            embedded.append(source)
            return [1.0, 0.0]

        provider = ScriptedProvider(
            [
                SOURCE.replace("seed", "renamed"),
                '{"novel": false, "reason": "Only a rename"}',
                SOURCE.replace("seed", "algorithm"),
                '{"novel": true, "reason": "New algorithm"}',
            ]
        )
        operations = ShinkaProposer("Improve source", provider)

        async def propose(parent, history):
            return await operations(parent, history, context=strategy.context)

        strategy = ShinkaEvolve(
            SOURCE,
            measured(1),
            propose,
            embed=embed,
            novelty=operations.assess_novelty,
            patch_types=(("full", 1),),
        )
        batch = await strategy.generate()
        self.assertEqual(len(strategy.history), 1)
        self.assertIn("algorithm", batch[0].source)
        self.assertEqual(embedded, ["renamed\n", "seed\n", "algorithm\n"])
        self.assertIn("Only a rename", provider.calls[2])
        await strategy.update(batch, [measured(2)])
        self.assertEqual(len(strategy.history), 2)
        self.assertEqual(sum(strategy.offspring.values()), 1)

    async def test_rejected_generation_and_infrastructure_failure_have_distinct_accounting(self):
        async def duplicate(parent, history):
            return parent.source

        strategy = ShinkaEvolve("seed", measured(1), duplicate, max_proposals=2)
        self.assertEqual(await strategy.generate(), [])
        self.assertEqual(len(strategy.proposals), 2)
        self.assertFalse(strategy.history[-1].evaluation.valid)
        self.assertEqual(strategy.model_gains, [[0]])
        self.assertEqual(strategy.offspring, {})
        await strategy.update([], [])

        async def offline(parent, history):
            raise RuntimeError("offline")

        strategy = ShinkaEvolve("seed", measured(1), offline)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await strategy.generate()
        self.assertEqual(len(strategy.history), 1)
        self.assertEqual(strategy.model_gains, [[]])
        self.assertIsNone(strategy.pending)

    async def test_proposer_diff_full_cross_ensemble_and_scope(self):
        parent = Candidate(id=0, source=SOURCE, evaluation=measured(1))
        donor = Candidate(id=1, source=SOURCE.replace("seed", "donor"), evaluation=measured(2))
        changed = SOURCE.replace("seed", "changed")
        first = ScriptedProvider(["<<<<<<< SEARCH\nseed\n=====\nchanged\n>>>>>>> REPLACE"])
        second = ScriptedProvider([changed, changed])
        operations = ShinkaProposer("task", first, ensemble=(first, second))
        for mode, model in (("diff", 0), ("full", 1), ("cross", 1)):
            result = await operations(
                parent,
                (parent, donor),
                context={
                    "operation": "shinkaevolve",
                    "patch": mode,
                    "model": model,
                    "parents": (parent, donor),
                    "inspirations": (),
                    "guidance": ("use feedback",),
                },
            )
            self.assertEqual(result, changed)
        self.assertEqual((len(first.calls), len(second.calls)), (1, 2))
        self.assertIn("donor", second.calls[-1])
        self.assertIn("use feedback", second.calls[-1])
        self.assertEqual(operations.records[-1]["source"], changed)

    async def test_periodic_meta_guidance_and_invalid_json_preserve_scratchpad(self):
        provider = ScriptedProvider(['{"recommendations": ["Reuse successful ideas"]}', "not JSON"])
        operations = ShinkaProposer("task", provider)
        guidance = []

        async def propose(parent, history):
            guidance.append(strategy.context["guidance"])
            return f"new {len(history)}"

        strategy = ShinkaEvolve(
            "seed",
            measured(5),
            propose,
            maximize=False,
            reflect=operations.summarize,
            meta_interval=1,
        )
        for score in (1, 2, 3):
            batch = await strategy.generate()
            await strategy.update(batch, [measured(score)])
        self.assertEqual(guidance, [(), ("Reuse successful ideas",), ("Reuse successful ideas",)])
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("minimize score", provider.calls[0])
        self.assertIn('"source": "seed"', provider.calls[0])
        self.assertEqual(operations.records[-1]["calls"][0]["response"], "not JSON")

    async def test_archive_migration_and_extreme_scores(self):
        async def propose(parent, history):
            return f"new {len(history)}"

        strategy = ShinkaEvolve(
            "seed",
            measured(-1e308),
            propose,
            islands=1,
            archive_size=2,
            elite_ratio=0.5,
        )
        for score in (1e308, 1e308, 0):
            batch = await strategy.generate()
            await strategy.update(batch, [measured(score)])
        self.assertEqual(strategy.best.id, 1)
        self.assertEqual(len(strategy.islands[0]), 2)
        self.assertTrue(all(math.isfinite(w) for w in strategy.model_weights()))
        strategy.islands = [[strategy.history[1], strategy.history[3]], [strategy.history[0]]]
        strategy.migration_rate = 1
        strategy.migrate()
        self.assertNotIn(strategy.history[1], strategy.islands[1])
        self.assertIn(strategy.history[3], strategy.islands[1])

    def test_diff_boundaries_and_templates(self):
        for search in ("fixed", "seed\n# EVOLVE-BLOCK-END", "missing"):
            with self.subTest(search=search), self.assertRaises(InvalidCandidate):
                apply_diff(SOURCE, f"<<<<<<< SEARCH\n{search}\n=====\nchanged\n>>>>>>> REPLACE")
        with self.assertRaises(InvalidCandidate):
            apply_diff("aaa", "<<<<<<< SEARCH\naa\n=====\nb\n>>>>>>> REPLACE")
        for template in (ROOT / "rsikit/prompts/shinka").glob("*.j2"):
            parsed = Environment().parse(template.read_text())
            self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()
