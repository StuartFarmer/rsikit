import asyncio
import json
import math
import unittest

from tests.providers import ScriptedProvider
from tests.search_helpers import measure, policy, proposal, templates

STRATEGY = """def select(population, state, rng):
    return {"parent_id": population[0]["id"], "operator": "refine", "inspiration_ids": []}
"""


class EvoXTests(unittest.IsolatedAsyncioTestCase):
    async def test_stagnation_changes_strategy_without_resetting_population(self):
        from research import evox
        from research.evox.runtime import run_python_strategy

        provider = ScriptedProvider(
            [
                json.dumps(dict(refine="tune", diverge="change approach")),
                proposal(1),
                proposal(1),
                json.dumps(dict(code=STRATEGY)),
                proposal(5),
            ]
        )
        events = []
        with templates(evox):
            agent = evox.EvoX(
                "task",
                provider,
                measure,
                run_strategy=run_python_strategy,
                config=evox.Config(iterations=3, window=2),
                on_event=lambda kind, data: events.append((kind, data)),
            )
            result = await agent.run()
        self.assertEqual(agent.best.id, policy(5).id)
        self.assertEqual(len(result.strategies), 2)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual([w.steps for w in result.windows], [2, 1])
        self.assertEqual(result.candidates[-1].strategy_id, 1)
        self.assertEqual(result.evaluation_calls, 3)
        self.assertTrue(any(kind == "window" for kind, _ in events))

    async def test_invalid_candidates_consume_attempts_and_errors_propagate(self):
        from research import evox
        from research.evox.runtime import run_python_strategy

        provider = ScriptedProvider(
            [json.dumps(dict(refine="tune", diverge="explore")), "bad json", proposal(2)]
        )
        with templates(evox):
            agent = evox.EvoX(
                "task",
                provider,
                measure,
                run_strategy=run_python_strategy,
                config=evox.Config(iterations=2, window=2),
            )
            result = await agent.run()
        self.assertEqual(result.steps, 2)
        self.assertTrue(result.candidates[0].error)
        self.assertEqual(result.evaluation_calls, 1)
        self.assertEqual(agent.best.id, policy(2).id)

    async def test_worker_validates_output_and_enforces_timeout(self):
        from research.evox.runtime import run_python_strategy

        population = [dict(id=0, quality=1)]
        self.assertEqual((await run_python_strategy(STRATEGY, population, {}, 0))["parent_id"], 0)
        for code in (
            "def select(population, state, rng):\n population.clear()\n return {}",
            "def select(population, state, rng):\n return {'x': 'x' * 1100000}",
        ):
            with self.assertRaises(ValueError):
                await run_python_strategy(code, population, {}, 0)
        with self.assertRaisesRegex(ValueError, "timed out"):
            await run_python_strategy("while True: pass", population, {}, 0, timeout=0.1)
        task = asyncio.create_task(run_python_strategy("while True: pass", population, {}, 0))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_bad_selections_and_failed_drafts_retain_previous_strategy(self):
        from research import evox
        from research.evox.runtime import run_python_strategy

        with templates(evox):
            provider = ScriptedProvider(
                [
                    json.dumps(dict(refine="tune", diverge="explore")),
                    proposal(1),
                    *[json.dumps(dict(code="not python")) for _ in range(3)],
                    proposal(2),
                ]
            )
            agent = evox.EvoX(
                "task",
                provider,
                measure,
                run_strategy=run_python_strategy,
                config=evox.Config(iterations=2, window=1),
            )
            result = await agent.run()
        self.assertEqual(len(result.strategies), 1)
        self.assertEqual(len(result.strategy_attempts), 3)
        for parent, inspirations in ((True, []), (99, []), (0, [0]), (0, [1, 1])):
            with self.assertRaises(ValueError):
                agent.check_selection(
                    dict(parent_id=parent, operator="refine", inspiration_ids=inspirations),
                    [dict(id=0), dict(id=1)],
                )

    async def test_provider_value_error_propagates_and_selector_fallback_is_recorded(self):
        from research import evox
        from research.evox.agent import Strategy
        from research.evox.runtime import run_python_strategy

        with templates(evox):
            agent = evox.EvoX(
                "task",
                ScriptedProvider(
                    [
                        json.dumps(dict(refine="tune", diverge="explore")),
                        proposal(1),
                        ValueError("provider failed"),
                    ]
                ),
                measure,
                run_strategy=run_python_strategy,
                config=evox.Config(iterations=2, window=1),
            )
            with self.assertRaisesRegex(ValueError, "provider failed"):
                await agent.run()
        events = []
        agent.on_event = lambda kind, data: events.append((kind, data))
        agent.active = Strategy(1, "def select(population, state, rng): return {}")
        selection = await agent.select_context(agent.population())
        self.assertEqual(selection.parent_id, 0)
        self.assertEqual(events[-1][0], "selection_fallback")

    def test_nonpositive_window_rejected_before_generation(self):
        from research.evox import Config, EvoX
        from research.evox.runtime import run_python_strategy

        for window in (0, -1):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window"):
                EvoX(
                    "task",
                    ScriptedProvider([]),
                    measure,
                    run_strategy=run_python_strategy,
                    config=Config(window=window),
                )

    def test_window_reward_uses_actual_steps(self):
        from research.evox.agent import window_score

        self.assertAlmostEqual(window_score(2, 5, 4), 3 * (1 + math.log1p(2)) / 2)
        self.assertEqual(window_score(-2, -1, 1), 1)
