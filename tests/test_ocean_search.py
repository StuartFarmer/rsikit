"""Budget, independent prompts, and frozen held-out selection without remote calls."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from slick import prompts

from research.elitesearch import Config
from rsikit import PolicyDefinition, Run
from tests.helpers import episodes
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program


class OceanSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_retries_preserve_usage_limits_and_error_boundaries(self):
        import httpx
        from openai import APIConnectionError, APIStatusError, APITimeoutError
        from slick.providers import OpenRouterAPI, ProviderError

        from research.providers import BudgetExceeded, BudgetProvider, UsageOpenRouter

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        connection = APIConnectionError(request=request)
        timeout = APITimeoutError(request=request)
        unauthorized = APIStatusError(
            "unauthorized", response=httpx.Response(401, request=request), body=None
        )
        usage = dict(prompt_tokens=4, completion_tokens=3, total_tokens=7, cost=0.00002)
        response = SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: usage),
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=[]))],
        )
        for responses, cap, error, calls in (
            ([connection, timeout, response], 10, None, 3),
            ([connection] * 5, 10, ProviderError, 4),
            ([connection, response], 1, BudgetExceeded, 1),
            ([unauthorized, response], 10, ProviderError, 1),
            ([ValueError("invalid request"), response], 10, ProviderError, 1),
            ([asyncio.CancelledError(), response], 10, asyncio.CancelledError, 1),
        ):
            with self.subTest(error=error, calls=calls):
                events = []
                provider = BudgetProvider(
                    UsageOpenRouter(model="scripted", max_retries=0),
                    max_calls=cap,
                    max_input_tokens=1100,
                    max_output_tokens=100,
                    input_price=1,
                    output_price=1,
                    log=events.append,
                )
                with (
                    patch.object(OpenRouterAPI, "_asend", new=AsyncMock(side_effect=responses)),
                    patch("asyncio.sleep", new=AsyncMock()),
                ):
                    if error:
                        with self.assertRaises(error):
                            await provider.acall("hello")
                    else:
                        self.assertEqual(await provider.acall("hello"), ("hello", []))
                self.assertEqual(provider.calls, calls)
                self.assertEqual(provider.reserved_tokens, calls * 1200)
                self.assertAlmostEqual(provider.reserved_cost, calls * 0.0012)
                self.assertEqual(len(events), calls * 2)
                self.assertTrue(all(e["status"] == "failed" for e in provider.events[:-1]))
                self.assertEqual(provider.events[-1]["status"], "failed" if error else "ok")
                if error is None:
                    self.assertEqual(provider.events[-1]["usage"], usage)

    async def test_retry_backoff_respects_call_deadline(self):
        import httpx
        from openai import APIConnectionError
        from slick.providers import OpenRouterAPI

        from research.providers import BudgetProvider, UsageOpenRouter

        provider = BudgetProvider(
            UsageOpenRouter(model="scripted", max_retries=0),
            max_input_tokens=1100,
            max_output_tokens=100,
        )
        failure = APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai"))
        with (
            patch.object(OpenRouterAPI, "_asend", new=AsyncMock(side_effect=failure)),
            self.assertRaises(asyncio.TimeoutError),
        ):
            await asyncio.wait_for(provider.acall("hello"), timeout=0.01)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.events[0]["status"], "failed")

    async def test_resume_reserves_calls_without_finish_records(self):
        from research.providers import BudgetExceeded, BudgetProvider

        provider = BudgetProvider(
            ScriptedProvider([]), max_calls=2, max_input_tokens=1100, max_output_tokens=100
        )
        provider.restore(
            [
                dict(call=1, event="llm_start", status="running"),
                dict(call=2, event="llm_start", status="running"),
                dict(call=1, event="llm_finish", status="ok"),
            ]
        )
        with self.assertRaises(BudgetExceeded):
            await provider.acall("hello")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.reserved_tokens, 2400)
        self.assertEqual([e["status"] for e in provider.events], ["ok", "running"])

    async def test_optional_budget_keeps_accounting_and_explicit_caps(self):
        from research.providers import BudgetExceeded, BudgetProvider

        for limits in ({}, {"max_calls": 1}, {"max_tokens": 1200}):
            with self.subTest(limits=limits):
                provider = BudgetProvider(
                    ScriptedProvider(["first", "second"]),
                    max_input_tokens=1100,
                    max_output_tokens=100,
                    **limits,
                )
                await provider.acall("hello")
                if limits:
                    with self.assertRaises(BudgetExceeded):
                        await provider.acall("hello")
                else:
                    await provider.acall("hello")
                    self.assertEqual(provider.calls, 2)
                    self.assertIsNone(provider.stopped)
                self.assertEqual(len(provider.events), provider.calls)
                self.assertIsNone(provider.reserved_cost)
                self.assertIsNone(provider.events[0]["reserved_cost"])
                json.dumps(provider.events, allow_nan=False)

    async def test_cli_population_and_seed_panel_reach_saved_run(self):
        from examples.ocean_search import main

        raw = ScriptedProvider(
            [
                program(0),
                program(1),
                *[
                    json.dumps(
                        dict(
                            name=f"Edit {i}",
                            description="Edit parent",
                            edits=[dict(search="return 0", replacement=f"return {i}")],
                        )
                    )
                    for i in (2, 3)
                ],
            ]
        )
        panels = []

        class Panels:
            events = []

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def evaluate(self, policies, seeds):
                panels.append(tuple(seeds))
                return {p.id: episodes({s: 1.0 for s in seeds}) for p in policies}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
            patch("examples.ocean_search.UsageOpenRouter", return_value=raw),
            patch("research.ocean.evaluator.PanelEvaluator", Panels),
        ):
            output = Path(directory) / "run"
            await main(
                [
                    "--output",
                    str(output),
                    "--model",
                    "scripted",
                    "--arm",
                    "elite",
                    "--spend-cap",
                    "100",
                    "--input-price",
                    "1",
                    "--output-price",
                    "1",
                    "--env",
                    "g2048",
                    "--population",
                    "2",
                    "--generations",
                    "2",
                    "--elites",
                    "1",
                    "--max-repairs",
                    "0",
                    "--seeds",
                    "7",
                    "9",
                    "--generation-concurrency",
                    "1",
                ]
            )
            manifest = json.loads((output / "manifest.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["proposals"], 4)
            self.assertEqual(summary["reason"], "completed")
            self.assertEqual(manifest["search_seeds"], [7, 9])
            self.assertEqual(manifest["max_calls"], 4)
            self.assertEqual(manifest["max_tokens"], 4 * (65536 + 16384))
            self.assertEqual(panels[:2], [(7, 9)] * 2)
            self.assertEqual(manifest["optimization_schedule"], "round-v1")
            self.assertEqual(summary["validation"][summary["winner"]]["scores"]["1000"], 1)
            self.assertIn("2 upstream Ocean episode seeds", raw.calls[0])

    async def test_installed_provider_records_wire_usage_and_honors_output_limit(self):
        from slick.providers import OpenRouterAPI

        from examples.ocean_search import BudgetProvider, UsageOpenRouter

        usage = dict(prompt_tokens=4, completion_tokens=3, total_tokens=7, cost=0.00002)
        response = SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: usage),
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=[]))],
        )
        raw = UsageOpenRouter(model="scripted", max_output_tokens=100, max_retries=0)
        provider = BudgetProvider(
            raw,
            max_calls=1,
            max_tokens=1200,
            spend_cap=1,
            max_input_tokens=1100,
            max_output_tokens=100,
            input_price=1,
            output_price=1,
        )
        with patch.object(OpenRouterAPI, "_asend", new=AsyncMock(return_value=response)) as send:
            self.assertEqual(await provider.acall("hello"), ("hello", []))
        self.assertEqual(provider.events[0]["usage"], usage)
        self.assertEqual(provider.events[0]["actual_cost"], 0.00002)
        self.assertEqual(send.call_args.args[0]["max_tokens"], 100)

    async def test_call_token_and_spend_reservations_include_failed_requests(self):
        from examples.ocean_search import BudgetExceeded, BudgetProvider

        for limits in (
            dict(max_calls=1, max_tokens=10000, spend_cap=10),
            dict(max_calls=10, max_tokens=1200, spend_cap=10),
            dict(max_calls=10, max_tokens=10000, spend_cap=0.0012),
        ):
            with self.subTest(limits=limits):
                raw = ScriptedProvider([RuntimeError("offline"), "should never be sent"])
                provider = BudgetProvider(
                    raw,
                    max_input_tokens=1100,
                    max_output_tokens=100,
                    input_price=1,
                    output_price=1,
                    **limits,
                )
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    await provider.acall("hello")
                with self.assertRaises(BudgetExceeded):
                    await provider.acall("hello")
                self.assertEqual(provider.calls, 1)
                self.assertEqual(provider.reserved_tokens, 1200)
                self.assertAlmostEqual(provider.reserved_cost, 0.0012)
                self.assertEqual(provider.events[0]["status"], "failed")
                self.assertIsNone(provider.events[0]["actual_cost"])
                self.assertGreaterEqual(provider.events[0]["finish"], provider.events[0]["start"])

    async def test_concurrent_reservations_and_oversized_input_never_exceed_cap(self):
        from examples.ocean_search import BudgetExceeded, BudgetProvider

        provider = BudgetProvider(
            ScriptedProvider(["ok"]),
            max_calls=1,
            max_tokens=1200,
            spend_cap=1,
            max_input_tokens=1100,
            max_output_tokens=100,
            input_price=1,
            output_price=1,
        )
        with self.assertRaises(BudgetExceeded):
            await provider.acall("x" * 1101)
        self.assertEqual(provider.calls, 0)
        provider.stopped = None
        results = await asyncio.gather(
            provider.acall("a"), provider.acall("b"), return_exceptions=True
        )
        self.assertEqual(sum(isinstance(r, BudgetExceeded) for r in results), 1)
        self.assertEqual(provider.calls, 1)

    async def test_independent_search_keeps_valid_policy_at_budget_and_hides_elites(self):
        from examples.ocean_search import BudgetProvider, Search, run_search

        raw = ScriptedProvider([program(0), "bad JSON", program(1)])
        provider = BudgetProvider(
            raw,
            max_calls=3,
            max_tokens=1000000,
            spend_cap=100,
            max_input_tokens=65536,
            max_output_tokens=1000,
            input_price=1,
            output_price=1,
        )

        class Panels:
            events = []

            async def evaluate(self, policies, seeds):
                return {p.id: episodes({s: int(p.name[-1]) for s in seeds}) for p in policies}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/elitesearch/prompts").resolve()),
        ):
            with Run.create(name="test", path=Path(directory) / "run") as run:
                agent = Search(
                    "2048",
                    provider,
                    None,
                    independent=True,
                    config=Config(
                        population_size=1,
                        generations=4,
                        elite_size=5,
                        new_fraction=1,
                        remix_fraction=0,
                        generation_concurrency=1,
                    ),
                )
                summary = await run_search(agent, provider, Panels(), run)
                self.assertEqual(summary["reason"], "budget_exhausted")
                self.assertEqual(summary["winner"], agent.elites[0].policy_id)
                self.assertEqual(summary["calls"], 3)
                self.assertEqual(summary["repairs"], 1)
                self.assertTrue((run.path / "arrival_trace.json").exists())
                self.assertNotIn("Policy 0", raw.calls[1])
                self.assertEqual(
                    PolicyDefinition.from_file(run.path / "winner.py").name, "Policy 1"
                )

    async def test_validation_selects_once_before_test_and_failure_uses_fallback(self):
        from examples.ocean_search import select_winner

        policies = [
            PolicyDefinition.from_text(json.loads(program(i))["implementation"], name=f"Policy {i}")
            for i in range(3)
        ]
        panels = []

        async def evaluate(candidates, seeds):
            seeds = tuple(seeds)
            panels.append(([p.id for p in candidates], seeds))
            return {
                p.id: episodes({s: (5 if p == policies[1] else 1) for s in seeds})
                for p in candidates
            }

        with tempfile.TemporaryDirectory() as directory:
            with Run.create(name="test", path=Path(directory) / "run") as run:
                result = await select_winner(policies[:2], policies[2], evaluate, run)
                self.assertEqual(result["winner"], policies[1].id)
                self.assertEqual(panels[0][1], tuple(range(1000, 1128)))
                self.assertEqual(panels[1], ([policies[1].id], tuple(range(2000, 2512))))
                panels.clear()
                result = await select_winner([], policies[2], evaluate, run)
                self.assertTrue(result["fallback"])
                self.assertEqual(panels[-1][0], [policies[2].id])

    async def test_incomplete_validation_is_not_averaged_and_tail_includes_repairs(self):
        from examples.ocean_search import active_seconds, generation_tails, select_winner

        candidate = PolicyDefinition.from_text(json.loads(program(0))["implementation"])
        fallback = PolicyDefinition.from_text(json.loads(program(1))["implementation"])

        async def evaluate(policies, seeds):
            return {p.id: episodes({next(iter(seeds)): 999}) for p in policies}

        with tempfile.TemporaryDirectory() as directory:
            with Run.create(name="partial", path=Path(directory) / "run") as run:
                result = await select_winner([candidate], fallback, evaluate, run)
                self.assertEqual(result["winner"], fallback.id)
                self.assertIsNone(result["test_mean"])
        events = [
            dict(generation=1, start=10, finish=14),
            dict(generation=1, start=12, finish=15),
            dict(generation=1, start=17, finish=18, repair=1),
        ]
        self.assertEqual(active_seconds(events), 6)
        tail = generation_tails(events, [dict(generation=1, persisted=19)], {1: 0.2})[0]
        self.assertEqual(tail["llm_window"], 8)
        self.assertEqual(tail["residual_tail"], 1)
        self.assertEqual(tail["tail_fraction"], 0.125)


if __name__ == "__main__":
    unittest.main()
