"""Poker rules, duplicate scoring and population evaluation checks."""

import asyncio
import io
import json
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console
from slick import prompts

from research.elitesearch import Config, Measurement
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program

from .game import play_hand, run_block, schedule
from .search import PokerSearch
from .tournament import Tournament, TournamentConfig, docker_block


class PopulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tournament_batches_failures_and_only_replays_changed_tables(self):
        from rsikit.policy import Policy

        candidates = [
            Policy.from_text(json.loads(program(i))["implementation"], name=str(i))
            for i in range(20)
        ]
        calls = []
        broken = {candidates[0]._implementation, candidates[6]._implementation}

        async def runner(request, config):
            calls.append(request)
            await asyncio.sleep(0.01)
            for index, source in enumerate(request["sources"]):
                if source in broken:
                    return {"failure": index, "error": "broken candidate"}
            return run_block(6, config.deals, request["seed"], lambda seat, o: 1)

        tournament = Tournament(TournamentConfig(rounds=1, deals=1, workers=2), runner=runner)
        groups = [list(range(i, i + 6)) for i in (0, 6, 12)]
        with patch(f"{__package__}.tournament.schedule", return_value=groups):
            result = await tournament(candidates[:18])
            self.assertEqual(
                {pid for pid, m in result.items() if m.failure},
                {candidates[0].id, candidates[6].id},
            )
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(not m.scores for m in result.values()))
            broken.clear()
            repaired = candidates[:18]
            repaired[0] = candidates[18]
            repaired[6] = candidates[19]
            result = await tournament(repaired)
            self.assertEqual(len(calls), 5, "Only failed or changed tables should run again")
            self.assertEqual(tournament.report["reused_blocks"], 1)
            self.assertTrue(all(m.scores == {0: 0} for m in result.values()))
            tournament.seed += 1
            await tournament(repaired)
            self.assertEqual(len(calls), 8, "New deals must not reuse old results")
            repaired[12], repaired[13] = repaired[13], repaired[12]
            await tournament(repaired)
            self.assertEqual(len(calls), 9, "Changed seat order invalidates that table")
            tournament.config = TournamentConfig(rounds=1, deals=2, workers=2)
            await tournament(repaired)
            self.assertEqual(len(calls), 12, "Changed budgets invalidate cached results")

    async def test_search_repairs_tournament_failures_together_without_screening(self):
        class Evaluator:
            report = {}
            scored = []

            async def screen(self, policies):
                raise AssertionError("Redundant screening pass")

            async def __call__(self, policies):
                self.scored.append([p.name for p in policies])
                return {
                    p.id: Measurement({}, "broken")
                    if p.name in ("Policy 0", "Policy 1")
                    else Measurement({0: 1})
                    for p in policies
                }

        evaluator = Evaluator()
        agent = PokerSearch(
            "Score",
            ScriptedProvider([program(i) for i in range(5)]),
            evaluator,
            config=Config(population_size=3, generations=1),
        )
        with patch.object(
            prompts,
            "TEMPLATE_ROOT",
            Path(__file__).resolve().parents[3] / "research/elitesearch/prompts",
        ):
            await agent.run()
        self.assertEqual(len(evaluator.scored), 2)
        self.assertEqual(set(evaluator.scored[1]), {"Policy 2", "Policy 3", "Policy 4"})
        self.assertEqual([row.repairs for row in agent.organisms], [1, 1, 0])

    async def test_resume_can_override_container_memory(self):
        from .__main__ import main, parser

        with tempfile.TemporaryDirectory() as directory:
            saved = vars(
                parser().parse_args(
                    ["search", "--output", directory, "--memory-gb", "4", "--seed", "7"]
                )
            )
            saved["output"] = directory
            (Path(directory) / "experiment.json").write_text(json.dumps({"arguments": saved}))
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch(f"{__package__}.__main__.run_search", new_callable=AsyncMock) as search,
            ):
                await main(["search", "--output", directory, "--resume", "--memory-gb", "32"])
            self.assertEqual(search.call_args.args[1].memory_gb, 32)
            self.assertEqual(
                json.loads((Path(directory) / "status.json").read_text())["status"], "completed"
            )

    async def test_pool_cleanup_survives_cancellation_and_failed_removal_can_retry(self):
        from rsikit.evaluation import InfrastructureError

        from .pool import TablePool

        entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def remove(*args):
            entered.set()
            await release.wait()
            cleaned.set()

        pool = TablePool(TournamentConfig())
        pool.name = "test-evaluator"
        with patch(f"{__package__}.pool.remove_container", side_effect=remove):
            closing = asyncio.create_task(pool.close())
            await entered.wait()
            closing.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertTrue(cleaned.is_set())
            self.assertTrue(pool.closed)

        pool = TablePool(TournamentConfig())
        pool.name = "test-evaluator"
        with patch(
            f"{__package__}.pool.remove_container", side_effect=[OSError("daemon"), None]
        ) as remove:
            with self.assertRaises(InfrastructureError):
                await pool.close()
            self.assertFalse(pool.closed)
            await pool.close()
            self.assertTrue(pool.closed)
            self.assertEqual(remove.await_count, 2)

    async def test_failed_docker_removal_is_not_reported_as_success(self):
        from rsikit.evaluation import InfrastructureError

        from .pool import remove_container

        cleanup = type("Cleanup", (), {"returncode": 1, "wait": AsyncMock(return_value=1)})()
        with patch(f"{__package__}.pool._spawn", return_value=cleanup):
            with self.assertRaises(InfrastructureError):
                await remove_container("test-evaluator", None)

    async def test_infrastructure_failure_cancels_active_blocks(self):
        from rsikit.evaluation import InfrastructureError

        from .baselines import policies

        started, cancelled = [], []
        ready = asyncio.Event()

        async def runner(request, config):
            index = len(started)
            started.append(index)
            if len(started) == 4:
                ready.set()
            await ready.wait()
            if index == 0:
                raise InfrastructureError("worker disconnected")
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(index)

        tournament = Tournament(TournamentConfig(rounds=4, workers=4), runner=runner)
        with self.assertRaisesRegex(InfrastructureError, "worker disconnected"):
            await asyncio.wait_for(tournament(policies(12)), 1)
        self.assertEqual(sorted(cancelled), started[1:])
        self.assertGreaterEqual(len(cancelled), 3)

    async def test_search_cli_exports_and_completed_resume(self):
        from .__main__ import parser, run_search

        pools = []

        class FakeTournament:
            def __init__(self, *args, **kwargs):
                self.report = {}
                self.progress = kwargs.get("progress")
                pools.append(kwargs.get("runner"))

            async def __call__(self, policies):
                if self.progress:
                    self.progress(0, 1)
                    self.progress(1, 1)
                self.report = {
                    "failures": {},
                    "leaderboard": [{"policy_id": p.id, "bb_per_100": 0} for p in policies],
                }
                return {p.id: Measurement({0: 0}) for p in policies}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                f"{__package__}.__main__.LoggedOpenRouter",
                return_value=ScriptedProvider(
                    [program(0), program(1)]
                    + [
                        json.dumps(
                            dict(
                                name=f"Policy {i}",
                                description="Edited contender",
                                edits=[dict(search="return 0", replacement=f"return {i}")],
                            )
                        )
                        for i in (2, 3)
                    ]
                ),
            ),
            patch(f"{__package__}.__main__.Tournament", FakeTournament),
        ):
            args = parser().parse_args(
                [
                    "search",
                    "--output",
                    tmp,
                    "--population",
                    "2",
                    "--elites",
                    "1",
                    "--generations",
                    "2",
                    "--seed",
                    "7",
                ]
            )
            output = io.StringIO()
            console = Console(file=output, force_terminal=True, color_system=None, width=120)
            await run_search(args, TournamentConfig(), console=console)
            self.assertIsNotNone(pools[0])
            self.assertTrue(all(pool is pools[0] for pool in pools))
            self.assertTrue(pools[0].closed)
            for text in (
                "Generations",
                "Population evaluated/discarded",
                "Table blocks",
                "Current elites",
                "BB/100",
                "Generating policies",
                "Elite slots filled",
                "Current elites — generation 1",
                "Current elites — generation 2",
            ):
                self.assertIn(text, output.getvalue())
            saved_log = (Path(tmp) / "run.log").read_text()
            self.assertIn("Generated Policy 0", saved_log)
            self.assertIn("score=0.000 BB/100", saved_log)
            leaderboard = (Path(tmp) / "leaderboard.json").read_text()
            args.resume = True
            await run_search(args, TournamentConfig(), console=console)
            self.assertEqual((Path(tmp) / "leaderboard.json").read_text(), leaderboard)
            self.assertTrue((Path(tmp) / "best.py").exists())
            self.assertEqual(len((Path(tmp) / "curves.csv").read_text().splitlines()), 3)

    async def test_cancelled_launch_removes_named_container(self):
        cleanup = type("Cleanup", (), {"returncode": 0, "wait": AsyncMock(return_value=0)})()
        with patch(
            f"{__package__}.pool._spawn", side_effect=[asyncio.CancelledError(), cleanup]
        ) as spawn:
            with self.assertRaises(asyncio.CancelledError):
                await docker_block({}, TournamentConfig())
            self.assertEqual(spawn.await_count, 2)
            self.assertEqual(spawn.call_args.args[:3], ("docker", "rm", "-f"))

    async def test_daemon_cleanup_has_deadline(self):
        class SlowProcess:
            returncode = None
            stdin = type(
                "Input",
                (),
                {"write": lambda *a: None, "drain": AsyncMock(), "close": lambda *a: None},
            )()
            stderr = type("Error", (), {"read": AsyncMock(return_value=b"")})()

            @property
            def stdout(self):
                return self

            async def readline(self):
                await asyncio.sleep(0.1)
                return b""

            async def wait(self):
                if self.returncode is None:
                    await asyncio.sleep(0.1)
                return 0

            def kill(self):
                self.returncode = -9

        from rsikit.evaluation import InfrastructureError

        with (
            patch(f"{__package__}.pool._spawn", side_effect=[SlowProcess(), SlowProcess()]),
            patch(f"{__package__}.pool.CLEANUP_TIMEOUT", 0.01),
            patch(f"{__package__}.pool.STARTUP_TIMEOUT", 0.005),
        ):
            started = time.monotonic()
            with self.assertRaises(InfrastructureError):
                await docker_block({}, TournamentConfig(block_timeout=0.005))
            self.assertLess(time.monotonic() - started, 0.1)

    async def test_resume_keeps_consumed_repair_budget(self):
        async def evaluate(policies):
            return {
                p.id: Measurement({}, "illegal") if p.name == "Policy 0" else Measurement({0: 1})
                for p in policies
            }

        config = Config(population_size=2, generations=1, max_repairs=2, generation_concurrency=1)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                prompts,
                "TEMPLATE_ROOT",
                Path(__file__).resolve().parents[3] / "research/elitesearch/prompts",
            ),
        ):
            checkpoint = Path(tmp) / "checkpoint.json"
            agent = PokerSearch(
                "Score",
                ScriptedProvider([program(0), program(1), RuntimeError("offline")]),
                evaluate,
                config=config,
            )
            agent.on_checkpoint = lambda current: current.save(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await agent.run()
            provider = ScriptedProvider([program(2)])
            resumed = PokerSearch("Score", provider, evaluate, config=config)
            resumed.load(checkpoint)
            await resumed.run()
            self.assertEqual(resumed.organisms[0].repairs, 2)
            self.assertIn("Repair this organism", provider.calls[0])

    async def test_discarded_invalid_source_can_resume(self):
        broken = json.loads(program(0))
        broken["implementation"] = "class Solution(Policy): broken @"

        async def evaluate(policies):
            return {p.id: Measurement({0: 1}) for p in policies}

        config = Config(population_size=3, generations=1, max_repairs=0)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                prompts,
                "TEMPLATE_ROOT",
                Path(__file__).resolve().parents[3] / "research/elitesearch/prompts",
            ),
        ):
            agent = PokerSearch(
                "Score",
                ScriptedProvider([json.dumps(broken), program(1), program(2)]),
                evaluate,
                config=config,
            )
            await agent.run()
            checkpoint = Path(tmp) / "checkpoint.json"
            agent.save(checkpoint)
            resumed = PokerSearch("Score", ScriptedProvider([]), evaluate, config=config)
            resumed.load(checkpoint)
            await resumed.run()
            self.assertEqual(resumed.organisms[0].status, "discarded")

    async def test_incumbents_rescored_history_preserved_and_resume(self):
        calls = []

        async def evaluate(policies):
            calls.append([p.name for p in policies])
            return {
                p.id: Measurement(
                    {
                        0: (10 - int(p.name.split()[-1]))
                        if len(calls) == 1
                        else int(p.name.split()[-1])
                    }
                )
                for p in policies
            }

        config = Config(
            population_size=3, elite_size=1, generations=2, new_fraction=1, remix_fraction=0
        )
        with patch.object(
            prompts,
            "TEMPLATE_ROOT",
            Path(__file__).resolve().parents[3] / "research/elitesearch/prompts",
        ):
            agent = PokerSearch(
                "Score", ScriptedProvider([program(i) for i in range(6)]), evaluate, config=config
            )
            await agent.run()
        self.assertEqual(list(map(len, calls)), [3, 4])
        self.assertEqual(agent.elites[0].name, "Policy 5")
        self.assertEqual(agent.history[0]["scores"]["1"], 10)
        self.assertEqual(agent.history[1]["scores"]["1"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            agent.save(path)
            restored = PokerSearch("Score", ScriptedProvider([]), evaluate, config=config)
            restored.load(path)
            self.assertEqual(restored.history, agent.history)
            self.assertEqual(restored.elites[0].policy_id, agent.elites[0].policy_id)
            await restored.run()
            self.assertEqual(len(calls), 2)

    async def test_repair_requests_complete_field_scores(self):
        calls = []

        async def evaluate(policies):
            calls.append([p.name for p in policies])
            return {
                p.id: Measurement({}, "illegal") if p.name == "Policy 0" else Measurement({0: 1})
                for p in policies
            }

        with patch.object(
            prompts,
            "TEMPLATE_ROOT",
            Path(__file__).resolve().parents[3] / "research/elitesearch/prompts",
        ):
            agent = PokerSearch(
                "Score",
                ScriptedProvider([program(i) for i in range(3)]),
                evaluate,
                config=Config(population_size=2, generations=1),
            )
            await agent.run()
        self.assertEqual(list(map(len, calls)), [2, 2])
        self.assertEqual(agent.organisms[0].repairs, 1)

    async def test_tournament_scores_and_equal_hands(self):
        from rsikit.policy import Policy

        policies = [
            Policy.from_text(json.loads(program(i))["implementation"], name=str(i))
            for i in range(7)
        ]

        async def runner(request, config):
            return run_block(
                len(request["sources"]),
                request["deals"],
                request["seed"],
                lambda seat, observation: 1,
            )

        tournament = Tournament(TournamentConfig(rounds=2, deals=1), runner=runner)
        results = await tournament(policies)
        self.assertTrue(all(m.scores == {0: 0, 1: 0} for m in results.values()))
        self.assertEqual({r["hands"] for r in tournament.report["leaderboard"]}, {72})


class PokerTests(unittest.TestCase):
    def test_unequal_all_ins_settle_side_pots_and_return_uncalled_chips(self):
        for seed in range(10):
            result = play_hand(
                3, seed, lambda seat, o: -1 if o["max_raise_to"] else 1, stack=(20, 60, 200)
            )
            self.assertEqual(sum(result), 0)
            self.assertTrue(-20 <= result[0] <= 40)
            self.assertTrue(all(-60 <= value <= 80 for value in result[1:]))

    def test_last_chip_can_be_bet_below_the_big_blind(self):
        short_bets = []

        def act(seat, o):
            if o["street"] == 0:
                return 199 if o["max_raise_to"] == 200 and max(o["bets"]) < 199 else 1
            if o["max_raise_to"] == 1:
                short_bets.append(o["min_raise_to"])
                return -1
            return 1

        payoffs = play_hand(2, 123, act)
        self.assertEqual(short_bets, [1])
        self.assertEqual(sorted(payoffs), [-200, 200])

    def test_balanced_reproducible_schedule_with_remainders(self):
        for n in (2, 6, 7, 12, 50, 60):
            groups = schedule(n, 17)
            self.assertEqual(groups, schedule(n, 17))
            counts = Counter(i for group in groups for i in group)
            self.assertEqual(set(counts), set(range(n)))
            self.assertEqual(len(set(counts.values())), 1)
            self.assertTrue(all(len(set(g)) == min(n, 6) for g in groups))

    def test_duplicate_callers_cancel_and_hide_other_cards(self):
        observations = []

        def call(seat, observation):
            observations.append(observation)
            return 1

        result = run_block(6, 3, 17, call)
        self.assertEqual(result["chips"], [0] * 6)
        self.assertEqual(result["hands"], 18)
        self.assertTrue(all(len(o["hole_cards"]) == 2 for o in observations))
        self.assertTrue(all("seed" not in o and "deck" not in o for o in observations))

    def test_fold_and_all_in_conserve_chips(self):
        for seed in range(10):
            for action in (0, 1, 200):
                result = play_hand(
                    6,
                    seed,
                    lambda seat, obs: action if action in (0, 1) else obs["max_raise_to"] or 1,
                )
                self.assertEqual(sum(result), 0)
                self.assertTrue(all(-200 <= chips <= 1000 for chips in result))

    def test_illegal_action_is_attributed(self):
        from .game import CandidateFailure

        for action in (-2, True, 1.5, float("nan"), 9999):
            with self.assertRaises(CandidateFailure) as failure:
                play_hand(6, 0, lambda seat, obs: action)
            self.assertEqual(failure.exception.seat, 2)


if __name__ == "__main__":
    unittest.main()
