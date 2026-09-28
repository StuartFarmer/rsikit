"""Opt-in real sandbox checks: POKER_DOCKER_TESTS=1 python -m unittest ..."""

import asyncio
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from research.elitesearch import Config
from tests.providers import ScriptedProvider

from .search import PokerSearch
from .tournament import CONTRACT, Tournament, TournamentConfig, docker_block

CALLER = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, o):\n        return 1\n"


@unittest.skipUnless(os.environ.get("POKER_DOCKER_TESTS") == "1", "requires poker Docker image")
class DockerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_container_reused_after_parallel_tables_and_cancellation(self):
        from .tournament import TablePool

        config = TournamentConfig(workers=2, rounds=1, deals=2)
        request = dict(
            sources=[CALLER] * 2, deals=2, stack=200, seed=42, call_timeout=2, instructions=CONTRACT
        )
        async with TablePool(config) as pool:
            await pool.start()
            name, forkserver = pool.name, pool.ready["forkserver_pid"]
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format",
                "{{.HostConfig.Memory}}",
                name,
                stdout=asyncio.subprocess.PIPE,
            )
            output, _ = await inspect.communicate()
            self.assertEqual(int(output), 32 * 1024**3)
            first, second = await asyncio.gather(pool(request, config), pool(request, config))
            self.assertEqual(first["chips"], [0, 0])
            self.assertEqual(second["chips"], [0, 0])
            self.assertNotEqual(first["uid_base"], second["uid_base"])
            self.assertNotEqual(first["worker_pid"], second["worker_pid"])
            slow = {
                **request,
                "sources": [CALLER.replace("return 1", "\n        while True: pass")] * 2,
            }
            pending = asyncio.create_task(pool(slow, config))
            paced = CALLER.replace(
                "return 1", "import time\n        time.sleep(0.04)\n        return 1"
            )
            survivor = asyncio.create_task(
                pool({**request, "sources": [paced, CALLER], "deals": 10, "policy_ms": 100}, config)
            )
            await asyncio.sleep(1)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 5)
            self.assertEqual((await asyncio.wait_for(survivor, 10))["chips"], [0, 0])
            from rsikit.evaluation import InfrastructureError

            with self.assertRaisesRegex(InfrastructureError, "deadline"):
                await pool(slow, replace(config, block_timeout=0.05))
            again = await pool(request, config)
            self.assertEqual(again["chips"], [0, 0])
            self.assertEqual(pool.name, name)
            self.assertEqual(pool.ready["forkserver_pid"], forkserver)
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                name,
                "python",
                "-c",
                "from pathlib import Path; print(sum(int(p.stat().st_uid) >= 1000 for p in Path('/proc').iterdir() if p.name.isdigit()))",
                stdout=asyncio.subprocess.PIPE,
            )
            output, _ = await inspect.communicate()
            self.assertEqual(output.strip(), b"0", "Player processes survived job cleanup")
        self.assertIsNotNone(pool.process.returncode)

    async def test_numba_compiles_once_per_rotation_and_instances_reset_each_hand(self):
        source = """from numba import njit
from rsikit import Policy
@njit(cache=False)
def strength(n):
    total = 0
    for i in range(n):
        total += i % 7
    return total
hands = 0
class Solution(Policy):
    async def reset(self, *, seed=None):
        global hands
        assert not hasattr(self, "first"), "Policy instance was reused"
        await super().reset(seed=seed)
        if hands:
            assert strength.signatures, "JIT was lost between hands"
        hands += 1
        self.first = True
    async def act(self, o):
        assert strength(10000) > 0
        self.first = False
        return 1
"""
        result = await self.block([source] * 2)
        self.assertNotIn("failure", result, result)
        self.assertIn("timings", result)

    async def test_slow_policy_gets_repairable_speed_failure_and_live_progress(self):
        slow = CALLER.replace("return 1", "import time\n        time.sleep(0.04)\n        return 1")
        with self.assertLogs(f"{__package__}.tournament", level="INFO") as logs:
            result = await docker_block(
                dict(
                    sources=[slow, CALLER],
                    deals=30,
                    stack=200,
                    seed=42,
                    call_timeout=2,
                    instructions=CONTRACT,
                    policy_ms=5,
                    warmup_timeout=15,
                ),
                TournamentConfig(),
            )
        self.assertEqual(result.get("failure"), 0, result)
        self.assertIn("Speed budget", result["error"])
        self.assertIn("hands", "\n".join(logs.output))

    async def test_two_generation_self_play_without_model_calls(self):
        proposals = [
            json.dumps(
                dict(
                    name=f"Caller {i}",
                    description="Scripted smoke policy",
                    implementation=CALLER.replace("return 1", f"marker = {i}\n        return 1"),
                )
            )
            for i in range(4)
        ]
        tournament = Tournament(TournamentConfig(rounds=1, deals=1, workers=2))
        from .tournament import TablePool

        async with TablePool(tournament.config) as pool:
            tournament.runner = pool
            with patch.object(prompts, "TEMPLATE_ROOT", Path(__file__).with_name("prompts")):
                agent = PokerSearch(
                    "Smoke check",
                    ScriptedProvider(proposals),
                    tournament,
                    config=Config(
                        population_size=2,
                        elite_size=2,
                        generations=2,
                        new_fraction=1,
                        remix_fraction=0,
                    ),
                )
                await agent.run()
            self.assertEqual(pool.sequence, 2, "One tournament per generation")
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(set(agent.history[1]["scores"]), {"1", "2", "3", "4"})
        self.assertTrue(all(score == 0 for score in agent.history[1]["scores"].values()))

    async def test_tournament_batches_failures_and_reuses_unaffected_blocks(self):
        from rsikit.policy import Policy

        from .tournament import TablePool

        policies = [
            Policy.from_text(
                CALLER.replace(
                    "return 1", f"marker = {i}\n        return " + ("-2" if i in (0, 2) else "1")
                ),
                name=str(i),
                description="Repair test",
            )
            for i in range(6)
        ]
        tournament = Tournament(TournamentConfig(rounds=1, deals=2, workers=2))
        with patch(f"{__package__}.tournament.schedule", return_value=[[0, 1], [2, 3], [4, 5]]):
            async with TablePool(tournament.config) as pool:
                tournament.runner = pool
                results = await tournament(policies)
                self.assertEqual(
                    {pid for pid, m in results.items() if m.failure},
                    {policies[0].id, policies[2].id},
                )
                self.assertEqual(pool.sequence, 3)
                for i in (0, 2):
                    policies[i] = Policy.from_text(
                        policies[i]._implementation.replace("return -2", "return 1"), name=str(i)
                    )
                results = await tournament(policies)
                self.assertTrue(all(m.scores == {0: 0} for m in results.values()))
                self.assertEqual(pool.sequence, 5)
                self.assertEqual(tournament.report["reused_blocks"], 1)

    async def block(self, sources, timeout=2):
        return await docker_block(
            dict(
                sources=sources,
                deals=2,
                stack=200,
                seed=42,
                call_timeout=timeout,
                instructions=CONTRACT,
            ),
            TournamentConfig(),
        )

    async def test_duplicate_symmetry_with_six_isolated_players(self):
        result = await self.block([CALLER] * 6)
        self.assertEqual(result["chips"], [0] * 6)

    async def test_invalid_action_and_timeout_are_attributed(self):
        for replacement in ("return -2", "\n        while True: pass"):
            broken = CALLER.replace("return 1", replacement)
            result = await self.block([CALLER, CALLER, broken, CALLER, CALLER, CALLER], timeout=0.5)
            self.assertEqual(result["failure"], 2)

    async def test_private_uid_and_supervisor_memory_boundary(self):
        source = """import os
import socket
from rsikit import Policy
class Solution(Policy):
    async def act(self, o):
        assert os.getuid() >= 1000
        try:
            os.setsid()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Policy escaped its table process group")
        try:
            open("/proc/1/environ", "rb").read()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Supervisor environment is readable")
        try:
            socket.socket().connect(("127.0.0.1", 1))
        except PermissionError:
            pass
        else:
            raise RuntimeError("Connection filter is missing")
        return 1
"""
        result = await self.block([source] * 2)
        self.assertNotIn("failure", result)


if __name__ == "__main__":
    unittest.main()
