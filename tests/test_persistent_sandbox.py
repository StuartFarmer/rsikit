"""Persistent Docker service and transport regression checks (requires the image)."""

import asyncio
import base64
import json
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import cloudpickle

from examples.benchmark_docker import PACKING
from rsikit.envs import BlackjackEnv, CirclePackingEnv
from rsikit.evaluation import InfrastructureError, PolicyError, PolicyTimeout
from rsikit.execution import Executor
from rsikit.sandbox.codec import decode_episode, encode_episode
from rsikit.sandbox.docker import DockerSandbox
from rsikit.sandbox.evaluate import ProcessPolicy
from tests.test_episode_storage import trajectory


def frame(job_id, reward=1, artifacts=None):
    return (
        json.dumps(
            {"id": job_id, "result": {"episode": encode_episode(trajectory(reward, artifacts))}}
        ).encode()
        + b"\n"
    )


def request(source=PACKING, environment=None, seed=1, timeout=3):
    return {
        "implementation": source,
        "environment": base64.b64encode(
            cloudpickle.dumps(environment if environment is not None else CirclePackingEnv(count=1))
        ).decode(),
        "python": list(sys.version_info[:2]),
        "seed": seed,
        "call_timeout": timeout,
    }


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.name = f"rsikit-test-{uuid4().hex}"
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--init",
            "-i",
            "--name",
            self.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "2g",
            "--memory-swap",
            "2g",
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--entrypoint",
            "python",
            "rsikit-sandbox:local",
            "-I",
            "-u",
            "-c",
            "from rsikit.sandbox.service import main; main(2)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=64 * 1024 * 1024 + 1,
        )
        self.addAsyncCleanup(self.cleanup_service)
        line = await asyncio.wait_for(self.process.stdout.readline(), 60)
        if not line:
            error = await self.process.stderr.read()
            self.fail(f"Service failed before readiness: {error.decode()}")
        self.ready = json.loads(line)
        self.assertEqual(self.ready["protocol"], 2)
        self.assertIs(self.ready["ready"], True)

    async def cleanup_service(self):
        cleanup = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            self.name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(cleanup.wait(), 15)
        await asyncio.wait_for(self.process.communicate(), 15)

    async def exchange(self, job_id, body):
        self.process.stdin.write(json.dumps({"id": job_id, "request": body}).encode() + b"\n")
        await self.process.stdin.drain()
        return json.loads(await asyncio.wait_for(self.process.stdout.readline(), 10))

    async def test_ready_and_repeated_requests(self):
        self.assertNotEqual(self.ready["supervisor_pid"], self.ready["forkserver_pid"])
        for job_id in ("first", "second"):
            response = await self.exchange(job_id, request())
            self.assertEqual(response["id"], job_id)
            episode = decode_episode(response["result"]["episode"])
            self.assertEqual(episode.total_reward, 0.5)
            self.assertEqual(len(episode.observations), len(episode.actions) + 1)
            self.assertTrue(episode.terminations[-1])
        self.assertIsNone(self.process.returncode)

    async def test_child_death_is_reported_and_service_recovers(self):
        source = PACKING.replace("return np.array", "import os; os._exit(7); return np.array")
        response = await self.exchange("dead", request(source, timeout=0.2))
        self.assertIn(response["result"]["kind"], ("policy", "timeout"))
        self.assertEqual(
            decode_episode(
                (await self.exchange("recovered", request()))["result"]["episode"]
            ).total_reward,
            0.5,
        )

    async def test_excess_concurrency_is_rejected_without_corrupting_results(self):
        source = PACKING.replace(
            "return np.array", "import asyncio; await asyncio.sleep(0.1)\n        return np.array"
        )
        self.process.stdin.write(
            b"".join(
                json.dumps({"id": str(index), "request": request(source)}).encode() + b"\n"
                for index in range(3)
            )
        )
        await self.process.stdin.drain()
        replies = {}
        for _ in range(3):
            result = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 10))
            replies[result["id"]] = result["result"]
        self.assertEqual(decode_episode(replies["0"]["episode"]).total_reward, 0.5)
        self.assertEqual(decode_episode(replies["1"]["episode"]).total_reward, 0.5)
        self.assertEqual(replies["2"]["kind"], "infrastructure")
        self.assertEqual(
            decode_episode(
                (await self.exchange("next", request()))["result"]["episode"]
            ).total_reward,
            0.5,
        )

    async def test_malformed_request_stops_the_service(self):
        self.process.stdin.write(b'{"id":false,"request":{}}\n')
        await self.process.stdin.drain()
        await asyncio.wait_for(self.process.communicate(), 10)
        self.assertNotEqual(self.process.returncode, 0)

    async def test_oversized_request_stops_the_service(self):
        await asyncio.wait_for(self.process.communicate(b"x" * (64 * 1024 * 1024 + 2) + b"\n"), 10)
        self.assertNotEqual(self.process.returncode, 0)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_episode_timeout_rejects_invalid_deadlines(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                DockerSandbox(episode_timeout=timeout)

    async def test_close_drains_a_backpressured_docker_client(self):
        sandbox = DockerSandbox()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x' * 262144)",
            stdout=asyncio.subprocess.PIPE,
            limit=128,
        )
        sandbox.process = process
        try:
            while not process.stdout._paused:
                await asyncio.sleep(0)
            await asyncio.wait_for(sandbox.close(), 1)
            self.assertIsNotNone(process.returncode)
        finally:
            await process.communicate()

    async def test_candidate_timeout_closes_backpressured_socket(self):
        channel, peer = socket.socketpair()
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        env = CirclePackingEnv(1)
        policy = ProcessPolicy(
            env.observation_space,
            env.action_space,
            source=PACKING,
            channel=channel,
            call_timeout=0.05,
        )
        policy.ready = True
        try:
            with self.assertRaises(PolicyTimeout):
                await asyncio.wait_for(
                    policy._request({"command": "start", "source": "x" * 524288}), 0.5
                )
            self.assertFalse(policy.ready)
            self.assertEqual(channel.fileno(), -1)
            await policy._destroy()  # Cleanup remains idempotent.
        finally:
            channel.close()
            peer.close()
            env.close()

    async def test_cancelled_write_invalidates_pending_jobs(self):
        sandbox = DockerSandbox()
        writing = asyncio.Event()

        async def drain():
            writing.set()
            await asyncio.Future()

        sandbox.name = "test"
        sandbox.process = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda data: None, drain=drain)
        )
        sandbox._reader = object()
        sandbox.close = AsyncMock()
        sibling = asyncio.get_running_loop().create_future()
        sandbox._pending["sibling"] = sibling
        pending = asyncio.create_task(sandbox.evaluate(PACKING, b"", 1, 3))
        await writing.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertIsInstance(sibling.exception(), InfrastructureError)
        self.assertEqual(sandbox._pending, {})
        with self.assertRaises(InfrastructureError):
            await sandbox.evaluate(PACKING, b"", 1, 3)

    async def test_duplicate_response_invalidates_other_jobs(self):
        sandbox = DockerSandbox()
        reader = asyncio.StreamReader()
        sandbox.process = SimpleNamespace(stdout=reader)
        sandbox.close = AsyncMock()
        first, second = (asyncio.get_running_loop().create_future() for _ in range(2))
        sandbox._pending = {"a": first, "b": second}
        reader.feed_data(frame("a") * 2)
        reader.feed_eof()
        await sandbox._read_results()
        self.assertEqual((await first).total_reward, 1)
        self.assertIsInstance(second.exception(), InfrastructureError)

    async def test_malformed_frames_fail_every_pending_job(self):
        for data in (
            b'{"id":"unknown","result":{"score":1,"artifacts":{}}}\n',
            b"{bad json}\n",
            b'{"id":"a","result":{"score":1,"artifacts":{}}}',
            b'{"id":"a","result":{"score":NaN,"artifacts":{}}}\n',
            b'{"id":"a","result":{"score":true,"artifacts":{}}}\n',
            b'{"id":"a","result":{"score":1,"artifacts":{"x":"!"}}}\n',
            b"x" * 129 + b"\n",
        ):
            with self.subTest(data=data[:60]):
                sandbox = DockerSandbox()
                reader = asyncio.StreamReader()
                sandbox.process = SimpleNamespace(stdout=reader)
                sandbox.close = AsyncMock()
                pending = {key: asyncio.get_running_loop().create_future() for key in ("a", "b")}
                sandbox._pending = pending.copy()
                reader.feed_data(data)
                reader.feed_eof()
                with patch("rsikit.sandbox.docker.MAX_RESULT", 128):
                    await sandbox._read_results()
                for future in pending.values():
                    self.assertTrue(future.done())
                    self.assertIsInstance(future.exception(), InfrastructureError)
                await asyncio.sleep(0)

    async def test_out_of_order_results_are_delivered_to_the_right_job(self):
        sandbox = DockerSandbox()
        reader = asyncio.StreamReader()
        sandbox.process = SimpleNamespace(stdout=reader)
        sandbox.close = AsyncMock()
        pending = {key: asyncio.get_running_loop().create_future() for key in ("a", "b")}
        sandbox._pending = pending.copy()
        reader.feed_data(frame("b", 2) + frame("a", 1, {"x": b"a"}))
        reader.feed_eof()
        await sandbox._read_results()
        self.assertEqual((await pending["a"]).total_reward, 1)
        self.assertEqual((await pending["a"]).artifacts, {"x": b"a"})
        self.assertEqual((await pending["b"]).total_reward, 2)
        await asyncio.sleep(0)


class PersistentTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_preload_fails_startup_and_removes_container(self):
        from rsikit.sandbox.docker import _spawn

        async def without_preload(*args, **kwargs):
            if args[:2] == ("docker", "run"):
                args = (
                    *args[:-1],
                    "import multiprocessing as mp; "
                    "mp.set_forkserver_preload = lambda _: None; " + args[-1],
                )
            return await _spawn(*args, **kwargs)

        sandbox = DockerSandbox()
        try:
            with patch("rsikit.sandbox.docker._spawn", side_effect=without_preload):
                with self.assertRaises(InfrastructureError):
                    await sandbox.start(1)
            self.assertIsNone(sandbox.name)
            self.assertIsNone(sandbox.process)
        finally:
            await sandbox.close()

    async def test_candidate_cannot_open_control_connections(self):
        source = PACKING.replace(
            "return np.array",
            """import socket
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind("candidate.sock")
            listener.listen(1)
            with socket.socket(socket.AF_UNIX) as client:
                try:
                    client.connect("candidate.sock")
                except PermissionError:
                    pass
                else:
                    raise AssertionError("Candidate can connect to a control socket")
        return np.array""",
        )
        sandbox = DockerSandbox()
        try:
            await sandbox.start(1)
            self.assertEqual(
                (
                    await sandbox.evaluate(source, cloudpickle.dumps(CirclePackingEnv(1)), 1, 3)
                ).total_reward,
                0.5,
            )
        finally:
            await sandbox.close()

    async def test_cancellation_and_supervisor_death_remove_active_service(self):
        for interrupt in ("cancel", "kill"):
            with self.subTest(interrupt=interrupt):
                sandbox = DockerSandbox()
                await sandbox.start(2)
                name, process = sandbox.name, sandbox.process
                # Calls are registered before interruption; the policy need not cooperate.
                bad = PACKING.replace(
                    "return np.array", "while True: pass\n        return np.array"
                )
                jobs = [
                    asyncio.create_task(
                        sandbox.evaluate(bad, cloudpickle.dumps(CirclePackingEnv(1)), seed, 60)
                    )
                    for seed in (1, 2)
                ]
                try:
                    while len(sandbox._pending) < 2:
                        await asyncio.sleep(0)
                    if interrupt == "cancel":
                        jobs[0].cancel()
                    else:
                        killed = await asyncio.create_subprocess_exec(
                            "docker",
                            "exec",
                            name,
                            "sh",
                            "-c",
                            f"kill -9 {sandbox.ready['supervisor_pid']}",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        self.assertEqual(await killed.wait(), 0)
                    results = await asyncio.wait_for(
                        asyncio.gather(*jobs, return_exceptions=True), 10
                    )
                    self.assertTrue(
                        all(
                            isinstance(result, (InfrastructureError, asyncio.CancelledError))
                            for result in results
                        )
                    )
                    self.assertIsNone(sandbox.name)
                    self.assertIsNotNone(process.returncode)
                finally:
                    for job in jobs:
                        job.cancel()
                    await asyncio.gather(*jobs, return_exceptions=True)
                    await sandbox.close()

    async def test_fresh_children_keep_service_alive_and_artifacts_isolated(self):
        class Environment(CirclePackingEnv):
            def reset(self, *, seed=None, options=None):
                import numpy as np

                assert not hasattr(np, "_rsikit_environment_marker")
                np._rsikit_environment_marker = True
                self.test_seed = seed
                return super().reset(seed=seed, options=options)

            def step(self, action):
                import json
                import os
                import time
                from pathlib import Path

                time.sleep(0.15 if self.test_seed == 1 else 0)
                observation, _, terminated, truncated, info = super().step(action)
                info["artifacts"] = {
                    "environment.json": json.dumps(
                        {"pid": os.getpid(), "parent": os.getppid(), "directory": str(Path.cwd())}
                    ).encode()
                }
                return observation, float(self.test_seed), terminated, truncated, info

        source = PACKING.replace(
            "return np.array",
            """import json, os
        from pathlib import Path
        assert not hasattr(np, "_rsikit_candidate_marker")
        np._rsikit_candidate_marker = True
        Path("candidate.json").write_text(json.dumps({"pid": os.getpid(), "parent": os.getppid()}))
        return np.array""",
        )
        sandbox = DockerSandbox()
        children = []
        directories = []
        async with Executor(sandbox=sandbox, concurrency=2) as executor:
            for seeds in ((1, 2), (3, 4)):
                results = [
                    result
                    async for result in executor.evaluate(
                        [(str(seed), source, seed) for seed in seeds], Environment(count=1)
                    )
                ]
                if seeds == (1, 2):
                    ready, process, name = sandbox.ready.copy(), sandbox.process, sandbox.name
                    self.assertEqual([int(policy_id) for policy_id, _, _ in results], [2, 1])
                self.assertEqual(sandbox.ready, ready)
                self.assertIs(sandbox.process, process)
                self.assertEqual(sandbox.name, name)
                for policy_id, seed, result in results:
                    self.assertEqual(result.total_reward, float(policy_id))
                    for key in ("candidate.json", "environment.json"):
                        identity = json.loads(result.artifacts[key])
                        self.assertEqual(identity["parent"], ready["forkserver_pid"])
                        children.append(identity["pid"])
                        if "directory" in identity:
                            directories.append(identity["directory"])
            self.assertEqual(len(set(children)), 8)
            self.assertIsNone(process.returncode)

            class CheckCleanup(CirclePackingEnv):
                def reset(self, **kwargs):
                    from pathlib import Path

                    assert all(not Path(directory).exists() for directory in directories)
                    return super().reset(**kwargs)

            results = [
                item async for item in executor.evaluate([("cleanup", PACKING, 1)], CheckCleanup(1))
            ]
            self.assertEqual(results[0][2].total_reward, 0.5)
        self.assertIsNone(sandbox.name)
        self.assertIsNotNone(process.returncode)

    async def test_timeout_then_success_reuses_service(self):
        sandbox = DockerSandbox()
        bad = PACKING.replace("return np.array", "while True: pass\n        return np.array")
        async with Executor(sandbox=sandbox, call_timeout=0.1) as executor:
            with self.assertRaises(PolicyTimeout):
                _ = [
                    item async for item in executor.evaluate([("bad", bad, 1)], CirclePackingEnv(1))
                ]
            ready, process = sandbox.ready.copy(), sandbox.process
            results = [
                item async for item in executor.evaluate([("ok", PACKING, 1)], CirclePackingEnv(1))
            ]
            self.assertEqual(results[0][2].total_reward, 0.5)
            self.assertEqual(sandbox.ready, ready)
            self.assertIs(sandbox.process, process)

    async def test_episode_timeout_preserves_siblings_and_service(self):
        fast = (
            "from rsikit import Policy\nclass Solution(Policy):\n"
            "    async def act(self, observation): return 4\n"
        )
        slow = fast.replace(
            "return 4",
            "\n        import asyncio\n        await asyncio.sleep(0.05)\n        return 4",
        )
        sandbox = DockerSandbox(episode_timeout=0.5)
        async with Executor(sandbox=sandbox, concurrency=2, call_timeout=3) as executor:
            results = []
            with self.assertRaisesRegex(PolicyTimeout, "Episode.*0.5s") as caught:
                async for result in executor.evaluate(
                    [("slow", slow, 0), ("fast", fast, 1)], BlackjackEnv(shoes_per_episode=1)
                ):
                    results.append(result)
            self.assertEqual(set(caught.exception.failures), {"slow"})
            self.assertEqual([(p, s, r.total_reward) for p, s, r in results], [("fast", 1, 0)])
            ready, process = sandbox.ready.copy(), sandbox.process
            results = [
                result
                async for result in executor.evaluate(
                    [("repaired", fast, 0)], BlackjackEnv(shoes_per_episode=1)
                )
            ]
            self.assertEqual(results[0][2].total_reward, 0)
            self.assertEqual(sandbox.ready, ready)
            self.assertIs(sandbox.process, process)

    async def test_illegal_blackjack_action_preserves_siblings_and_service(self):
        bad = (
            "from rsikit import Policy\nclass Solution(Policy):\n"
            "    async def act(self, observation): return 3\n"
        )
        repaired = bad.replace("return 3", "return 4")
        sandbox = DockerSandbox()
        async with Executor(sandbox=sandbox, concurrency=2) as executor:
            results = []
            with self.assertRaisesRegex(PolicyError, "legal-action mask") as caught:
                async for result in executor.evaluate(
                    [("bad", bad, 0), ("good", repaired, 1)], BlackjackEnv()
                ):
                    results.append(result)
            self.assertEqual(set(caught.exception.failures), {"bad"})
            self.assertEqual([(p, s, r.total_reward) for p, s, r in results], [("good", 1, 0)])
            self.assertIsNone(sandbox._failure)
            ready, process = sandbox.ready.copy(), sandbox.process
            results = [
                result async for result in executor.evaluate([("bad", repaired, 0)], BlackjackEnv())
            ]
            self.assertEqual(results[0][2].total_reward, 0)
            self.assertEqual(sandbox.ready, ready)
            self.assertIs(sandbox.process, process)

    async def test_environment_death_fails_all_waiters_and_restart_works(self):
        class Broken(CirclePackingEnv):
            def reset(self, **kwargs):
                import os

                os._exit(8)

        sandbox = DockerSandbox()
        try:
            await sandbox.start(2)
            results = await asyncio.wait_for(
                asyncio.gather(
                    sandbox.evaluate(PACKING, cloudpickle.dumps(Broken(1)), 1, 3),
                    sandbox.evaluate(PACKING, cloudpickle.dumps(Broken(1)), 2, 3),
                    return_exceptions=True,
                ),
                10,
            )
            self.assertTrue(all(isinstance(result, InfrastructureError) for result in results))
            await sandbox.close()
            await sandbox.start(1)
            self.assertEqual(
                (
                    await sandbox.evaluate(PACKING, cloudpickle.dumps(CirclePackingEnv(1)), 1, 3)
                ).total_reward,
                0.5,
            )
        finally:
            await sandbox.close()


if __name__ == "__main__":
    unittest.main()
