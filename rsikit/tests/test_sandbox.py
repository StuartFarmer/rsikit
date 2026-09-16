"""Exercise the real persistent container without any model calls."""

import asyncio
import os
import unittest

from rsikit.sandbox import PythonSandbox


@unittest.skipUnless(
    os.environ.get("RSIKIT_DOCKER_TESTS") == "1", "Set RSIKIT_DOCKER_TESTS=1 for Docker integration"
)
class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_isolation_errors_and_timeout_recovery(self):
        async with PythonSandbox(timeout=0.3) as sandbox:
            process = sandbox.process
            source = """def pack_circles():
    import math
    import numpy as np
    from scipy.optimize import minimize
    assert not hasattr(math, "candidate_marker")
    math.candidate_marker = True
    print("this must not corrupt the protocol")
    return [float(np.sqrt(4)), float(minimize(lambda x: x[0]**2, [1.0]).x[0])]
"""
            for _ in range(2):
                result = await sandbox(source)
                self.assertEqual(result["value"][0], 2.0)
                self.assertAlmostEqual(result["value"][1], 0.0, places=5)
            for body, message in [
                ("raise RuntimeError('broken')", "RuntimeError: broken"),
                ("while True: pass", "timeout"),
                ("import os; os._exit(1)", "without a JSON result"),
                ("return [float('nan')]", "ValueError"),
                ("return 'x' * 100000", "64 KiB"),
                ("open('/forbidden', 'w')", "Read-only file system"),
                (
                    "import socket; socket.create_connection(('1.1.1.1', 443), timeout=.1)",
                    "Network is unreachable",
                ),
                ("import os; os.fork()", "BlockingIOError"),
            ]:
                with self.subTest(body=body):
                    result = await sandbox("def pack_circles():\n    " + body)
                    self.assertIn(message, result["error"])
            self.assertEqual(await sandbox("def pack_circles(): return []"), {"value": []})
            self.assertIs(sandbox.process, process)
        self.assertIsNotNone(process.returncode)

    async def test_cancellation_removes_container(self):
        sandbox = PythonSandbox(timeout=10)

        async def run():
            async with sandbox:
                await sandbox("def pack_circles():\n    while True: pass")

        task = asyncio.create_task(run())
        while sandbox.process is None:
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(sandbox.process)
