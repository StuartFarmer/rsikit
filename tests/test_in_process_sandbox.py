"""Whole rollouts share a process inside Docker, never on the host."""

import unittest

from rsikit import Executor, InProcessDockerSandbox
from rsikit.envs import CirclePackingEnv
from rsikit.episode import PolicyError, PolicyTimeout
from tests.test_persistent_sandbox import PACKING


class InProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_process_fresh_episodes_and_artifacts(self):
        class Environment(CirclePackingEnv):
            def step(self, action):
                import os

                obs, reward, terminated, truncated, info = super().step(action)
                info["artifacts"] = {"environment.txt": str(os.getpid()).encode()}
                return obs, reward, terminated, truncated, info

        source = PACKING.replace(
            "return np.array",
            """import os, socket
        from pathlib import Path
        assert os.getuid() != 0
        assert not hasattr(np, "_direct_episode_marker")
        np._direct_episode_marker = True
        try:
            Path("/opt/worker/should-not-exist").write_text("x")
        except OSError:
            pass
        else:
            raise AssertionError("Root filesystem is writable")
        with socket.socket() as sock:
            try:
                sock.connect(("1.1.1.1", 80))
            except PermissionError:
                pass
            else:
                raise AssertionError("Connections are allowed")
        Path("policy.txt").write_text(str(os.getpid()))
        return np.array""",
        )
        pids = []
        async with Executor(concurrency=2) as executor:
            for _ in range(2):
                results = [
                    r
                    async for _, _, r in executor.evaluate(
                        [(str(seed), source, seed) for seed in (1, 2)], Environment(1)
                    )
                ]
                self.assertEqual(len(results), 2)
                for result in results:
                    self.assertEqual(result.score, 0.5)
                    self.assertEqual(
                        result.artifacts["policy.txt"], result.artifacts["environment.txt"]
                    )
                    pids.append(result.artifacts["policy.txt"])
        self.assertEqual(len(set(pids)), 4)

    async def test_errors_timeouts_and_process_exit_allow_next_episode(self):
        sandbox = InProcessDockerSandbox(episode_timeout=0.3)
        async with Executor(sandbox=sandbox, call_timeout=0.001) as executor:
            cases = [
                ("invalid python!", PolicyError),
                (
                    PACKING.replace(
                        "return np.array", "raise ValueError('bad policy')\n        return np.array"
                    ),
                    PolicyError,
                ),
                (
                    PACKING.replace(
                        "return np.array", "import os; os._exit(8)\n        return np.array"
                    ),
                    PolicyError,
                ),
                (
                    PACKING.replace("return np.array", "while True: pass\n        return np.array"),
                    PolicyTimeout,
                ),
            ]
            for source, error in cases:
                with self.subTest(error=error.__name__):
                    with self.assertRaises(error):
                        _ = [
                            r
                            async for r in executor.evaluate(
                                [("bad", source, 1)], CirclePackingEnv(1)
                            )
                        ]
                    results = [
                        r
                        async for _, _, r in executor.evaluate(
                            [("good", PACKING, 1)], CirclePackingEnv(1)
                        )
                    ]
                    self.assertEqual(results[0].score, 0.5)
                    self.assertIsNone(sandbox._failure)

    async def test_subprocess_cleanup_and_episode_only_deadline(self):
        source = PACKING.replace(
            "return np.array",
            """import subprocess, sys, time
        from pathlib import Path
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path("child.pid").write_text(str(child.pid))
        time.sleep(0.03)
        return np.array""",
        )
        async with Executor(sandbox=InProcessDockerSandbox(), call_timeout=0.001) as executor:
            (result,) = [
                r
                async for _, _, r in executor.evaluate([("spawn", source, 1)], CirclePackingEnv(1))
            ]
            pid = int(result.artifacts["child.pid"])
            check = PACKING.replace(
                "return np.array",
                f"""from pathlib import Path
        status = Path("/proc/{pid}/stat")
        if status.exists():
            assert status.read_text().split()[2] == "Z", "Episode subprocess is still running"
        return np.array""",
            )
            (result,) = [
                r async for _, _, r in executor.evaluate([("check", check, 1)], CirclePackingEnv(1))
            ]
            self.assertEqual(result.score, 0.5)

    async def test_subprocesses_cannot_escape_episode_process_group(self):
        source = PACKING.replace(
            "return np.array",
            """import subprocess, sys
        try:
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        except PermissionError:
            pass
        else:
            raise AssertionError("Child escaped into a new session")
        subprocess.run([sys.executable, "-c", "import os\\ntry: os.setpgid(0, 0)\\nexcept PermissionError: pass\\nelse: raise AssertionError('Child changed process group')"], check=True)
        return np.array""",
        )
        async with Executor(sandbox=InProcessDockerSandbox()) as executor:
            (result,) = [
                r
                async for _, _, r in executor.evaluate([("groups", source, 1)], CirclePackingEnv(1))
            ]
            self.assertEqual(result.score, 0.5)


if __name__ == "__main__":
    unittest.main()
