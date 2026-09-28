"""Whole rollouts share a process inside Docker, never on the host."""

import os
import tempfile
import unittest
from pathlib import Path

from rsikit import DockerSandbox, Executor, run_program
from rsikit.envs import CirclePackingEnv
from rsikit.evaluation import PolicyError, PolicyTimeout
from tests.test_persistent_sandbox import PACKING


class DockerTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_run_program_deadline_does_not_create_an_environment(self):
        created = []

        def environment():
            created.append(True)
            return CirclePackingEnv(1)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.py"
            path.write_text(PACKING)
            with self.assertRaises(ValueError):
                await run_program(path, environment, episode_timeout=0)
        self.assertEqual(created, [])

    async def test_run_program_moves_environment_into_docker_and_preserves_options(self):
        import gymnasium as gym

        host_pid = os.getpid()

        class Environment(gym.Env):
            observation_space = gym.spaces.Discrete(1)
            action_space = gym.spaces.Discrete(1)

            def reset(self, *, seed=None, options=None):
                import os

                assert os.getpid() != host_pid
                assert seed == 3
                return 0, {}

            def step(self, action):
                return 0, 1, False, False, {}

        source = """from rsikit import Policy
class Solution(Policy):
    async def reset(self, *, seed=None):
        assert seed == 7
        assert self.instructions == "explicit instructions"
    async def act(self, observation):
        return 0
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.py"
            path.write_text(source)
            result = await run_program(
                path,
                Environment,
                env_seed=3,
                policy_seed=7,
                max_steps=2,
                instructions="explicit instructions",
            )
        self.assertEqual(result.total_reward, 2)
        self.assertEqual(result.truncations, [False, True])

    async def test_returns_every_transition(self):
        import gymnasium as gym
        import numpy as np

        source = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, o): return 0\n"
        with gym.make("CartPole-v1", max_episode_steps=3) as env:
            initial, _ = env.reset(seed=42)
            expected = [initial.copy()]
            for _ in range(3):
                obs, *_ = env.step(0)
                expected.append(obs.copy())
            async with Executor(sandbox=DockerSandbox()) as executor:
                results = [
                    episode
                    async for _, _, episode in executor.evaluate([("constant", source, 42)], env)
                ]
            episode = results[0]
            np.testing.assert_array_equal(episode.observations, expected)
            self.assertEqual(episode.actions, [0, 0, 0])
            self.assertEqual(episode.rewards, [1.0, 1.0, 1.0])
            self.assertEqual(episode.terminations, [False, False, False])
            self.assertEqual(episode.truncations, [False, False, True])
            self.assertEqual(len(episode.infos), 4)
            self.assertEqual(episode.infos[-1]["episode"]["l"], 3)

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
        async with Executor(sandbox=DockerSandbox(), concurrency=2) as executor:
            for _ in range(2):
                results = [
                    r
                    async for _, _, r in executor.evaluate(
                        [(str(seed), source, seed) for seed in (1, 2)], Environment(1)
                    )
                ]
                self.assertEqual(len(results), 2)
                for result in results:
                    self.assertEqual(result.total_reward, 0.5)
                    self.assertEqual(
                        result.artifacts["policy.txt"], result.artifacts["environment.txt"]
                    )
                    pids.append(result.artifacts["policy.txt"])
        self.assertEqual(len(set(pids)), 4)

    async def test_container_output_reaches_host_logging_without_corrupting_results(self):
        source = PACKING.replace(
            "return np.array",
            """import os
        print("policy log: café")
        os.write(1, b"raw stdout\\n")
        os.write(2, b"raw stderr\\n")
        return np.array""",
        )
        with self.assertLogs("rsikit.sandbox.docker", level="INFO") as logs:
            async with Executor(sandbox=DockerSandbox()) as executor:
                (result,) = [
                    r
                    async for _, _, r in executor.evaluate(
                        [("logs", source, 1)], CirclePackingEnv(1)
                    )
                ]
        self.assertEqual(result.total_reward, 0.5)
        output = "\n".join(logs.output)
        for text in ("policy log: café", "raw stdout", "raw stderr"):
            self.assertIn(text, output)

    async def test_prints_are_forwarded_even_when_the_episode_times_out(self):
        source = PACKING.replace(
            "return np.array",
            'print("before timeout")\n        while True: pass\n        return np.array',
        )
        with self.assertLogs("rsikit.sandbox.docker", level="INFO") as logs:
            async with Executor(sandbox=DockerSandbox(episode_timeout=0.3)) as executor:
                with self.assertRaises(PolicyTimeout):
                    _ = [
                        r
                        async for r in executor.evaluate([("logs", source, 1)], CirclePackingEnv(1))
                    ]
        self.assertIn("before timeout", "\n".join(logs.output))

    async def test_errors_timeouts_and_process_exit_allow_next_episode(self):
        sandbox = DockerSandbox(episode_timeout=0.3)
        async with Executor(sandbox=sandbox) as executor:
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
                    self.assertEqual(results[0].total_reward, 0.5)
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
        async with Executor(sandbox=DockerSandbox()) as executor:
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
            self.assertEqual(result.total_reward, 0.5)

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
        async with Executor(sandbox=DockerSandbox()) as executor:
            (result,) = [
                r
                async for _, _, r in executor.evaluate([("groups", source, 1)], CirclePackingEnv(1))
            ]
            self.assertEqual(result.total_reward, 0.5)


if __name__ == "__main__":
    unittest.main()
