"""Small codec checks and optional real-Docker episode smoke checks."""

import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout, run_episode
from rsikit.policy import Policy
from rsikit.sandbox import SandboxPolicy, run_program
from rsikit.sandbox.codec import decode, decode_space, dumps, encode, encode_space, loads

INSTRUCTIONS = "Count from zero.\nPreserve café and π exactly."
COUNTER_SOURCE = """
import asyncio
import os
from rsikit import Policy

class Solution(Policy):
    async def reset(self, *, seed=None):
        await super().reset(seed=seed)
        self.count = 0 if self.instructions == "Count from zero.\\nPreserve café and π exactly." else 5
        self.loop = asyncio.get_running_loop()
    async def act(self, observation):
        assert asyncio.get_running_loop() is self.loop
        assert os.getuid() != 0
        print("candidate output must not become protocol")
        os.write(1, b"raw stdout is separate too\\n")
        action = self.count
        self.count += 1
        return action
"""


class CounterEnv(gym.Env):
    instructions = INSTRUCTIONS

    def __init__(self):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return 0, {"hidden": "record only"}

    def step(self, action):
        self.count += 1
        return self.count, float(action + 1), self.count == 2, False, {}


class CounterPolicy(Policy):
    async def reset(self, *, seed=None):
        await super().reset(seed=seed)
        self.count = 0 if self.instructions == INSTRUCTIONS else 5

    async def act(self, observation):
        action = self.count
        self.count += 1
        return action


class CodecSmoke(unittest.TestCase):
    def test_round_trip_and_rejected_allocations(self):
        value = {"a": (np.array([[1, 2]], dtype=np.int16), [True, None, "π", float("inf")]), 3: 4}
        restored = decode(loads(dumps(encode(value))))
        np.testing.assert_array_equal(restored["a"][0], value["a"][0])
        self.assertEqual(restored["a"][0].dtype, np.int16)
        self.assertIsInstance(restored["a"], tuple)
        self.assertEqual(restored["a"][1], value["a"][1])
        self.assertEqual(restored[3], 4)
        space = spaces.Dict(
            [
                ("z", spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)),
                (
                    "a",
                    spaces.Tuple(
                        (spaces.Discrete(3, start=-1), spaces.Text(8, min_length=0, charset="aπ"))
                    ),
                ),
            ]
        )
        recovered = decode_space(loads(dumps(encode_space(space))))
        self.assertEqual(list(recovered.spaces), ["z", "a"])
        self.assertEqual(recovered, space)
        for encoded in (
            ["array", "O", [1], "AAAAAAAAAAA="],
            ["array", "f8", [1_000_000], ""],
            ["array", "f8", [1], ""],
        ):
            with self.assertRaises((ValueError, TypeError)):
                decode(encoded)
        with self.assertRaises(InfrastructureError):
            SandboxPolicy(
                spaces.MultiDiscrete([2, 3]),
                spaces.Discrete(2),
                instructions="",
                source="raise AssertionError('must not run')",
            )


class ClientFailureSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_cannot_replace_timeout_or_cancellation(self):
        for trigger, expected in (
            (asyncio.TimeoutError, PolicyTimeout),
            (asyncio.CancelledError, asyncio.CancelledError),
        ):
            policy = SandboxPolicy(
                spaces.Discrete(2), spaces.Discrete(2), instructions="", source=COUNTER_SOURCE
            )
            policy.ready = True
            policy.process = SimpleNamespace(
                stdin=SimpleNamespace(write=lambda _: None, drain=AsyncMock()),
                stdout=SimpleNamespace(readline=AsyncMock(side_effect=trigger)),
            )
            policy._destroy = AsyncMock(side_effect=InfrastructureError("cleanup unavailable"))
            with self.assertRaises(expected) as caught:
                await policy.act(0)
            self.assertIsInstance(caught.exception.cleanup_error, InfrastructureError)


class SandboxEpisodeSmoke(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("docker") is None:
            raise unittest.SkipTest("Docker is not installed; no local source execution fallback")
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", "rsikit-sandbox:local"],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise unittest.SkipTest(f"Docker unavailable: {exc}") from exc
        if result.returncode:
            raise unittest.SkipTest("Docker or rsikit-sandbox:local image unavailable")

    async def run_source(self, source, *, timeout=3):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.py"
            path.write_text(source, encoding="utf-8")
            return await run_program(
                path, CounterEnv, env_seed=1, policy_seed=2, max_steps=5, call_timeout=timeout
            )

    async def test_state_instructions_and_examples(self):
        local = await run_episode(CounterEnv, CounterPolicy, env_seed=1, policy_seed=2, max_steps=5)
        for _ in range(2):
            isolated = await self.run_source(COUNTER_SOURCE)
            self.assertEqual(isolated[:4], local[:4])
            self.assertEqual(isolated[4]["episode"]["r"], 3.0)
            self.assertEqual(isolated[4]["episode"]["l"], 2)
        from examples import cartpole
        from examples.circle_packing import initial
        from rsikit.envs import CirclePackingEnv

        packing = await run_program(
            Path(initial.__file__), CirclePackingEnv, env_seed=1, policy_seed=2, max_steps=5
        )
        self.assertTrue(packing[2])
        self.assertAlmostEqual(packing[4]["episode"]["r"], 1.0)
        cart = await run_program(
            Path(cartpole.__file__),
            "CartPole-v1",
            env_seed=1,
            policy_seed=2,
            max_steps=8,
        )
        self.assertTrue(cart[2] or cart[3])
        self.assertGreater(cart[4]["episode"]["l"], 1)

    async def test_alphaevolve_uses_native_environments_and_isolated_evaluation(self):
        from functools import partial

        from slick import prompts

        import rsikit.alphaevolve as alphaevolve
        from examples import cartpole
        from examples.alphaevolve import INITIAL
        from rsikit.alphaevolve import AlphaEvolve, Config, evaluate_program
        from rsikit.alphaevolve.edits import Program
        from tests.providers import ScriptedProvider

        provider = ScriptedProvider([Program(source=Path(cartpole.__file__).read_text())])
        evaluate = partial(evaluate_program, make_env="CartPole-v1", seeds=(1, 2), max_steps=50)
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent / "prompts"):
            agent = AlphaEvolve(
                "Balance CartPole", provider, evaluate, config=Config(mode="rewrite", islands=1)
            )
            best = await agent.run(INITIAL, attempts=1, concurrency=1)
        self.assertGreater(best.metrics["reward"], agent.programs[0].metrics["reward"])
        self.assertEqual(best.metrics["reward"], 50)
        self.assertIn('"seed": 2', best.feedback)
        self.assertEqual(len(provider.calls), 1)
        rejected = await evaluate(INITIAL.replace("return 0", "return 50"))
        self.assertIn("PolicyError", rejected.error)
        self.assertEqual(rejected.metrics, {})
        with (
            patch(
                "rsikit.alphaevolve.evaluation.run_program",
                new=AsyncMock(side_effect=InfrastructureError("Docker unavailable")),
            ),
            self.assertRaisesRegex(InfrastructureError, "Docker unavailable"),
        ):
            await evaluate(INITIAL)

    async def test_inner_loop_demo_passes_generated_policies_to_run(self):
        import contextlib
        import io
        import json

        from slick import prompts

        import examples.inner_loop as demo
        import rsikit.generation as generation
        from rsikit import Run
        from tests.providers import ScriptedProvider

        def response(name, action):
            return json.dumps(
                {
                    "name": name,
                    "implementation": (
                        "from rsikit import Policy\nclass Solution(Policy):\n"
                        f"    async def act(self, observation):\n        return {action}\n"
                    ),
                }
            )

        provider = ScriptedProvider(
            [
                response("Dice", "self.action_space.sample()"),
                response("Angle", "int(observation[2] > 0)"),
                response("PD", "int(observation[2] + 0.5 * observation[3] > 0)"),
                response("Broken", "50"),
                response("Other", "int(observation[2] + 0.6 * observation[3] > 0)"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            with patch.object(
                prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts"
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = await demo.run_demo(provider, output, seeds=(1, 2), max_steps=30)
            self.assertEqual(len(provider.calls), 5)
            self.assertEqual(len(list((output / "exports").glob("*.py"))), 5)
            self.assertEqual(json.loads((output / "results.json").read_text()), result)
            self.assertEqual(result[2]["name"], "PD")
            self.assertEqual(result[2]["mean_score"], 30)
            self.assertIsNone(result[3]["mean_score"])
            with Run.open(output) as run:
                self.assertEqual(len(run.policies()), 5)
                self.assertEqual(sum(len(run.scores(p)) for p in run.policies()), 10)
                with self.assertRaises(PolicyError):
                    await run.resume()

    async def test_run_records_real_video_artifact(self):
        import json
        from importlib.util import find_spec

        from slick import prompts

        import rsikit.generation as generation
        from rsikit import Run, generate
        from tests.providers import ScriptedProvider

        if find_spec("moviepy") is None or find_spec("pygame") is None:
            self.skipTest("Install .[video] for video checks")
        with patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts"):
            policy = await generate(
                "Balance CartPole",
                provider=ScriptedProvider(
                    [
                        json.dumps(
                            {
                                "name": "Steady",
                                "implementation": (
                                    "from rsikit import Policy\nclass Solution(Policy):\n"
                                    "    async def act(self, observation):\n"
                                    "        return int(observation[2] + 0.5 * observation[3] > 0)\n"
                                ),
                            }
                        )
                    ]
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "video"
            with Run.create(
                name="video", path=output, environment="CartPole-v1", max_steps=5, record_video=True
            ) as run:
                self.assertEqual(await run.evaluate(policy, seeds=[1]), {policy.id: {1: 5.0}})
                videos = list((output / "videos" / policy.id / "1").glob("*.mp4"))
                self.assertEqual(len(videos), 1)
                video = videos[0]
                self.assertGreater(video.stat().st_size, 100)
                from moviepy import VideoFileClip

                with VideoFileClip(str(video)) as clip:
                    self.assertGreater(clip.duration, 0)
                    self.assertEqual(clip.get_frame(0).shape[2], 3)
            with Run.open(output) as run:
                self.assertEqual(run.scores(run.policies()[0]), {1: 5.0})
                self.assertTrue(video.exists())

    async def test_failure_deadline_and_cleanup(self):
        for body in ("raise RuntimeError('candidate failure')", "return object()", "os._exit(3)"):
            with self.assertRaises(PolicyError):
                await self.run_source(COUNTER_SOURCE.replace("return action", body))
        timeout_source = COUNTER_SOURCE.replace("return action", "while True: pass")
        instances = []

        def factory(observation_space, action_space, *, instructions):
            policy = SandboxPolicy(
                observation_space,
                action_space,
                instructions=instructions,
                source=timeout_source,
                call_timeout=0.5,
            )
            instances.append(policy)
            return policy

        with self.assertRaises(PolicyTimeout):
            await run_episode(CounterEnv, factory, env_seed=1, policy_seed=2, max_steps=5)
        self.assertIsNone(instances[0].process)
        result = await asyncio.create_subprocess_exec(
            "docker",
            "inspect",
            instances[0].container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.assertNotEqual(await result.wait(), 0)
        with self.assertRaises(InfrastructureError):
            await run_episode(
                CounterEnv,
                lambda obs, act, **kw: SandboxPolicy(
                    obs,
                    act,
                    **kw,
                    source=COUNTER_SOURCE,
                    image="rsikit-intentionally-missing:local",
                ),
                env_seed=1,
                policy_seed=2,
                max_steps=5,
            )


if __name__ == "__main__":
    unittest.main()
