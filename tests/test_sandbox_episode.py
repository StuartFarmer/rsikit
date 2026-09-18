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

    async def test_box2d_examples_preserve_instructions_and_evaluate_multiple_seeds(self):
        import importlib.util
        import io
        import math

        import cloudpickle
        from rich.console import Console
        from slick import prompts

        import alphaevolve
        from alphaevolve.edits import Program
        from alphaevolve.improved import AlphaEvolve
        from examples.alphaevolve import make_environment, run_search
        from rsikit import Run
        from tests.providers import ScriptedProvider

        if importlib.util.find_spec("Box2D") is None:
            self.skipTest("Install the box2d extra to test Box2D environments")
        for name, limit, actions in (("LunarLander-v3", 1000, 2), ("BipedalWalker-v3", 1600, 4)):
            with self.subTest(environment=name):
                with make_environment(name) as native:
                    self.assertEqual(native.spec.max_episode_steps, limit)
                    if name == "LunarLander-v3":
                        self.assertTrue(
                            native.unwrapped.continuous and native.unwrapped.enable_wind
                        )
                with (
                    tempfile.TemporaryDirectory() as directory,
                    make_environment(name, max_steps=3) as env,
                    cloudpickle.loads(cloudpickle.dumps(env)) as restored,
                    Run.create(
                        name="box2d", path=Path(directory) / "run", environment=restored
                    ) as run,
                    patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
                ):
                    self.assertEqual(restored.instructions, env.instructions)
                    self.assertEqual(restored.spec.max_episode_steps, 3)
                    provider = ScriptedProvider(
                        [
                            Program(
                                name="Random motors",
                                description="Exercise continuous actions and environment instructions.",
                                implementation=f"""from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        assert {name!r} in self.instructions
        assert "truncated after 3 steps" in self.instructions
        assert self.action_space.shape == ({actions},)
        return self.action_space.sample()
""",
                            )
                        ]
                    )
                    agent = AlphaEvolve("Maximize reward", provider, context=restored.instructions)
                    await run_search(
                        agent,
                        run,
                        generations=1,
                        batch_size=1,
                        seeds=[0, 1],
                        console=Console(file=io.StringIO()),
                    )
                    self.assertIn(restored.instructions, provider.calls[0])
                    scores = run.scores(agent.best)
                    self.assertEqual(set(scores), {0, 1})
                    self.assertTrue(all(math.isfinite(score) for score in scores.values()))
                    self.assertEqual(agent.completed, 1)
                    self.assertEqual(agent._best.seed_scores, scores)

    async def test_shinkaevolve_runs_all_presets_and_persists_islands(self):
        import io
        import math

        from rich.console import Console
        from slick import prompts
        from sqlmodel import select

        import shinkaevolve
        from alphaevolve.edits import Program
        from examples.alphaevolve import make_environment
        from examples.shinkaevolve import run_search
        from rsikit import Executor, Run
        from shinkaevolve import Config, Evaluation, Generation, ShinkaEvolve
        from tests.providers import ScriptedProvider

        try:
            import Box2D  # noqa: F401
        except ImportError:
            self.skipTest("Install the box2d extra to test Box2D environments")
        implementation = """from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return self.action_space.sample()
"""
        for name in ("CartPole-v1", "LunarLander-v3", "BipedalWalker-v3"):
            with (
                self.subTest(environment=name),
                tempfile.TemporaryDirectory() as directory,
                make_environment(name, max_steps=3) as environment,
                patch.object(
                    prompts, "TEMPLATE_ROOT", Path(shinkaevolve.__file__).parent / "prompts"
                ),
                Run.create(
                    name="shinka-smoke",
                    path=Path(directory) / "run",
                    environment=environment,
                    executor=Executor(concurrency=2),
                ) as run,
            ):
                provider = ScriptedProvider(
                    [
                        Program(
                            name="RandomPolicy",
                            description="Sample a valid action.",
                            implementation=implementation,
                        ),
                        Program(
                            name="RandomPolicyV2",
                            description="Sample another valid action.",
                            implementation=implementation + "\n# second generation\n",
                        ),
                    ]
                )
                agent = ShinkaEvolve(
                    "task",
                    provider,
                    context=environment.instructions,
                    config=Config(islands=1, meta_interval=0, patch_types=(("full", 1),)),
                )
                await run_search(
                    agent,
                    run,
                    generations=2,
                    batch_size=1,
                    seeds=(0, 1),
                    console=Console(file=io.StringIO()),
                )
                self.assertEqual(len(run.policies()), 2)
                self.assertTrue(
                    all(
                        math.isfinite(value)
                        for policy in run.policies()
                        for value in run.scores(policy).values()
                    )
                )
                self.assertIn(name, provider.calls[0])
                with run.database() as db:
                    self.assertEqual(len(db.exec(select(Evaluation)).all()), 2)
                    generations = db.exec(select(Generation)).all()
                    self.assertTrue(
                        all(row.complete and row.seeds == [0, 1] for row in generations)
                    )
                    self.assertEqual(len(generations[-1].islands[0]), 2)

    async def test_alphaevolve_uses_native_environments_and_isolated_evaluation(self):
        import io

        import gymnasium as gym
        from rich.console import Console
        from slick import prompts

        import alphaevolve
        from alphaevolve.edits import Program
        from alphaevolve.improved import AlphaEvolve, Config
        from examples import cartpole
        from examples.alphaevolve import run_search
        from rsikit import Executor, Run
        from tests.providers import ScriptedProvider

        initial = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n"
        provider = ScriptedProvider(
            [
                Program(description="Test policy approach.", name="Left", implementation=initial),
                Program(
                    description="Test policy approach.",
                    name="Balance",
                    implementation=Path(cartpole.__file__).read_text(),
                ),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            gym.make("CartPole-v1", max_episode_steps=50) as environment,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            Run.create(
                name="evolution",
                environment=environment,
                path=Path(folder) / "run",
                executor=Executor(concurrency=2),
            ) as run,
        ):
            agent = AlphaEvolve(
                "Balance CartPole", provider, config=Config(mode="rewrite", islands=1)
            )
            output = io.StringIO()
            await run_search(
                agent,
                run,
                generations=2,
                batch_size=1,
                console=Console(file=output, width=120, force_terminal=False),
            )
            generation_scores = [run.scores(policy)[0] for policy in run.policies()]
            self.assertIn("Generated Left", output.getvalue())
            self.assertIn("score=50", output.getvalue())
            self.assertIn("Best so far: Balance", (run.path / "run.log").read_text())
            self.assertGreater(generation_scores[1], generation_scores[0])
            self.assertEqual(generation_scores[1], 50)
            self.assertEqual(agent.best.name, "Balance")
            self.assertEqual(len(run.policies()), 2)
            self.assertEqual(len(list((run.path / "exports").glob("*.py"))), 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn('Per-seed rewards: {"0":', provider.calls[1])

    async def test_repairs_syntax_and_constructor_failures_through_real_docker(self):
        import io

        from rich.console import Console
        from slick import prompts

        import alphaevolve
        from alphaevolve.edits import Program
        from alphaevolve.improved import AlphaEvolve
        from examples.alphaevolve import run_search
        from rsikit import Executor, Run
        from tests.providers import ScriptedProvider

        broken = """from rsikit import Policy
class Solution(Policy):
    def __init__(self, observation_space, action_space, instructions):
        super().__init__(observation_space, action_space, instructions)
    async def act(self, observation):
        return 0
"""
        fixed = """from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return 0
"""
        provider = ScriptedProvider(
            [
                Program(name="Malformed", description="Baseline.", implementation=broken + "}"),
                Program(name="Constructor error", description="Baseline.", implementation=broken),
                Program(
                    name="Repaired", description="Inherit the constructor.", implementation=fixed
                ),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=5) as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            Run.create(
                name="healing", path=Path(directory) / "run", environment=env, executor=Executor()
            ) as run,
        ):
            generator = AlphaEvolve("Balance CartPole", provider)
            output = io.StringIO()
            await run_search(
                generator,
                run,
                generations=1,
                batch_size=1,
                console=Console(file=output, force_terminal=False, width=120),
            )
            self.assertEqual(generator.repair_calls, 2)
            self.assertEqual(generator.completed, 1)
            self.assertEqual(generator.best.name, "Repaired")
            self.assertEqual(run.scores(generator.best), {0: 5.0})
            failed, repaired = run.policies()
            self.assertEqual(run.scores(failed), {0: None})
            self.assertEqual(repaired.id, generator.best.id)
            self.assertIn("unmatched", provider.calls[1])
            self.assertIn("positional arguments", provider.calls[2])
            self.assertIn("Repairing proposal 1 (2/2)", (run.path / "run.log").read_text())

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
                    "description": "Test policy approach.",
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
            with (
                gym.make("CartPole-v1", max_episode_steps=30) as env,
                Run.open(output, environment=env) as run,
            ):
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
                                "description": "Balance using pole angle and velocity.",
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
            env = gym.wrappers.RecordVideo(
                gym.make("CartPole-v1", max_episode_steps=5, render_mode="rgb_array"),
                str(Path(directory) / "recordings"),
                episode_trigger=lambda _: True,
            )
            with env, Run.create(name="video", path=output, environment=env) as run:
                self.assertEqual(await run.evaluate([policy], seeds=[1]), {policy.id: 5.0})
                videos = list((output / "artifacts" / policy.id / "1").rglob("*.mp4"))
                self.assertEqual(len(videos), 1)
                video = videos[0]
                self.assertGreater(video.stat().st_size, 100)
                from moviepy import VideoFileClip

                with VideoFileClip(str(video)) as clip:
                    self.assertGreater(clip.duration, 0)
                    self.assertEqual(clip.get_frame(0).shape[2], 3)
            with (
                gym.make("CartPole-v1", max_episode_steps=5) as env,
                Run.open(output, environment=env) as run,
            ):
                self.assertEqual(run.scores(run.policies()[0]), {1: 5.0})
                self.assertTrue(video.exists())

    async def test_replay_records_best_completed_policy_without_changing_original_scores(self):
        import io
        from importlib.util import find_spec

        from rich.console import Console

        from examples.replay import record_best
        from rsikit import Executor, Run
        from rsikit.policy import _policy_class
        from tests.test_run import RESPONSE, FakeSandbox

        if find_spec("moviepy") is None or find_spec("pygame") is None:
            self.skipTest("Install .[video] for video checks")
        policies = [
            _policy_class(name, RESPONSE["implementation"]) for name in ("Low", "Best", "Failed")
        ]
        sandbox = FakeSandbox()
        sandbox.evaluate.side_effect = [(2.0, {}), (30.0, {}), PolicyError("bad policy")]
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            output = Path(directory) / "replay"
            with (
                gym.make("CartPole-v1", max_episode_steps=3) as env,
                Run.create(
                    name="search",
                    path=original,
                    environment=env,
                    executor=Executor(sandbox=sandbox),
                ) as source,
            ):
                with self.assertRaises(PolicyError):
                    await source.evaluate(policies)
            env = gym.wrappers.RecordVideo(
                gym.make("CartPole-v1", max_episode_steps=3, render_mode="rgb_array"),
                str(Path(directory) / "recordings"),
                episode_trigger=lambda _: True,
                disable_logger=True,
            )
            with env:
                path = await record_best(
                    original,
                    env,
                    top=1,
                    seeds=[0],
                    output=output,
                    console=Console(file=io.StringIO()),
                )
            with (
                gym.make("CartPole-v1", max_episode_steps=3) as env,
                Run.open(original, environment=env) as source,
                Run.open(path, environment=env) as replay,
            ):
                self.assertEqual([source.scores(p)[0] for p in policies], [2.0, 30.0, None])
                self.assertEqual([p.id for p in replay.policies()], [policies[1].id])
                self.assertEqual(replay.scores(policies[1]), {0: 3.0})
                videos = list((path / "artifacts" / policies[1].id / "0").rglob("*.mp4"))
                self.assertEqual(len(videos), 1)
                self.assertGreater(videos[0].stat().st_size, 100)

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

    async def test_batch_shares_one_container_with_independent_environment_processes(self):
        import json

        from slick import prompts

        import rsikit.generation as generation
        from rsikit import DockerSandbox, Executor, Run, generate
        from tests.providers import ScriptedProvider

        class Environment(gym.Env):
            instructions = "independent environment"

            def __init__(self):
                self.observation_space = spaces.Discrete(3)
                self.action_space = spaces.Discrete(1)
                self.count = 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.count = 0
                return 0, {}

            def step(self, action):
                import os
                import socket
                import time

                time.sleep(0.1)
                self.count += 1
                return (
                    self.count,
                    1.0,
                    False,
                    False,
                    {"artifacts": {"worker.txt": f"{socket.gethostname()}:{os.getpid()}".encode()}},
                )

        response = json.dumps(
            {
                "name": "Single container",
                "description": "Check environment instructions in an isolated worker.",
                "implementation": (
                    "from rsikit import Policy\nclass Solution(Policy):\n"
                    "    async def act(self, observation):\n"
                    "        assert self.instructions == 'independent environment'\n"
                    "        return 0\n"
                ),
            }
        )
        with patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts"):
            policy = await generate("test", provider=ScriptedProvider([response]))
        sandbox = DockerSandbox()
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.wrappers.TimeLimit(Environment(), max_episode_steps=2) as env,
        ):
            with (
                patch.object(sandbox, "start", wraps=sandbox.start) as start,
                Run.create(
                    name="pool",
                    path=Path(directory) / "run",
                    environment=env,
                    executor=Executor(sandbox=sandbox, concurrency=2),
                ) as run,
            ):
                result = await run.evaluate([policy], seeds=[0, 1, 2, 3])
                self.assertEqual(result, {policy.id: 2.0})
                start.assert_awaited_once_with(2)
                workers = [p.read_text().split(":") for p in run.path.rglob("worker.txt")]
                self.assertEqual(len(workers), 4)
                self.assertEqual(len({host for host, pid in workers}), 1)
                self.assertEqual(len({pid for host, pid in workers}), 4)
                self.assertEqual(env.unwrapped.count, 0)
                self.assertIsNone(sandbox.name)

    async def test_batch_timeout_and_cancellation_remove_shared_container(self):
        from rsikit import DockerSandbox, Executor, Run
        from rsikit.policy import _policy_class

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        self.addCleanup(loop.set_exception_handler, previous_handler)
        loop_errors = []
        loop.set_exception_handler(lambda loop, context: loop_errors.append(context))
        policy = _policy_class(
            "stuck",
            "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        while True: pass\n",
        )
        sandbox = DockerSandbox()
        executor = Executor(sandbox=sandbox, call_timeout=0.1)
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1") as env,
            Run.create(
                name="cleanup",
                path=Path(directory) / "run",
                environment=env,
                executor=executor,
            ) as run,
        ):
            with self.assertRaises(PolicyTimeout):
                await run.evaluate([policy])
            self.assertEqual(run.scores(policy), {0: None})
            self.assertIsNone(sandbox.name)

            started, release = asyncio.Event(), asyncio.Event()
            create_process = asyncio.create_subprocess_exec
            interrupted_creation, processes = [], []

            async def delayed_create(*args, **kwargs):
                process = await create_process(*args, **kwargs)
                if args[:2] == ("docker", "exec"):
                    processes.append(process)
                    started.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        interrupted_creation.append(True)
                        raise
                return process

            executor.call_timeout = 60
            with patch("rsikit.sandbox.docker.asyncio.create_subprocess_exec", delayed_create):
                pending = asyncio.create_task(run.resume())
                await asyncio.wait_for(started.wait(), 10)
                name = sandbox.name
                pending.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
            for process in processes:
                await process.communicate()
            self.assertEqual(interrupted_creation, [])
            self.assertIsNone(sandbox.name)
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self.assertNotEqual(await inspect.wait(), 0)
            self.assertEqual(loop_errors, [])


if __name__ == "__main__":
    unittest.main()
