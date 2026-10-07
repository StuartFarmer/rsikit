"""Saved-value compatibility and full application episode checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces

from research.rewards import mean_rewards
from rsikit import Executor, Job, PolicyDefinition
from rsikit.episode import Episode
from rsikit.evaluation import InfrastructureError, PolicyError
from rsikit.policy import Policy
from tests.helpers import fake_executor, finish_pending, recorded_run, run_episode

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
    def test_legacy_json_rejects_unsafe_allocations(self):
        for encoded in (
            ["array", "O", [1], "AAAAAAAAAAA="],
            ["array", "f8", [1_000_000], ""],
            ["array", "f8", [1], ""],
        ):
            with self.assertRaises((ValueError, TypeError)):
                Episode.from_data(
                    dict(
                        observations=["list", [encoded]],
                        actions=["list", []],
                        rewards=["list", []],
                        terminations=["list", []],
                        truncations=["list", []],
                        infos=["list", [["dict", []]]],
                        artifacts=["dict", []],
                        error="failed",
                    )
                )


class ApplicationEpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_source(self, source, *, timeout=3, raw=False):
        policy = PolicyDefinition.from_text(source)
        with CounterEnv() as environment:
            async with Executor(episode_timeout=timeout) as executor:
                results = [
                    job.result
                    async for job in executor.iterate(
                        [Job(policy, environment, seed=1, max_steps=5)]
                    )
                ]
        (episode,) = results
        return episode if raw else episode.final_step

    async def test_state_instructions_and_examples(self):
        local = await run_episode(CounterEnv, CounterPolicy, seed=1, max_steps=5)
        for _ in range(2):
            isolated = await self.run_source(COUNTER_SOURCE)
            self.assertEqual(isolated[:4], local[:4])
            self.assertEqual(isolated[4]["episode"]["r"], 3.0)
            self.assertEqual(isolated[4]["episode"]["l"], 2)
        from examples import cartpole
        from examples.circle_packing import initial
        from rsikit.envs import CirclePackingEnv

        episodes = []
        for path, make_env, limit in (
            (initial.__file__, CirclePackingEnv, 5),
            (cartpole.__file__, lambda: gym.make("CartPole-v1"), 8),
        ):
            policy = PolicyDefinition.from_file(path)
            with make_env() as environment:
                async with Executor() as executor:
                    episodes.extend(
                        [
                            job.result
                            async for job in executor.iterate(
                                [Job(policy, environment, seed=1, max_steps=limit)]
                            )
                        ]
                    )
        packing, cart = episodes
        self.assertTrue(packing.terminations[-1])
        self.assertAlmostEqual(packing.infos[-1]["episode"]["r"], 1.0)
        self.assertTrue(cart.terminations[-1] or cart.truncations[-1])
        self.assertGreater(cart.infos[-1]["episode"]["l"], 1)

    async def test_scientific_libraries_in_episode_process(self):
        source = COUNTER_SOURCE.replace(
            "        self.count = 0 if",
            "        from tests.test_scientific_libraries import check\n"
            "        check()\n"
            "        self.count = 0 if",
        )
        result = await self.run_source(source, timeout=10)
        self.assertEqual(result[4]["episode"]["r"], 3.0)

    async def test_box2d_examples_preserve_instructions_and_evaluate_multiple_seeds(self):
        import importlib.util
        import io
        import math

        import cloudpickle
        from rich.console import Console
        from slick import prompts

        from examples.alphaevolve import run_search
        from research import alphaevolve
        from research.alphaevolve.generation import _PolicyResponse
        from research.alphaevolve.improved import AlphaEvolve
        from rsikit.envs.tasks import make_environment
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
                ):
                    async with recorded_run(
                        name="box2d",
                        path=Path(directory) / "run",
                        environment=restored,
                        console=Console(file=io.StringIO()),
                    ) as (run, rollouts):
                        with patch.object(
                            prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent
                        ):
                            self.assertEqual(restored.instructions, env.instructions)
                            self.assertEqual(restored.spec.max_episode_steps, 3)
                            provider = ScriptedProvider(
                                [
                                    _PolicyResponse(
                                        name="Random motors",
                                        description="Exercise continuous actions and environment instructions.",
                                        implementation=f'from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        assert {name!r} in self.instructions\n        assert "truncated after 3 steps" in self.instructions\n        assert self.action_space.shape == ({actions},)\n        return self.action_space.sample()\n',
                                    )
                                ]
                            )
                            agent = AlphaEvolve(
                                "Maximize reward", provider, context=restored.instructions
                            )
                            await run_search(
                                agent,
                                run,
                                rollouts,
                                generations=1,
                                batch_size=1,
                                seeds=[0, 1],
                            )
                            self.assertIn(restored.instructions, provider.calls[0])
                            scores = run.scores(agent.best)
                            self.assertEqual(set(scores), {0, 1})
                            self.assertTrue(
                                all((math.isfinite(score) for score in scores.values()))
                            )
                            self.assertEqual(agent.completed, 1)
                            self.assertEqual(agent._best.seed_scores, scores)

    async def test_shinkaevolve_runs_all_presets_and_persists_islands(self):
        import io
        import math

        from rich.console import Console
        from slick import prompts
        from sqlmodel import select

        from examples.shinkaevolve import run_search
        from research import shinkaevolve
        from research.alphaevolve.generation import _PolicyResponse
        from research.shinkaevolve import Config, Evaluation, Generation, ShinkaEvolve
        from rsikit import Executor
        from rsikit.envs.tasks import make_environment
        from tests.providers import ScriptedProvider

        try:
            import Box2D  # noqa: F401 — availability check
        except ImportError:
            self.skipTest("Install the box2d extra to test Box2D environments")
        implementation = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return self.action_space.sample()\n"
        for name in ("CartPole-v1", "LunarLander-v3", "BipedalWalker-v3"):
            with (
                self.subTest(environment=name),
                tempfile.TemporaryDirectory() as directory,
                make_environment(name, max_steps=3) as environment,
                patch.object(
                    prompts, "TEMPLATE_ROOT", Path(shinkaevolve.__file__).parent / "prompts"
                ),
            ):
                async with recorded_run(
                    name="shinka-smoke",
                    path=Path(directory) / "run",
                    environment=environment,
                    executor=Executor(concurrency=2),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts):
                    provider = ScriptedProvider(
                        [
                            _PolicyResponse(
                                name="RandomPolicy",
                                description="Sample a valid action.",
                                implementation=implementation,
                            ),
                            _PolicyResponse(
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
                        agent, run, rollouts, generations=2, batch_size=1, seeds=(0, 1)
                    )
                    self.assertEqual(len(run.policies()), 2)
                    self.assertTrue(
                        all(
                            (
                                math.isfinite(value)
                                for policy in run.policies()
                                for value in run.scores(policy).values()
                            )
                        )
                    )
                    self.assertIn(name, provider.calls[0])
                    with run.database() as db:
                        self.assertEqual(len(db.exec(select(Evaluation)).all()), 2)
                        generations = db.exec(select(Generation)).all()
                        self.assertTrue(
                            all((row.complete and row.seeds == [0, 1] for row in generations))
                        )
                        self.assertEqual(len(generations[-1].islands[0]), 2)

    async def test_alphaevolve_uses_native_environments_and_isolated_evaluation(self):
        import io

        import gymnasium as gym
        from rich.console import Console
        from slick import prompts

        from examples import cartpole
        from examples.alphaevolve import run_search
        from research import alphaevolve
        from research.alphaevolve.generation import _PolicyResponse
        from research.alphaevolve.improved import AlphaEvolve, Config
        from rsikit import Executor
        from tests.providers import ScriptedProvider

        initial = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n"
        provider = ScriptedProvider(
            [
                _PolicyResponse(
                    description="Test policy approach.", name="Left", implementation=initial
                ),
                _PolicyResponse(
                    description="Test policy approach.",
                    name="Balance",
                    implementation=Path(cartpole.__file__).read_text(),
                ),
            ]
        )
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as folder,
            gym.make("CartPole-v1", max_episode_steps=50) as environment,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
        ):
            async with recorded_run(
                name="evolution",
                environment=environment,
                path=Path(folder) / "run",
                executor=Executor(concurrency=2),
                console=Console(file=output, width=120, force_terminal=False),
            ) as (run, rollouts):
                agent = AlphaEvolve(
                    "Balance CartPole", provider, config=Config(mode="rewrite", islands=1)
                )
                await run_search(agent, run, rollouts, generations=2, batch_size=1)
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

    async def test_repairs_syntax_and_constructor_failures_through_real_execution(self):
        import io

        from rich.console import Console
        from slick import prompts

        from examples.alphaevolve import run_search
        from research import alphaevolve
        from research.alphaevolve.generation import _PolicyResponse
        from research.alphaevolve.improved import AlphaEvolve
        from rsikit import Executor
        from tests.providers import ScriptedProvider

        broken = "from rsikit import Policy\nclass Solution(Policy):\n    def __init__(self, observation_space, action_space, instructions):\n        super().__init__(observation_space, action_space, instructions)\n    async def act(self, observation):\n        return 0\n"
        fixed = "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n"
        provider = ScriptedProvider(
            [
                _PolicyResponse(
                    name="Malformed", description="Baseline.", implementation=broken + "}"
                ),
                _PolicyResponse(
                    name="Constructor error", description="Baseline.", implementation=broken
                ),
                _PolicyResponse(
                    name="Repaired", description="Inherit the constructor.", implementation=fixed
                ),
            ]
        )
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=5) as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
        ):
            async with recorded_run(
                name="healing",
                path=Path(directory) / "run",
                environment=env,
                executor=Executor(),
                console=Console(file=output, force_terminal=False, width=120),
            ) as (run, rollouts):
                generator = AlphaEvolve("Balance CartPole", provider)
                await run_search(generator, run, rollouts, generations=1, batch_size=1)
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
            with gym.make("CartPole-v1", max_episode_steps=30) as env:
                async with recorded_run(output, environment=env) as (run, rollouts):
                    self.assertEqual(len(run.policies()), 5)
                    self.assertEqual(sum(len(run.scores(p)) for p in run.policies()), 10)
                    with self.assertRaises(PolicyError):
                        await finish_pending(rollouts)

    async def test_run_records_real_video_artifact(self):
        import json
        from importlib.util import find_spec

        from slick import prompts

        import rsikit.generation as generation
        from rsikit import generate
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
            with env:
                async with recorded_run(name="video", path=output, environment=env) as (
                    run,
                    rollouts,
                ):
                    self.assertEqual(
                        await mean_rewards(rollouts, [policy], seeds=[1]), {policy.id: 5.0}
                    )
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
                recorded_run(output, environment=env) as (run, rollouts),
            ):
                self.assertEqual(run.scores(run.policies()[0]), {1: 5.0})
                self.assertTrue(video.exists())

    async def test_replay_records_best_completed_policy_without_changing_original_scores(self):
        import io
        from importlib.util import find_spec

        from rich.console import Console

        from examples.replay import record_best
        from rsikit.policy import PolicyDefinition
        from tests.test_episode_storage import trajectory
        from tests.test_run import RESPONSE, FakeEvaluation

        if find_spec("moviepy") is None or find_spec("pygame") is None:
            self.skipTest("Install .[video] for video checks")
        policies = [
            PolicyDefinition.from_text(RESPONSE["implementation"], name=name)
            for name in ("Low", "Best", "Failed")
        ]
        evaluation = FakeEvaluation()
        evaluation.evaluate.side_effect = [
            trajectory(2.0),
            trajectory(30.0),
            PolicyError("bad policy"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            output = Path(directory) / "replay"
            with (
                gym.make("CartPole-v1", max_episode_steps=3) as env,
                recorded_run(
                    name="search",
                    path=original,
                    environment=env,
                    executor=fake_executor(evaluation=evaluation),
                ) as (source, source_rollouts),
            ):
                with self.assertRaises(PolicyError):
                    await mean_rewards(source_rollouts, policies)
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
                recorded_run(original, environment=env) as (source, source_rollouts),
                recorded_run(path, environment=env) as (replay, replay_rollouts),
            ):
                self.assertEqual([source.scores(p)[0] for p in policies], [2.0, 30.0, None])
                self.assertEqual([p.id for p in replay.policies()], [policies[1].id])
                self.assertEqual(replay.scores(policies[1]), {0: 3.0})
                videos = list((path / "artifacts" / policies[1].id / "0").rglob("*.mp4"))
                self.assertEqual(len(videos), 1)
                self.assertGreater(videos[0].stat().st_size, 100)

    async def test_failure_deadline_and_cleanup(self):
        for body in ("raise RuntimeError('candidate failure')", "return object()"):
            episode = await self.run_source(COUNTER_SOURCE.replace("return action", body), raw=True)
            self.assertIsNotNone(episode.error)
        with self.assertRaises(InfrastructureError):
            await self.run_source(COUNTER_SOURCE.replace("return action", "os._exit(3)"))
        timeout_source = COUNTER_SOURCE.replace("return action", "while True: pass")
        episode = await self.run_source(timeout_source, timeout=0.5, raw=True)
        self.assertIn("timeout", episode.error)


if __name__ == "__main__":
    unittest.main()
