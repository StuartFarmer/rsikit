"""Search examples and real episode processes need no live model calls."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sqlmodel import select

from rsikit import Executor, Run
from tests.providers import ScriptedProvider
from tests.search_helpers import policy, proposal, templates
from tests.test_execution import ProcessEnv


class SearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_policy_searches_persist_real_measurements_and_events(self):
        for name, class_name in (
            ("evox", "EvoX"),
            ("gepa", "GEPA"),
            ("eoh", "EoH"),
            ("promptbreeder", "PromptBreeder"),
        ):
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(f"research.{name}")
                event_type = getattr(
                    importlib.import_module(f"research.{name}.records"), f"{class_name}Event"
                )
                with ProcessEnv() as env, templates(module):
                    async with (
                        Executor(concurrency=2) as executor,
                        Run.create(name=name, path=Path(directory) / "run") as run,
                    ):

                        async def evaluate(policies, *, seeds=(0, 1)):
                            return await run.evaluate(
                                policies, environment=env, executor=executor, seeds=seeds
                            )

                        def record(kind, data):
                            run.save(event_type(kind=kind, data=data))

                        if name == "evox":
                            from research.evox.runtime import run_python_strategy

                            provider = ScriptedProvider(
                                ['{"refine":"tune","diverge":"explore"}', proposal(1), proposal(2)]
                            )
                            agent = module.EvoX(
                                "task",
                                provider,
                                evaluate,
                                run_strategy=run_python_strategy,
                                config=module.Config(iterations=2, window=2),
                                on_event=record,
                            )
                        elif name == "gepa":
                            agent = module.GEPA(
                                "task",
                                ScriptedProvider([proposal(2)]),
                                evaluate,
                                run.load_episode,
                                initial_policy=policy(1),
                                train_seeds=[0],
                                selection_seeds=[10],
                                budget=4,
                                minibatch_size=1,
                                on_event=record,
                            )
                        elif name == "eoh":
                            agent = module.EoH(
                                "task",
                                ScriptedProvider([proposal(1), proposal(2)]),
                                population_size=1,
                                generations=1,
                                operators=("M3",),
                                on_event=record,
                            )
                        else:
                            agent = module.PromptBreeder(
                                "task",
                                ScriptedProvider([proposal(1), proposal(2)]),
                                population_size=2,
                                tournaments=0,
                                on_event=record,
                            )
                        panel = (10,) if name == "gepa" else (0, 1)
                        while policies := await agent.propose():
                            agent.update(await evaluate(policies, seeds=panel))
                        self.assertEqual(len(run.policies()), 2)
                        self.assertIsNotNone(agent.best)
                        self.assertEqual(set(run.scores(agent.best).values()), {2.0})
                        with run.database() as db:
                            self.assertGreater(len(db.exec(select(event_type)).all()), 0)

    def test_all_six_examples_have_offline_help(self):
        for name in ("reevo", "evox", "gepa", "promptbreeder", "eoh", "stop_optimizer"):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, "-m", f"examples.{name}", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)

    async def test_malformed_metadata_is_rejected_by_every_policy_port(self):
        from tests.search_helpers import measure

        for name, class_name in (
            ("reevo", "ReEvo"),
            ("evox", "EvoX"),
            ("gepa", "GEPA"),
            ("eoh", "EoH"),
            ("promptbreeder", "PromptBreeder"),
        ):
            with self.subTest(name=name):
                module = importlib.import_module(f"research.{name}")
                bad = json.loads(proposal(1))
                bad["implementation"] = "# rsikit-policy: {}\n" + bad["implementation"]
                responses = [json.dumps(bad), proposal(2)]
                if name == "evox":
                    responses.insert(0, '{"refine":"tune","diverge":"explore"}')
                provider = ScriptedProvider(responses)
                with templates(module):
                    if name == "reevo":
                        agent = module.ReEvo(
                            "task",
                            provider,
                            measure,
                            config=module.Config(initial_size=2, max_evaluations=2),
                        )
                        await agent.run()
                    elif name == "evox":
                        from research.evox.runtime import run_python_strategy

                        agent = module.EvoX(
                            "task",
                            provider,
                            measure,
                            run_strategy=run_python_strategy,
                            config=module.Config(iterations=2, window=2),
                        )
                        await agent.run()
                    elif name == "gepa":
                        from rsikit import Episode

                        agent = module.GEPA("task", provider, measure, lambda *_: Episode())
                        await agent.run(
                            policy(1),
                            train_seeds=[0],
                            selection_seeds=[10],
                            budget=5,
                            minibatch_size=1,
                        )
                    elif name == "eoh":
                        agent = module.EoH("task", provider, measure)
                        await agent.run(population_size=1, generations=0)
                    else:
                        agent = module.PromptBreeder("task", provider, measure)
                        await agent.run(population_size=1, tournaments=0)
                self.assertEqual(agent.best.id, policy(2).id)
