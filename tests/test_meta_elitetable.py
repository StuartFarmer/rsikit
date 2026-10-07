"""The trusted evaluator, not generated code, owns the five-generation protocol."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from slick import prompts

from research.meta_ocean.elitetable import (
    Config,
    EvolverTrial,
    _campaign,
    comparison_scorecard,
    workload,
)
from research.ocean.baselines import policies
from research.ocean.environment import MAZE_CONTEXT
from research.providers import RESPONSE, BudgetProvider
from rsikit.progress import bind_run


def model(provider, calls=100):
    return BudgetProvider(
        provider,
        max_calls=calls,
        max_tokens=10_000_000,
        spend_cap=100,
        max_input_tokens=65536,
        max_output_tokens=16384,
        input_price=1,
        output_price=1,
    )


EVOLVER = """from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        variant = 0
        assert observation["generations"] == 5
        replies = await self.generate(["test inner model gateway"])
        assert replies[0]["text"] == "ack"
        if observation["generation"] > 1:
            assert observation["results"][0]["source"] == observation["starter"]
            assert "score" in observation["results"][0]
        return [observation["starter"]]
"""


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_report_tracks_busy_queued_and_cancelled_slots_in_rich(self):
        from research.meta_ocean.elitetable import WorkerPool, report_workers
        from research.meta_ocean.experiment import progress

        slots = {name: WorkerPool(1) for name in ("trials", "evaluations", "models")}
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=160, height=35)
        with (
            tempfile.TemporaryDirectory() as directory,
            bind_run(Path(directory), console=console) as display,
        ):
            progress("Starting", kind="search_started", optimizer="Meta", total_candidates=20)
            await slots["evaluations"].acquire()
            queued = asyncio.create_task(slots["evaluations"].acquire())
            report = asyncio.create_task(report_workers(slots))
            try:
                await asyncio.sleep(0.35)
                self.assertEqual(display.model.workers["Panels"]["active"], 1)
                self.assertEqual(display.model.workers["Panels"]["queued"], 1)
                self.assertIn("Panels 1/1", output.getvalue())
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
                slots["evaluations"].release()
                await asyncio.sleep(1.1)
                self.assertEqual(
                    display.model.workers["Panels"], dict(active=0, limit=1, queued=0, finished=1)
                )
                self.assertIn("Panels 0/1", output.getvalue())
                self.assertFalse(display.model.batches)
            finally:
                queued.cancel()
                report.cancel()
                await asyncio.gather(queued, report, return_exceptions=True)

    async def test_five_generations_with_concurrent_proposals_errors_and_hard_caps(self):
        active = peak = 0

        async def evaluate(source, seeds):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return dict(score=10, scores=[10] * 10, steps=10)

        source = policies()[0].source
        with tempfile.TemporaryDirectory() as directory:
            trial = EvolverTrial(
                Config(generation_size=2, inner_max_repairs=0),
                Path(directory),
                source,
                evaluate,
                None,
                0,
                model_slots=asyncio.Semaphore(2),
            )
            rejected = await trial.handle(
                dict(op="population", value=dict(generation=1, policies=[source] * 3))
            )
            self.assertIn("error", rejected)
            self.assertEqual(trial.evaluations, 0)
            for generation in range(1, 6):
                result = await trial.handle(
                    dict(
                        op="population",
                        value=dict(
                            generation=generation,
                            policies=[source, "invalid !"] if generation == 1 else [source, source],
                        ),
                    )
                )
                self.assertEqual(len(result["results"]), 2)
                if generation == 1:
                    self.assertEqual(result["results"][1]["source"], "invalid !")
                    self.assertIn("error", result["results"][1])
                duplicate = await trial.handle(
                    dict(op="population", value=dict(generation=generation, policies=[source]))
                )
                self.assertIn("error", duplicate)
            self.assertEqual(trial.generations, 5)
            self.assertEqual(trial.evaluations, 10)
            self.assertEqual(trial.failures, 1)
            self.assertEqual(peak, 2)
            self.assertEqual(trial.incumbent, source)
            self.assertIn("error", await trial.handle(dict(op="evaluate", value=source)))

    async def test_batched_model_calls_are_capped_per_generation_and_preserve_order(self):
        received = []

        class Model:
            async def acall(self, prompt):
                received.append(prompt)
                await asyncio.sleep(0)
                return prompt.rsplit("\n\nSub-evolver request:\n", 1)[-1] + " response", []

        source = policies()[0].source

        async def evaluate(source, seeds):
            return dict(score=0, scores=[0] * 10, steps=10)

        with tempfile.TemporaryDirectory() as directory:
            trial = EvolverTrial(
                Config(generation_size=2),
                Path(directory),
                source,
                evaluate,
                Model(),
                0,
                model_slots=asyncio.Semaphore(2),
            )
            trial.task = MAZE_CONTEXT
            result = await trial.handle(dict(op="generate", value=["first", "second"]))
            self.assertTrue(all(MAZE_CONTEXT in prompt for prompt in received))
            self.assertEqual(
                [r["text"] for r in result["responses"]], ["first response", "second response"]
            )
            self.assertIn("error", await trial.handle(dict(op="generate", value=["third"])))
            await trial.handle(dict(op="population", value=dict(generation=1, policies=[source])))
            self.assertIn(
                "responses", await trial.handle(dict(op="generate", value=["new generation"]))
            )

    async def test_outer_writer_receives_all_task_specifications(self):
        from research.meta_ocean import elitetable
        from research.ocean.environment import BREAKOUT_CONTEXT, CONTEXT

        class Captured(Exception):
            pass

        def capture(*args, context, **kwargs):
            for task in (CONTEXT, BREAKOUT_CONTEXT, MAZE_CONTEXT):
                self.assertIn(task, context)
            raise Captured

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(elitetable, "provider", return_value=None),
            patch.object(elitetable, "MetaSearch", side_effect=capture),
        ):
            with self.assertRaises(Captured):
                await _campaign(Config(), Path(directory), {})

    async def test_host_enforces_five_repairs_without_optimizer_cooperation(self):
        source = policies()[0].source
        failed = source + "\n# broken_marker\n"

        class Model:
            calls = 0

            async def acall(self, prompt):
                self.calls += 1
                assert "shape mismatch" in prompt
                assert "broken_marker" in prompt
                repaired = source if self.calls == 5 else failed
                return json.dumps(
                    dict(
                        name="Repair",
                        description="Fix actual error",
                        implementation=f"```python\n{repaired}\n```",
                    )
                ), []

        async def evaluate(candidate, seeds):
            if "broken_marker" in candidate:
                raise ValueError("shape mismatch")
            return dict(score=10, scores=[10] * 10, steps=10)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/elitesearch/prompts")),
        ):
            model = Model()
            trial = EvolverTrial(
                Config(generation_size=1),
                Path(directory),
                source,
                evaluate,
                model,
                0,
                model_slots=asyncio.Semaphore(1),
            )
            trial.generations = 4  # Repairs must work even in the final generation.
            response = await trial.handle(
                dict(op="population", value=dict(generation=5, policies=[failed]))
            )
            feedback = response["results"][0]
            self.assertEqual(feedback["score"], 10)
            self.assertEqual(feedback["repairs"], 5)
            self.assertEqual(feedback["submitted_source"], failed)
            self.assertNotIn("broken_marker", feedback["source"])
            self.assertEqual(model.calls, 5)
            self.assertEqual(trial.evaluations, 6)
            self.assertEqual(trial.failures, 5)
            self.assertEqual(trial.incumbent, feedback["source"])

    async def test_syntax_failures_stop_after_five_unsuccessful_repairs(self):
        class Model:
            calls = 0

            async def acall(self, prompt):
                self.calls += 1
                return json.dumps(
                    dict(name="Still broken", description="Invalid", implementation="invalid !")
                ), []

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/elitesearch/prompts")),
        ):
            model = Model()
            trial = EvolverTrial(
                Config(generation_size=1),
                Path(directory),
                policies()[0].source,
                None,
                model,
                0,
                model_slots=asyncio.Semaphore(1),
            )
            result = await trial.handle(
                dict(op="population", value=dict(generation=1, policies=["invalid !"]))
            )
            self.assertIn("error", result["results"][0])
            self.assertEqual(result["results"][0]["repairs"], 5)
            self.assertEqual(model.calls, 5)
            self.assertEqual(trial.evaluations, 6)
            self.assertEqual(trial.failures, 6)

    def test_outer_and_inner_budgets_are_separate(self):
        config = Config()
        self.assertEqual(
            (
                config.population,
                config.generations,
                config.inner_generations,
                config.generation_size,
            ),
            (10, 5, 5, 50),
        )
        self.assertEqual(config.evaluations, 1500)
        self.assertEqual(workload(config)["outer_candidates"], 50)
        self.assertEqual(workload(config)["max_policies_per_trial"], 250)
        with self.assertRaises(ValueError):
            Config(generation_size=51)
        with self.assertRaises(ValueError):
            Config(inner_generations=6)


class DisplayTests(unittest.TestCase):
    def test_setup_releases_terminal_to_the_elitetable_leaderboard(self):
        from types import SimpleNamespace

        from research.meta_ocean import elitetable, runner
        from research.meta_ocean.experiment import progress
        from rsikit.progress.controller import _current_run

        async def calibrate(config, path):
            self.fail("EliteTable must not run heuristic calibration")

        def campaign(config, path, anchors, loop):
            self.assertIsNone(_current_run.get().live)
            path.mkdir()
            with bind_run(path):
                progress(
                    "EliteTable",
                    kind="search_started",
                    total_candidates=50,
                    optimizer="EliteTable meta",
                )
                self.assertIsNotNone(_current_run.get().live)
                progress("done", kind="search_finished", status="completed", reason="test")
            return {}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, OPENROUTER_API_KEY="test"),
            patch(
                "rsikit.progress.controller.Console",
                return_value=Console(file=io.StringIO(), force_terminal=True),
            ),
            patch.object(runner, "metadata", return_value={}),
            patch.object(
                runner.subprocess, "run", return_value=SimpleNamespace(stdout="sha256:test")
            ),
            patch.object(runner, "calibrate", side_effect=calibrate),
            patch.object(elitetable, "campaign", side_effect=campaign),
        ):
            result = runner.run(Config(environments=["g2048"]), Path(directory) / "run")
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("anchors", result)


class ComparisonTests(unittest.TestCase):
    def test_improvement_is_zero_at_baseline_and_independent_of_game_units(self):
        baseline = [
            dict(environment=name, replicate=0, score=score, output_tokens=100, evaluations=10)
            for name, score in (("g2048", 1000), ("maze", 0.25))
        ]
        candidate = [dict(row, score=row["score"] * 2) for row in baseline]
        self.assertEqual(comparison_scorecard(baseline, baseline)["S"], 0)
        self.assertAlmostEqual(comparison_scorecard(candidate, baseline)["S"], 200 / 3)
        candidate[0]["score"] *= 10000
        baseline[0]["score"] *= 10000
        self.assertAlmostEqual(comparison_scorecard(candidate, baseline)["S"], 200 / 3)
        candidate = [dict(row, score=row["score"] / 2) for row in baseline]
        self.assertLess(comparison_scorecard(candidate, baseline)["S"], 0)
        for row in baseline:
            row["score"] = 0
        self.assertEqual(comparison_scorecard(baseline, baseline)["S"], 0)
        self.assertEqual(comparison_scorecard(candidate, baseline)["S"], 200)
        with self.assertRaises(ValueError):
            comparison_scorecard(candidate[:1], baseline)
        baseline[0]["audit_error"] = "timeout"
        with self.assertRaises(ValueError):
            comparison_scorecard(candidate, baseline)

    def test_workload_replaces_calibration_with_one_baseline_per_split(self):
        result = workload(Config())
        self.assertEqual(result["calibration_games"], 0)
        self.assertEqual(result["fixed_baseline_trials"], 30)
        self.assertEqual(result["max_controller_trials"], 1854)
        self.assertNotIn("calibration_cases", Config().model_dump())


@unittest.skipUnless(os.environ.get("RSIKIT_META_DOCKER_TESTS"), "requires rebuilt meta worker")
class EliteIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_elitetable_writes_mutates_and_scores_five_generation_evolvers(self):
        import re

        from research.meta_ocean import elitetable

        baseline_runs = []

        class BaselineModel:
            def __init__(self):
                self.calls = 0

            async def acall(self, prompt):
                self.calls += 1
                RESPONSE.get().update(
                    usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0
                )
                assert "Write a sub-evolver" not in prompt
                if "Return exact search/replacement edits" in prompt:
                    old = re.search(r"variant = (\d+)", prompt).group(0)
                    response = dict(
                        name=f"baseline-{self.calls}",
                        description="Fixed EliteTable edit",
                        edits=[dict(search=old, replacement=f"variant = {1000 + self.calls}")],
                    )
                else:
                    source = f"""import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        variant = {self.calls}
        return np.zeros(len(observation), dtype=np.int64)
"""
                    response = dict(
                        name=f"baseline-{self.calls}",
                        description="Fixed EliteTable invention",
                        implementation="invalid !" if self.calls == 1 else source,
                    )
                return json.dumps(response), []

        class Writer:
            async def acall(self, prompt):
                event = RESPONSE.get()
                event.update(usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0)
                self.assert_contract(prompt)
                if "Return exact search/replacement edits" in prompt:
                    response = dict(
                        name="Mutated evolver",
                        description="Changed search parameter",
                        edits=[dict(search="variant = 0", replacement="variant = 1")],
                    )
                elif "Repair this organism" in prompt:
                    assert "broken evolver" in prompt
                    response = dict(
                        name="Repaired evolver",
                        description="Fixed actual execution failure",
                        implementation=EVOLVER,
                    )
                else:
                    response = dict(
                        name="Written evolver",
                        description="Carries a population across generations",
                        implementation=EVOLVER.replace(
                            'return [observation["starter"]]', 'raise ValueError("broken evolver")'
                        ),
                    )
                return json.dumps(response), []

            def assert_contract(self, prompt):
                assert "Write a sub-evolver" in prompt
                assert (
                    "Implement async act(self, observation)\nreturning a finite action"
                    not in prompt
                )

        def provider(config, path, *, editor=False):
            if editor:
                return model(Writer())
            if "baseline" in path.parts:
                baseline_runs.append(path)
                return model(BaselineModel())

            class InnerModel:
                async def acall(self, prompt):
                    assert prompt.endswith("test inner model gateway")
                    assert "Observation is uint8" in prompt
                    RESPONSE.get().update(
                        usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0
                    )
                    return "ack", []

            return model(InnerModel())

        config = Config(
            objective="performance",
            population=1,
            generations=2,
            elites=1,
            generation_size=2,
            environments=["g2048", "maze"],
            development=[0],
            validation=[1],
            test=[2],
            max_steps=2,
            audit_cases=1,
            final_audit_cases=2,
            max_repairs=1,
            trial_workers=2,
            evaluation_workers=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (
                patch.object(elitetable, "provider", provider),
                bind_run(path, console=Console(file=io.StringIO())) as display,
            ):
                result = await _campaign(
                    config,
                    path,
                    {},
                )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["editor"]["calls"], 3)
                leaderboard = json.loads((path / "leaderboard.json").read_text())
                self.assertEqual(leaderboard[0]["name"], "Mutated evolver")
                self.assertIn("metrics", leaderboard[0])
                self.assertEqual(len(display.model.leaders), 1)
                self.assertEqual(
                    len(baseline_runs), 6
                )  # Reused across all outer proposals/repairs.
                for baseline_path in baseline_runs:
                    row = json.loads((baseline_path.parent / "summary.json").read_text())
                    self.assertEqual(row["generations"], 5)
                    self.assertEqual(row["evaluations"], 11)
                    self.assertEqual(row["calls"], 11)
                    self.assertEqual(row["failures"], 1)
                    self.assertIsNone(row["error"])
                for phase in ("validation", "test"):
                    for row in result["heldout"][phase]["trials"]:
                        self.assertEqual(row["generations"], 5)
                        self.assertEqual(row["evaluations"], 5)
                        self.assertEqual(row["calls"], 5)
                        self.assertEqual(row["completed_transitions"], 100)
                        self.assertEqual(row["audit_transitions"], 4 if phase == "test" else 2)
                        base_path = (
                            path / "baseline" / phase / row["environment"] / str(row["replicate"])
                        )
                        candidate_path = path / phase / row["environment"] / str(row["replicate"])
                        base_audit = json.loads((base_path / "audit.json").read_text())
                        candidate_audit = json.loads((candidate_path / "audit.json").read_text())
                        self.assertEqual(
                            [r["seed"] for r in base_audit["results"]],
                            [r["seed"] for r in candidate_audit["results"]],
                        )
                    self.assertEqual(result["heldout"][phase]["baseline"], "fixed_elitetable")
                self.assertNotIn("Invalid progress record", (path / "run.log").read_text())

                # A completed campaign resumes without any new provider calls or evaluations.
                async def forbidden(*args, **kwargs):
                    self.fail("Completed work must be reused")

                with (
                    patch.object(BaselineModel, "acall", forbidden),
                    patch.object(Writer, "acall", forbidden),
                ):
                    resumed = await _campaign(config, path, {})
                self.assertEqual(resumed["status"], "completed")
                self.assertEqual(len(baseline_runs), 6)

    async def test_real_shape_error_is_repaired_even_when_evolver_ignores_feedback(self):
        from research.meta_ocean.elitetable import run_trial

        broken = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.zeros((len(observation), 2)) + np.zeros((len(observation), 3))
"""
        evolver = f"""from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return [{broken!r}]
"""
        source = policies()[0].source

        class RepairModel:
            async def acall(self, prompt):
                assert "operands could not be broadcast" in prompt
                assert broken in prompt
                RESPONSE.get().update(
                    usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0
                )
                return json.dumps(
                    dict(
                        name="Fixed shape",
                        description="Repaired runtime failure",
                        implementation=source,
                    )
                ), []

        config = Config(generation_size=1, environments=["g2048"], max_steps=1, audit_cases=1)
        slots = {key: asyncio.Semaphore(2) for key in ("trials", "models", "evaluations")}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/elitesearch/prompts")),
        ):
            path = Path(directory)
            with bind_run(path, console=Console(file=io.StringIO())):
                result = await run_trial(
                    config, evolver, path / "trial", 0, "g2048", slots, model=model(RepairModel())
                )
            self.assertIsNone(result["error"])
            self.assertIsNone(result["audit_error"])
            self.assertEqual(result["calls"], 5)
            self.assertEqual(result["evaluations"], 10)
            self.assertEqual(result["failures"], 5)
            for f in (path / "trial").glob("generation-*.json"):
                feedback = json.loads(f.read_text())[0]
                self.assertEqual(feedback["repairs"], 1)
                self.assertEqual(feedback["source"], source)
                self.assertIn("score", feedback)


if __name__ == "__main__":
    unittest.main()
