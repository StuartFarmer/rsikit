"""Meta-search accounting: proposals, games and aggregate ratios are distinct."""

import asyncio
import io
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from research.meta_ocean.experiment import Config, Trial, scorecard, suite_scorecard
from research.ocean.baselines import policies


class MetaProgressTests(unittest.TestCase):
    def test_rich_starts_before_setup_and_reports_live_calibration(self):
        from types import SimpleNamespace

        from rich.console import Console

        from research.meta_ocean import runner
        from rsikit.progress import _current_run

        terminal = io.StringIO()
        console = Console(file=terminal, force_terminal=True, width=120, height=35)
        displays = []

        def metadata(name):
            binding = _current_run.get()
            self.assertIsNotNone(binding)
            self.assertTrue(binding["display"].active)
            displays.append(binding["display"])
            return {}

        async def rollout(source, seeds, *args, **kwargs):
            score = 10 if source == policies()[0].source else 20
            rows = [dict(score=score, steps=1) for _ in seeds]
            kwargs["_on_batch"](rows[:1])
            self.assertIn("1/2 games", (output / "run.log").read_text())
            self.assertTrue(
                any(r["status"] == "evaluating" for r in displays[0].candidates.values())
            )
            kwargs["_on_batch"](rows)
            return dict(results=rows, steps=len(rows))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict(os.environ, OPENROUTER_API_KEY="test"),
                patch("rsikit.progress.Console", return_value=console),
                patch.object(runner, "metadata", side_effect=metadata),
                patch.object(
                    runner.subprocess, "run", return_value=SimpleNamespace(stdout="sha256:test")
                ),
                patch.object(runner, "version", return_value="test"),
                patch.object(runner, "rollout", side_effect=rollout),
                patch.object(runner, "campaign", return_value={}),
            ):
                result = runner.run(
                    Config(objective="performance", environments=["g2048"], calibration_cases=2),
                    output,
                )
            log = (output / "run.log").read_text()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(displays[0].status, "completed")
            self.assertIn("baseline", log)
            self.assertIn("reference", log)
            self.assertIn("2/2 games", log)
            self.assertNotIn("Invalid progress record", log)
            self.assertIn("GEPA", terminal.getvalue())


class MetaAccountingTests(unittest.TestCase):
    def test_suite_normalizes_before_averaging_with_equal_environment_weights(self):
        rows = [dict(environment="g2048", score=15000, output_tokens=100, evaluations=2)]
        rows += [dict(environment="maze", score=0.5, output_tokens=300, evaluations=4)] * 2
        anchors = dict(
            g2048=dict(baseline=5000, reference=15000), maze=dict(baseline=0, reference=1)
        )
        result = suite_scorecard(rows, anchors, "tokens")
        self.assertEqual(result["S"], 75)
        self.assertEqual(result["T"], 200)
        self.assertEqual(result["E"], 3)
        self.assertEqual(result["token_efficiency"], 375000)
        self.assertFalse(result["qualified"])
        rows[0]["output_tokens"] = None
        self.assertIsNone(suite_scorecard(rows, anchors, "tokens")["T"])

    def test_mixed_defaults_and_workload_include_final_audits(self):
        from research.meta_ocean.runner import workload

        config = Config()
        self.assertEqual(config.environments, ["g2048", "breakout", "maze"])
        self.assertEqual(
            (config.calibration_cases, config.audit_cases, config.final_audit_cases), (64, 64, 512)
        )
        result = workload(config)
        self.assertEqual(result["calibration_games"], 384)
        self.assertEqual(result["max_controller_trials"], 306)
        self.assertEqual(result["private_audit_games"], 59904)
        self.assertEqual(result["max_transitions_per_evaluation"], 20000)
        with self.assertRaises(ValueError):
            Config(environments=["maze", "maze"])
        with self.assertRaises(ValueError):
            Config(environments=[])

    def test_ratios_use_aggregate_resources_and_quality_floor(self):
        rows = [
            dict(score=120, output_tokens=100, evaluations=1),
            dict(score=180, output_tokens=900, evaluations=9),
        ]
        result = scorecard(rows, baseline=0, reference=100, objective="tokens")
        self.assertEqual(result["S"], 150)
        self.assertEqual(result["T"], 500)
        self.assertEqual(result["E"], 5)
        self.assertEqual(result["token_efficiency"], 300000)
        self.assertEqual(result["evaluation_efficiency"], 3000)
        self.assertTrue(result["qualified"])
        rows[0]["output_tokens"] = None
        result = scorecard(rows, baseline=0, reference=100, objective="tokens")
        self.assertIsNone(result["token_efficiency"])
        self.assertLess(result["selection_score"], 0)

    def test_rejects_overlapping_replicates_and_invalid_caps(self):
        with self.assertRaises(ValueError):
            Config(development=[0], validation=[0])
        with self.assertRaises(ValueError):
            Config(evaluations=0)
        with self.assertRaises(ValueError):
            Config(batch_size=0)


class TrialTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonfinite_json_is_rejected_at_worker_boundary(self):
        from types import SimpleNamespace

        from research.meta_ocean.sandbox import Worker
        from rsikit.evaluation import PolicyError

        for value in (b"NaN", b"1e999", b"-Infinity"):
            reader = asyncio.StreamReader()
            reader.feed_data(b'{"op":"commit","value":' + value + b"}\n")
            reader.feed_eof()
            worker = Worker("unused")
            worker.process = SimpleNamespace(stdout=reader)
            with self.assertRaises(PolicyError):
                await worker.receive()

    async def test_interrupted_evaluation_remains_charged_and_unknown(self):
        from research.providers import BudgetProvider
        from tests.providers import ScriptedProvider

        async def evaluate(source, seeds):
            raise asyncio.CancelledError

        model = BudgetProvider(
            ScriptedProvider([]),
            max_calls=1,
            max_tokens=10000,
            spend_cap=1,
            max_input_tokens=4096,
            max_output_tokens=128,
            input_price=1,
            output_price=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            trial = Trial(Config(), Path(directory), policies()[0].source, evaluate, model, 0)
            with self.assertRaises(asyncio.CancelledError):
                await trial.handle(dict(op="evaluate", value=policies()[0].source))
            row = trial.finish("cancelled")
            self.assertEqual(row["evaluations"], 1)
            self.assertEqual(row["successful_evaluations"], 0)
            self.assertTrue(row["incomplete_transitions_unknown"])

    async def test_failed_and_duplicate_submissions_cost_credits_and_keep_commit(self):
        starter = policies()[0].source
        submitted = []

        async def evaluate(source, seeds):
            submitted.append((source, list(seeds)))
            return dict(score=42, scores=[42] * len(seeds), steps=len(seeds) * 5)

        with tempfile.TemporaryDirectory() as directory:
            trial = Trial(Config(evaluations=3), Path(directory), starter, evaluate, None, 0)
            bad = await trial.handle(dict(op="evaluate", value="invalid python !"))
            self.assertIn("error", bad)
            good = await trial.handle(dict(op="evaluate", value=starter))
            await trial.handle(dict(op="commit", value=good["id"]))
            await trial.handle(dict(op="evaluate", value=starter))
            exhausted = await trial.handle(dict(op="evaluate", value=starter))
            self.assertIn("error", exhausted)
            self.assertEqual(trial.evaluations, 3)
            self.assertEqual(len(submitted), 2)
            self.assertEqual(submitted[0][1], list(range(10)))
            self.assertEqual(trial.incumbent, starter)
            self.assertEqual(trial.transitions, 100)
            unknown = await trial.handle(dict(op="commit", value="not-evaluated"))
            self.assertIn("error", unknown)


@unittest.skipUnless(find_spec("gepa"), "install the meta extra")
class GepaTests(unittest.TestCase):
    def test_unpromoted_child_cannot_be_selected_at_budget_boundary(self):
        from research.meta_ocean.gepa import optimize

        def benchmark(source):
            return dict(selection_score=10 if "improved" in source else 0)

        with tempfile.TemporaryDirectory() as directory:
            result = optimize(
                "def search(**kwargs): pass",
                benchmark,
                lambda _: "def search(**kwargs): pass # improved",
                Config(revisions=1, benchmark_limit=3),
                Path(directory),
            )
            self.assertNotIn("improved", result["controller"])

    def test_revision_cap_does_not_launch_extra_parent_searches(self):
        from research.meta_ocean.gepa import optimize

        with tempfile.TemporaryDirectory() as directory:
            result = optimize(
                "def search(**kwargs): pass",
                lambda source: dict(selection_score=0),
                lambda _: "def search(**kwargs): pass # same quality",
                Config(revisions=1, benchmark_limit=20),
                Path(directory),
            )
            self.assertEqual(result["benchmarks"], 3)

    def test_real_gepa_edits_controller_and_respects_complete_benchmark_limit(self):
        from research.meta_ocean.gepa import optimize

        evaluated, edited = [], []

        def benchmark(source):
            evaluated.append(source)
            return dict(
                selection_score=10 if "improved" in source else 0,
                S=100,
                T=1000,
                E=2,
                qualified=True,
            )

        def edit(prompt):
            edited.append(prompt)
            return "def search(**kwargs):\n    pass  # improved\n"

        with tempfile.TemporaryDirectory() as directory:
            result = optimize(
                "def search(**kwargs): pass",
                benchmark,
                edit,
                Config(revisions=1, benchmark_limit=4),
                Path(directory),
            )
            self.assertIn("improved", result["controller"])
            self.assertEqual(result["scorecard"]["selection_score"], 10)
            self.assertLessEqual(len(evaluated), 4)
            self.assertEqual(len(edited), 1)
            self.assertIn("controller", edited[0])


@unittest.skipUnless(os.environ.get("RSIKIT_META_DOCKER_TESTS"), "requires built meta worker image")
class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_private_action_is_failure_score_not_campaign_crash(self):
        from research.meta_ocean.runner import run_trial
        from research.providers import BudgetProvider
        from tests.providers import ScriptedProvider

        model = BudgetProvider(
            ScriptedProvider([]),
            max_calls=1,
            max_tokens=10000,
            spend_cap=1,
            max_input_tokens=4096,
            max_output_tokens=128,
            input_price=1,
            output_price=1,
        )
        policy = """from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return ["bad"] * len(observation) if len(observation) == 2 else [0] * len(observation)
"""
        source = f"""def search(**kwargs):
    result = kwargs["evaluate"]({policy!r})
    kwargs["commit"](result["id"])
"""
        with tempfile.TemporaryDirectory() as directory:
            row = await run_trial(
                Config(evaluations=1, audit_cases=2, max_steps=5),
                source,
                Path(directory) / "trial",
                0,
                model=model,
            )
            self.assertEqual(row["successful_evaluations"], 1)
            self.assertEqual(row["score"], 0)
            self.assertIsNone(row["audit_transitions"])
            self.assertIn("PolicyError", row["audit_error"])

    async def test_failed_controller_keeps_committed_policy_and_audits_disjoint_cases(self):
        from research.meta_ocean.runner import run_trial
        from research.providers import BudgetProvider
        from tests.providers import ScriptedProvider

        config = Config(evaluations=1, audit_cases=2, max_steps=5)
        model = BudgetProvider(
            ScriptedProvider([]),
            max_calls=1,
            max_tokens=10000,
            spend_cap=1,
            max_input_tokens=4096,
            max_output_tokens=128,
            input_price=1,
            output_price=1,
        )
        source = """
def search(**kwargs):
    result = kwargs["evaluate"](kwargs["starter"])
    kwargs["commit"](result["id"])
    raise RuntimeError("after commit")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial"
            row = await run_trial(config, source, path, 0, model=model)
            self.assertEqual(row["evaluations"], 1)
            self.assertEqual(row["completed_transitions"], 50)
            self.assertEqual(row["audit_transitions"], 10)
            self.assertIn("after commit", row["error"])
            import json

            audit = json.loads((path / "audit.json").read_text())
            self.assertEqual([r["seed"] for r in audit["results"]], [1000, 1001])
            self.assertTrue((path / "incumbent.py").is_file())

    async def test_controller_cannot_read_host_or_simulator_and_is_killed_on_timeout(self):
        from research.meta_ocean.sandbox import run_controller

        seen = []

        async def handle(request):
            seen.append(request)
            return {}

        source = """
def search(**kwargs):
    import os, importlib.util
    assert "OPENROUTER_API_KEY" not in os.environ
    assert importlib.util.find_spec("pufferlib") is None
    assert not os.path.exists("/app/runs")
    kwargs["generate"]("isolated")
"""
        await run_controller(source, {}, handle, image="rsikit-meta-worker:local", timeout=30)
        self.assertEqual(seen, [dict(op="generate", value="isolated")])
        with self.assertRaises(asyncio.TimeoutError):
            await run_controller(
                "while True: pass", {}, handle, image="rsikit-meta-worker:local", timeout=2
            )

    async def test_native_rollout_uses_ten_games_and_only_remote_actions(self):
        from functools import partial

        from research.meta_ocean.sandbox import RemotePolicy
        from research.ocean.evaluator import rollout

        source = policies()[0].source
        result = await rollout(
            source,
            list(range(10)),
            batch_size=32,
            max_steps=5,
            _policy_factory=partial(RemotePolicy, image="rsikit-meta-worker:local"),
        )
        self.assertEqual(result["steps"], 50)
        self.assertEqual(len(result["results"]), 10)
        self.assertTrue(all(r["episodes"] == 1 for r in result["results"]))


@unittest.skipUnless(
    os.environ.get("RSIKIT_META_DOCKER_TESTS") and find_spec("gepa"),
    "requires meta extra and built worker image",
)
class CampaignSmokeTests(unittest.TestCase):
    def test_real_gepa_docker_ocean_campaign_with_scripted_model(self):
        from unittest.mock import patch

        from rich.console import Console

        from research.meta_ocean import runner
        from research.providers import RESPONSE, BudgetProvider
        from rsikit.progress import bind_run

        class ScriptedModel:
            def __init__(self, editor):
                self.editor = editor

            async def acall(self, prompt):
                event = RESPONSE.get()
                event.update(usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0)
                if self.editor:
                    return (
                        """def search(**kwargs):
    result = kwargs["evaluate"](kwargs["starter"])
    kwargs["commit"](result["id"])
""",
                        [],
                    )
                environment = (
                    "maze"
                    if "Ocean Maze:" in prompt
                    else "breakout"
                    if "Ocean Breakout:" in prompt
                    else "g2048"
                )
                return policies(environment)[-1].source, []

        def model(config, path, *, editor=False):
            return BudgetProvider(
                ScriptedModel(editor),
                max_calls=1,
                max_tokens=100000,
                spend_cap=1,
                max_input_tokens=65536,
                max_output_tokens=16384,
                input_price=1,
                output_price=1,
            )

        config = Config(
            objective="performance",
            development=[0],
            validation=[1],
            test=[2],
            evaluations=2,
            audit_cases=2,
            final_audit_cases=3,
            max_steps=5,
            revisions=1,
            benchmark_limit=4,
        )
        loop = asyncio.new_event_loop()
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(runner, "provider", model),
                bind_run(Path(directory), console=Console(file=io.StringIO())),
            ):
                path = Path(directory) / "campaign"
                anchors = {name: dict(baseline=0, reference=100) for name in config.environments}
                result = runner.campaign(config, path, anchors, loop)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["editor"]["calls"], 1)
                self.assertEqual(result["editor"]["output_tokens"], 10)
                self.assertTrue((path / "winner.py").is_file())
                for phase in ("validation", "test"):
                    baseline = result["heldout"][phase]["baseline"]
                    self.assertEqual(baseline["E"], 2)
                    self.assertEqual(baseline["T"], 10)
                    self.assertEqual(set(baseline["environments"]), set(config.environments))
                    self.assertEqual(len(baseline["trials"]), 3)
                    for trial in baseline["trials"]:
                        self.assertEqual(trial["completed_transitions"], 100)
                        self.assertEqual(trial["audit_transitions"], 15 if phase == "test" else 10)
                        self.assertEqual(trial["failures"], 0)
                log = (Path(directory) / "run.log").read_text()
                self.assertIn("GEPA accepted controller", log)
                self.assertIn("audit: 2/2 games", log)
                self.assertNotIn("Invalid progress record", log)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
