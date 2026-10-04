"""Resume at trusted trial boundaries, preserving budgets and the measured baseline."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from slick import prompts

from research.meta_ocean.checkpoints import read_json, run_lock, save_json, upgrade_candidates
from research.meta_ocean.elitetable import Config, EvolverTrial, provider, restore_trial, run_trial
from research.meta_ocean.experiment import append_json, digest, source_text
from research.ocean.baselines import policies
from research.providers import RESPONSE, BudgetProvider
from rsikit.policy import MAX_SOURCE


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_exhausted_baseline_audits_incumbent_instead_of_failing(self):
        from research.meta_ocean import runner

        source = policies()[0]._implementation

        class Model:
            async def acall(self, prompt):
                RESPONSE.get().update(
                    usage=dict(prompt_tokens=20, completion_tokens=10), actual_cost=0
                )
                return json.dumps(
                    dict(name="Valid", description="One admitted policy", implementation=source)
                ), []

        model = BudgetProvider(Model(), max_calls=1, max_input_tokens=65536, max_output_tokens=4096)
        slots = {key: asyncio.Semaphore(1) for key in ("trials", "models", "evaluations")}

        async def evaluate(config, candidate, seeds, path, **kwargs):
            self.assertEqual(candidate, source)
            return dict(score=7, scores=[7] * len(seeds), steps=len(seeds))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/elitesearch/prompts")),
            patch.object(runner, "evaluate_policy", side_effect=evaluate),
        ):
            row = await run_trial(
                Config(generation_size=1),
                None,
                Path(directory) / "trial",
                0,
                "g2048",
                slots,
                model=model,
            )
        self.assertIsNone(row["error"])
        self.assertIsNone(row["audit_error"])
        self.assertEqual(row["stop_reason"], "budget_exhausted")
        self.assertEqual(row["score"], 7)
        self.assertEqual(row["calls"], 1)

    async def test_recover_saved_budget_failure_runs_only_the_missing_audit(self):
        from research.meta_ocean import elitetable, runner

        source = policies()[0]._implementation
        slots = {key: asyncio.Semaphore(1) for key in ("trials", "models", "evaluations")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            row = dict(
                environment="g2048",
                replicate=0,
                generations=4,
                calls=250,
                policy_hash=digest(source),
                score=0,
                error="PolicyError: EliteTable baseline stopped before five complete generations: call cap",
                audit_error="incomplete sub-evolver",
            )
            save_json(path / "commit.json", row)
            save_json(path / "summary.json", row)
            (path / "incumbent.py").write_text(source)
            with (
                patch.object(elitetable, "provider", return_value=None),
                patch.object(
                    elitetable,
                    "execute_baseline",
                    side_effect=AssertionError("Must not rerun search"),
                ),
                patch.object(
                    runner, "evaluate_policy", return_value=dict(score=123, steps=64)
                ) as evaluate,
            ):
                result = await run_trial(Config(), None, path, 0, "g2048", slots)
            self.assertEqual(evaluate.call_count, 1)
            self.assertIsNone(result["error"])
            self.assertEqual(result["score"], 123)
            self.assertEqual(result["stop_reason"], "budget_exhausted")
            self.assertEqual(result["calls"], 250)
            self.assertEqual(read_json(path / "summary.json.interrupted-1")["score"], 0)

    def test_source_extraction_is_unambiguous_and_bounded(self):
        source = policies()[0]._implementation
        self.assertEqual(source_text(source), source)
        fenced = f"A policy:\n```python\n{source}\n```\nExplanation."
        self.assertEqual(source_text(fenced), source.rstrip() + "\n")
        ambiguous = fenced + "\n```python\npass\n```"
        self.assertEqual(source_text(ambiguous), ambiguous)
        with self.assertRaises(ValueError):
            source_text("x" * (MAX_SOURCE + 1))

    def test_migration_keeps_sources_and_usage_but_archives_old_candidate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "performance"
            path.mkdir()
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "evidence").write_text("keep")
            (path / "development").mkdir()
            (path / "development" / "evidence").write_text("old protocol")
            save_json(
                path / "organisms.json",
                [
                    dict(
                        generation=1,
                        policy_id="abc",
                        implementation="source",
                        score=-100,
                        seed_scores={"0": -100},
                        status="evaluated",
                    )
                ],
            )
            save_json(
                path / "generations.json",
                [dict(number=1, status="running", elite_ids=[], promoted_ids=[], error=None)],
            )
            (path / "editor_calls.jsonl").write_text("paid calls")
            upgrade_candidates(path)
            upgrade_candidates(path)  # Idempotent, including after a setup interruption.
            row = read_json(path / "checkpoint.json")["organisms"][0]
            self.assertEqual(row["implementation"], "source")
            self.assertEqual(row["status"], "generated")
            self.assertIsNone(row["score"])
            self.assertEqual((path / "editor_calls.jsonl").read_text(), "paid calls")
            self.assertEqual((baseline / "evidence").read_text(), "keep")
            self.assertEqual(
                (path / "development.interrupted-1" / "evidence").read_text(), "old protocol"
            )

    def test_live_legacy_run_and_duplicate_writer_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            process = f"12345 python -m research.meta_ocean --output {path}"
            with patch(
                "research.meta_ocean.checkpoints.subprocess.run",
                return_value=SimpleNamespace(stdout=process),
            ):
                with self.assertRaisesRegex(ValueError, "still active"):
                    with run_lock(path):
                        self.fail("Must not acquire a live run")
            with patch(
                "research.meta_ocean.checkpoints.subprocess.run",
                return_value=SimpleNamespace(stdout=""),
            ):
                with run_lock(path):
                    with self.assertRaisesRegex(ValueError, "already active"):
                        with run_lock(path):
                            self.fail("Must not acquire twice")

    def test_task_context_upgrade_archives_scores_but_keeps_baseline_and_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "performance"
            (path / "development").mkdir(parents=True)
            (root / "baseline").mkdir()
            (root / "baseline" / "evidence").write_text("measured baseline")
            row = dict(generation=1, policy_id="abc", implementation="saved source", score=9)
            save_json(path / "organisms.json", [row])
            save_json(path / "generations.json", [dict(number=1, status="completed")])
            save_json(path / "checkpoint.json", dict(organisms=[row]))
            save_json(path / "development" / "old-score.json", dict(score=9))
            (path / "editor_calls.jsonl").write_text("paid calls")
            upgrade_candidates(path, upgrade="task-context")
            upgrade_candidates(path, upgrade="task-context")
            saved = read_json(path / "checkpoint.json")["organisms"][0]
            self.assertIsNone(saved["score"])
            self.assertEqual(saved["status"], "generated")
            self.assertEqual(saved["implementation"], "saved source")
            self.assertFalse((path / "development").exists())
            self.assertTrue((path / "development.interrupted-1" / "old-score.json").exists())
            self.assertEqual((root / "baseline" / "evidence").read_text(), "measured baseline")
            self.assertEqual((path / "editor_calls.jsonl").read_text(), "paid calls")

    def test_resume_ignores_standalone_video_changes_but_rejects_baseline_engine_changes(self):
        from research.meta_ocean import elitetable, runner

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "metadata", return_value={}),
            patch.object(
                runner.subprocess, "run", return_value=SimpleNamespace(stdout="sha256:test")
            ),
            patch.object(elitetable, "campaign", return_value={}),
        ):
            path = Path(directory)
            config = Config(environments=["g2048"])
            runner._run(config, path)
            manifest = read_json(path / "manifest.json")
            manifest["protocol"] = "ocean-elitetable-meta-v5"
            manifest["baseline_sources"]["elitesearch/videos.py"] = "old replay tool"
            save_json(path / "manifest.json", manifest)
            self.assertEqual(runner._run(config, path, resume=True)["status"], "completed")
            manifest = read_json(path / "manifest.json")
            self.assertEqual(manifest["protocol"], "ocean-elitetable-meta-v7")
            manifest["baseline_sources"]["elitesearch/agent.py"] = "changed optimizer"
            save_json(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "baseline_sources changed"):
                runner._run(config, path, resume=True)

    def test_resume_allows_worker_changes_without_resetting_saved_results(self):
        import yaml

        from research.meta_ocean import elitetable, runner

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "metadata", return_value={}),
            patch.object(
                runner.subprocess, "run", return_value=SimpleNamespace(stdout="sha256:test")
            ),
            patch.object(elitetable, "campaign", return_value={}),
        ):
            path = Path(directory)
            config = Config(environments=["g2048"], trial_workers=4, evaluation_workers=4)
            runner._run(config, path)
            manifest = read_json(path / "manifest.json")
            manifest["protocol"] = "ocean-elitetable-meta-v6"
            save_json(path / "manifest.json", manifest)
            (path / "performance").mkdir()
            (path / "performance" / "checkpoint.json").write_text("preserve checkpoint")
            faster = config.model_copy(update=dict(trial_workers=8, evaluation_workers=16))
            runner._run(faster, path, resume=True)
            self.assertEqual(
                (path / "performance" / "checkpoint.json").read_text(), "preserve checkpoint"
            )
            saved = yaml.safe_load((path / "config.yaml").read_text())
            self.assertEqual((saved["trial_workers"], saved["evaluation_workers"]), (8, 16))
            with self.assertRaisesRegex(ValueError, "Resume config differs"):
                runner._run(faster.model_copy(update=dict(audit_cases=32)), path, resume=True)

    def test_unfinished_model_calls_keep_their_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            append_json(
                path, dict(call=1, status="ok", usage={"completion_tokens": 7}, actual_cost=0.1)
            )
            append_json(path, dict(call=2, status="running", usage=None, actual_cost=None))
            with patch("research.meta_ocean.elitetable.UsageOpenRouter"):
                model = provider(Config(), path)
            self.assertEqual(model.calls, 2)
            self.assertEqual(model.reserved_tokens, 2 * (65536 + 4096))
            self.assertIsNone(model.events[-1]["usage"])

    async def test_completed_trial_and_pending_audit_do_not_repeat_search(self):
        from research.meta_ocean import elitetable, runner

        source = policies()[0]._implementation
        slots = {name: asyncio.Semaphore(1) for name in ("trials", "models", "evaluations")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            row = dict(
                environment="g2048",
                replicate=0,
                error=None,
                policy_hash=digest(source),
                generations=5,
            )
            save_json(path / "commit.json", row)
            (path / "incumbent.py").write_text(source)
            save_json(path / "audit.json", dict(results=[dict(seed=1000, score=123)], steps=10))
            with (
                patch.object(elitetable, "provider", return_value=None),
                patch.object(
                    elitetable, "execute_baseline", side_effect=AssertionError("repeated search")
                ),
                patch.object(
                    runner, "evaluate_policy", side_effect=AssertionError("repeated audit")
                ),
            ):
                result = await run_trial(Config(), None, path, 0, "g2048", slots)
            self.assertEqual(result["score"], 123)
            with patch.object(elitetable, "provider", side_effect=AssertionError("new provider")):
                self.assertEqual(await run_trial(Config(), None, path, 0, "g2048", slots), result)

    def test_interrupted_evaluations_remain_charged(self):
        source = policies()[0]._implementation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            append_json(path / "oracle.jsonl", dict(event="evaluation_admitted", number=1))
            append_json(
                path / "oracle.jsonl",
                dict(
                    request=dict(op="evaluate", value=source),
                    response=dict(id=digest(source), score=10),
                ),
            )
            append_json(path / "oracle.jsonl", dict(event="evaluation_admitted", number=2))
            append_json(path / "oracle.jsonl", dict(event="baseline_proposal_rejected"))
            trial = EvolverTrial(Config(), path, source, None, None, 0, model_slots=None)
            restore_trial(trial)
            self.assertEqual((trial.evaluations, trial.failures, trial.best), (3, 2, 10))


if __name__ == "__main__":
    unittest.main()
