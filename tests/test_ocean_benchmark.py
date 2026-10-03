"""Benchmark arithmetic and finite real-process replay checks."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.benchmark_ocean import load_trace, summarize


class BenchmarkTests(unittest.TestCase):
    def test_batch_dependent_policy_fails_correctness_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "dependent.py"
            policy.write_text("""import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.full(len(observation), 0 if len(observation) == 1 else 1)
""")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.benchmark_ocean",
                    "correctness",
                    "--output",
                    str(root / "run"),
                    "--policy",
                    str(policy),
                    "--seeds",
                    "2",
                    "--max-steps",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertFalse(json.loads((root / "run/report.json").read_text())["passed"])
            self.assertNotEqual(completed.returncode, 0)

    def test_capacity_preserves_three_policy_mix_with_four_workers_and_counts_drain(self):
        from research.ocean.baselines import policies

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for index, policy in enumerate(policies()[:3]):
                path = root / f"policy-{index}.py"
                policy.to_file(path)
                sources.extend(["--policy", str(path)])
            output = root / "run"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.benchmark_ocean",
                    "capacity",
                    "--output",
                    str(output),
                    "--duration",
                    "0.001",
                    "--panels",
                    "1",
                    "--repeats",
                    "1",
                    "--seeds",
                    "1",
                    "--max-steps",
                    "1",
                    "--workers",
                    "4",
                    *sources,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            run = json.loads((output / "report.json").read_text())["runs"][0]
            self.assertEqual(run["valid_panels"] % 12, 0)
            self.assertEqual(
                {row["valid_panels"] for row in run["per_policy"].values()},
                {run["valid_panels"] // 3},
            )
            events = json.loads(next(output.glob("0-*/events.json")).read_text())
            events = [row for row in events if row["job_id"].startswith("job-")]
            drain = max(row["persisted"] for row in events) - max(
                row["submitted"] for row in events
            )
            self.assertGreaterEqual(run["drain_seconds"], drain - 0.01)

    def test_capacity_counts_only_success_and_includes_drain(self):
        rows = [
            dict(
                policy_id="p",
                status="ok",
                steps=40,
                service_seconds=2,
                queue_seconds=0,
                response_seconds=2,
            ),
            dict(
                policy_id="p",
                status="failed",
                steps=99,
                service_seconds=1,
                queue_seconds=3,
                response_seconds=4,
            ),
        ]
        result = summarize(rows, elapsed=10)
        self.assertEqual(result["valid_panels_per_second"], 0.1)
        self.assertEqual(result["actual_steps_per_second"], 4)
        self.assertEqual(result["failed_panels"], 1)
        self.assertEqual(result["per_policy"]["p"]["valid_panels"], 1)

    def test_trace_rejects_nonfinite_or_reversed_arrivals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            for arrivals in (
                [{"at": float("nan"), "policy_id": "p"}],
                [{"at": 2, "policy_id": "p"}, {"at": 1, "policy_id": "p"}],
            ):
                path.write_text(json.dumps(arrivals))
                with self.assertRaises(ValueError):
                    load_trace(path, {"p": object()})

    def test_tiny_replay_accounts_for_every_job_without_claiming_pilot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.benchmark_ocean",
                    "replay",
                    "--output",
                    str(output),
                    "--duration",
                    "0.01",
                    "--rate",
                    "400",
                    "--repeats",
                    "1",
                    "--seeds",
                    "2",
                    "--max-steps",
                    "3",
                    "--queue-limit",
                    "1",
                    "--workers",
                    "1",
                    "--batch-size",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            report = json.loads((output / "report.json").read_text())
            run = report["runs"][0]
            self.assertEqual(run["scheduled_jobs"], 4)
            self.assertEqual(run["accounted_jobs"], 4)
            self.assertEqual(run["valid_panels"], 4)
            self.assertEqual(run["dropped_jobs"], 0)
            self.assertGreater(run["backpressure_seconds"], 0)
            self.assertFalse(report["pilot_pass"])


if __name__ == "__main__":
    unittest.main()
