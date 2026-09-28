"""Check that Rich output is emitted before a generation or evaluation batch ends."""

import asyncio
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from rich.console import Console
from rich.text import Text
from slick import prompts

import examples.alphaevolve as example
from research import alphaevolve
from research.alphaevolve.generation import _PolicyResponse
from research.alphaevolve.improved import AlphaEvolve, Config
from rsikit.evaluation import PolicyError
from tests.helpers import fake_executor, recorded_run
from tests.providers import ScriptedProvider
from tests.test_episode_storage import trajectory
from tests.test_run import RESPONSE, FakeEvaluation


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_policy_summaries_scores_and_failure_log(self):
        output = io.StringIO()
        console = Console(file=output, width=160, force_terminal=False)
        provider = ScriptedProvider(
            [
                _PolicyResponse(
                    name="First [bold]",
                    description="Push left as a baseline.",
                    implementation=RESPONSE["implementation"],
                ),
                _PolicyResponse(
                    name="Second",
                    description="Try a different strategy.",
                    implementation=RESPONSE["implementation"] + "\n# second\n",
                ),
            ]
        )
        acall = provider.acall
        calls = 0
        generation_started, generation_release = asyncio.Event(), asyncio.Event()

        async def generate(*args, **kwargs):
            nonlocal calls
            calls += 1
            response = await acall(*args, **kwargs)
            if calls == 2:
                generation_started.set()
                await generation_release.wait()
            return response

        provider.acall = generate
        evaluation = FakeEvaluation()
        second_started, release = asyncio.Event(), asyncio.Event()

        async def evaluate(implementation, environment, seed):
            if "# second" in implementation:
                second_started.set()
                await release.wait()
                raise PolicyError("bad action")
            return trajectory(7.0, {})

        evaluation.evaluate.side_effect = evaluate
        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1", max_episode_steps=3) as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent),
            recorded_run(
                name="progress",
                console=console,
                path=Path(directory) / "run",
                environment=env,
                executor=fake_executor(evaluation=evaluation, concurrency=2),
            ) as (run, rollouts),
        ):
            agent = AlphaEvolve("task", provider, config=Config(max_repairs=0))
            task = asyncio.create_task(
                example.run_search(agent, run, rollouts, generations=1, batch_size=2)
            )
            try:
                await asyncio.wait_for(generation_started.wait(), 2)
                for _ in range(100):
                    if "Push left as a baseline." in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("First [bold]", output.getvalue())
                self.assertIn("Push left as a baseline.", output.getvalue())
                self.assertFalse(task.done())
                generation_release.set()
                await asyncio.wait_for(second_started.wait(), 2)
                for _ in range(100):
                    if "score=7" in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("score=7", output.getvalue())
                self.assertFalse(task.done())
            finally:
                generation_release.set()
                release.set()
                await task
            text = output.getvalue()
            self.assertIn("Second", text)
            self.assertIn("bad action", text)
            saved = (run.path / "run.log").read_text()
            self.assertIn("Push left as a baseline.", saved)
            self.assertIn("score=7", saved)
            self.assertIn("bad action", saved)
            self.assertEqual(run.policies()[0].description, "Push left as a baseline.")
            self.assertIn("Discarded Second", saved)
            self.assertEqual(agent.completed, 1)
            self.assertEqual(agent._pending, {})


if __name__ == "__main__":
    unittest.main()


class DashboardTests(unittest.TestCase):
    def display(self, width=120, height=45):
        from rsikit.progress import RunDisplay

        return RunDisplay(
            Path("/tmp/example-run"), Console(file=io.StringIO(), width=width, height=height)
        )

    def send(self, display, kind, **data):
        display.consume(
            logging.makeLogRecord(
                dict(
                    msg=kind,
                    levelno=logging.INFO,
                    levelname="INFO",
                    name="research.test",
                    progress=dict(kind=kind, **data),
                )
            )
        )

    def candidate(self, display, attempt, **changes):
        self.send(
            display,
            "candidate",
            **dict(
                dict(
                    batch_id="1",
                    attempt_id=str(attempt),
                    revision=0,
                    status="generated",
                    proposal_done=True,
                    name="[bold] literal",
                    description="A long description. " * 10,
                    policy_id="same-source",
                ),
                **changes,
            ),
        )

    def test_combined_progress_identity_repairs_and_overlap(self):
        d = self.display()
        self.send(
            d, "search_started", optimizer="Test", total_candidates=100, columns={}, resumed=False
        )
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=50)
        for i in range(44):
            self.candidate(d, i, status="evaluated" if i < 26 else "generated")
        self.assertEqual(d.batch_counts("1"), (70, 100))
        self.candidate(d, 0, status="evaluated")
        self.assertEqual(d.batch_counts("1"), (70, 100))
        self.candidate(d, 26, revision=1, status="repairing")
        self.candidate(d, 26, revision=0, status="evaluated")
        self.assertEqual(d.batch_counts("1"), (70, 100))
        self.send(d, "batch_started", batch_id="2", label="Generation 2", total_candidates=50)
        self.candidate(d, 0, batch_id="2", status="evaluated")
        self.assertEqual(d.current_batch, "1")
        self.assertEqual(d.completed, 72)
        self.assertEqual(d.total, 200)
        self.send(d, "search_finished", status="stopped", reason="target_reached")
        self.assertEqual(d.completed, 72)
        self.candidate(d, 27, status="evaluated")
        self.assertEqual(d.completed, 72)

    def test_policy_colors_follow_rows_across_panels_and_rank_changes(self):
        from rich.table import Column

        d = self.display()
        self.send(d, "search_started", total_candidates=2, columns={"origin": Column("Origin")})
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=2)
        leaders = []
        for attempt, (prefix, name) in enumerate((("a81bcd", "ALPHA"), ("c012f3", "BETA"))):
            policy_id = prefix + "0" * 58
            self.candidate(
                d, attempt, policy_id=policy_id, name=name, description=f"{name} description"
            )
            leaders.append(
                dict(
                    id=policy_id,
                    name=name,
                    description=f"{name} description",
                    score=123.456 + attempt,
                    generation=1,
                    extras={"origin": f"{name}-extra"},
                )
            )

        def colors(display, token):
            return [
                segment.style.color if segment.style else None
                for segment in display.console.render(display.render())
                if token in segment.text
            ]

        self.send(d, "leaderboard", rows=leaders)
        alpha = colors(d, "ALPHA")
        self.assertGreaterEqual(len(alpha), 4)  # Leaderboard and proposal table cells.
        self.assertIsNotNone(alpha[0])
        self.assertEqual(set(alpha), {alpha[0]})
        self.assertEqual(set(colors(d, "a81bcd")), {alpha[0]})
        self.assertEqual(colors(d, "123.456"), [alpha[0]])
        self.assertNotEqual(set(colors(d, "BETA")), {alpha[0]})
        self.candidate(
            d,
            0,
            policy_id=leaders[0]["id"],
            name="ALPHA",
            description="ALPHA description",
            status="evaluated",
            score=123.456,
        )
        self.send(d, "leaderboard", rows=list(reversed(leaders)))
        self.assertEqual(set(colors(d, "ALPHA")), {alpha[0]})
        assigned = [d._color(f"policy-{i}") for i in range(1000)]
        self.assertEqual(len(set(assigned)), len(assigned))
        self.assertEqual(assigned, [d._color(f"policy-{i}") for i in range(1000)])
        self.assertIsNone(d._color(None))

    def test_work_tables_show_completions_and_use_terminal_height(self):
        d = self.display(height=60)
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=2)
        self.candidate(d, 0, policy_id="a81bcd", name="ALPHA", description="First proposal")
        proposal = tuple(d.proposals)
        self.candidate(d, 0, policy_id="a81bcd", name="ALPHA", status="evaluating")
        self.assertFalse(d.evaluations)
        with (
            patch("logging.time.time", return_value=45296),
            patch("logging.time.time_ns", return_value=45296_000_000_000),
        ):
            self.candidate(d, 0, policy_id="a81bcd", name="ALPHA", status="evaluated", score=7)
        self.assertEqual(d.evaluations[0]["time"], "12:34:56")
        self.assertEqual(tuple(d.proposals), proposal)
        self.assertEqual(d.evaluations[0]["score"], 7)
        before = len(d.evaluations)
        self.candidate(d, 0, policy_id="a81bcd", name="ALPHA", status="evaluated", score=7)
        self.assertEqual(len(d.evaluations), before)
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=2)
        self.assertEqual(tuple(d.proposals), proposal)
        self.assertEqual(len(d.evaluations), before)
        d.console.print(d.render())
        output = d.console.file.getvalue()
        self.assertEqual(len(output.splitlines()), 60)
        self.assertIn("Score", output)
        self.assertIn("Duration", output)
        evaluation_output = "\n".join(line[60:] for line in output.splitlines())
        self.assertIn("Time", evaluation_output)
        self.assertIn("12:34:56", evaluation_output)
        self.assertIn("ALPHA", evaluation_output)
        self.assertNotIn("BETA", evaluation_output)
        self.send(d, "batch_finished", batch_id="1", status="completed")
        self.assertEqual(tuple(d.proposals), proposal)
        self.assertEqual(len(d.evaluations), before)
        self.send(d, "batch_started", batch_id="2", label="Generation 2", total_candidates=1)
        self.assertFalse(d.proposals)
        self.assertFalse(d.evaluations)
        self.candidate(
            d, 1, batch_id="2", policy_id="c012f3", name="BETA", description="Second proposal"
        )
        self.assertEqual([row["name"] for row in d.proposals], ["BETA"])
        self.candidate(d, 1, batch_id="2", name="BETA", status="evaluated", score=8)
        self.assertEqual([row["name"] for row in d.evaluations], ["BETA"])

    def test_proposals_only_queue_completed_generations_once(self):
        d = self.display()
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=None)
        for status in ("planned", "generating", "failed"):
            self.candidate(d, 0, status=status, proposal_done=False)
        self.assertFalse(d.proposals)
        with (
            patch("logging.time.time", return_value=45296),
            patch("logging.time.time_ns", return_value=45296_000_000_000),
        ):
            for attempt in range(1, 53):
                self.candidate(d, attempt, name=f"Proposal {attempt}", description="Complete")
                self.candidate(d, attempt, name=f"Proposal {attempt}", description="Complete")
        self.assertEqual(len(d.proposals), 50)
        self.assertEqual(
            [row["attempt_id"] for row in d.proposals], [str(i) for i in range(52, 2, -1)]
        )
        d.console.print(d.render())
        output = d.console.file.getvalue()
        self.assertIn("12:34:56", output)
        self.assertLess(output.index("Proposal 52"), output.index("Proposal 51"))
        self.assertNotIn("Proposal 3 ", output)

    def test_completed_evaluations_are_a_bounded_newest_first_queue(self):
        d = self.display()
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=None)
        for attempt, status in enumerate(
            (
                "generating",
                "generated",
                "evaluating",
                "repairing",
                "failed",
                "discarded",
                "cancelled",
            )
        ):
            self.candidate(d, attempt, status=status)
        self.assertFalse(d.evaluations)
        for attempt in range(100, 152):
            self.candidate(
                d,
                attempt,
                name=f"Result {attempt}",
                policy_id=f"{attempt:064x}",
                status="evaluated",
                score=float(attempt),
                duration=12,
            )
        self.assertEqual(len(d.evaluations), 50)
        self.assertEqual(
            [row["attempt_id"] for row in d.evaluations], [str(i) for i in range(151, 101, -1)]
        )
        self.candidate(d, 151, status="evaluated", score=999)
        self.assertEqual(d.evaluations[0]["score"], 151)
        d.console.print(d.render())
        output = d.console.file.getvalue()
        self.assertLess(output.index("Result 151"), output.index("Result 150"))
        self.assertNotIn("Result 102", output)
        self.assertIn("00:00:12", output)

    def test_followup_search_and_logged_evaluation_duration(self):
        d = self.display()
        self.send(d, "search_started", total_candidates=1)
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=None)
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=1)
        self.assertEqual(d.batch_counts("1"), (0, 2))
        self.candidate(d, 1, status="evaluated")
        self.send(d, "batch_finished", batch_id="1", status="completed")
        self.send(d, "search_finished", status="completed", reason="completed")
        self.send(d, "search_started", total_candidates=2)
        self.assertIsNone(d.finished)
        self.assertEqual(d.status, "running")
        self.assertEqual((d.completed, d.total), (2, 4))
        self.send(d, "batch_started", batch_id="2", label="Generation 2", total_candidates=1)
        with patch("rsikit.progress.monotonic", return_value=100):
            self.candidate(d, 2, batch_id="2", status="evaluating")
        with patch("rsikit.progress.monotonic", return_value=107):
            self.candidate(d, 2, batch_id="2", status="evaluated", score=3)
            d.console.print(d.render())
        self.assertIn("00:00:07", d.console.file.getvalue())

    def test_terminal_counts_and_unknown_totals(self):
        d = self.display()
        self.send(d, "batch_started", batch_id="1", label="Batch 1", total_candidates=None)
        for i, status in enumerate(("discarded", "failed", "cancelled")):
            self.candidate(d, i, status=status, proposal_done=status != "cancelled")
        self.assertEqual(d.batch_counts("1"), (4, None))
        self.assertIsNone(d.total)
        self.send(d, "batch_finished", batch_id="1", status="cancelled")
        self.assertEqual(d.completed, 4)
        empty = self.display()
        self.send(
            empty, "search_started", optimizer="Test", total_candidates=0, columns={}, resumed=False
        )
        self.send(empty, "search_finished", status="completed", reason="completed")
        self.assertEqual(empty.completed, 0)
        empty.console.print(empty.render())
        zero = self.display()
        self.send(zero, "batch_started", batch_id="1", label="Generation 1", total_candidates=0)
        self.assertEqual(zero.batches["1"]["status"], "completed")
        finished = self.display()
        self.send(finished, "batch_started", batch_id="1", label="Generation 1", total_candidates=1)
        self.candidate(finished, 1, status="evaluated")
        self.send(finished, "batch_finished", batch_id="1", status="completed")
        finished.console.print(finished.render())
        self.assertEqual(finished.console.file.getvalue().count("1/1"), 2)

    def test_progress_bars_live_in_work_panel_titles(self):
        d = self.display()
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=4)
        self.candidate(d, 0)
        self.candidate(d, 1)
        self.candidate(d, 0, status="evaluated", score=7)
        d.console.print(d.render())
        output = d.console.file.getvalue()
        titles = next(line for line in output.splitlines() if "PROPOSALS" in line)
        proposal, evaluation = titles.split("EVALUATIONS")
        self.assertIn("2/4", proposal)
        self.assertIn("1/4", evaluation)
        self.assertIn("━", proposal)
        self.assertIn("━", evaluation)
        self.assertNotIn("Total Run Progress:", output)
        self.assertNotIn("Generation Progress:", output)
        self.assertEqual(len(output.splitlines()), 45)

    def test_exception_reason_is_literal_and_cannot_control_terminal(self):
        d = self.display()
        self.send(d, "search_started", total_candidates=0)
        self.send(
            d, "search_finished", status="failed", reason="bad \x1b]0;UNSAFE\x07 \x1b[2J [bold]"
        )
        d.console.print(d.render())
        output = d.console.file.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertIn("[bold]", output)

    def test_exact_history_supersedes_conservative_checkpoint_count(self):
        d = self.display()
        self.send(d, "search_started", total_candidates=4, completed_candidates_before=1)
        self.assertEqual(d.completed, 2)
        self.send(d, "batch_started", batch_id="1", label="Restored", total_candidates=3)
        for attempt, status in enumerate(("evaluated", "discarded", "cancelled")):
            self.candidate(d, attempt, status=status, restored=True, proposal_done=False)
        self.send(d, "batch_finished", batch_id="1", status="stopped", restored=True)
        self.assertEqual(d.completed, 4)
        self.assertEqual(d.total, 8)

    def test_narrow_terminal_only_shows_completed_evaluations(self):
        d = self.display(80, 24)
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=1)
        self.candidate(d, 1, name="ACTIVE_POLICY", status="evaluating")
        d.console.print(d.render())
        self.assertNotIn("ACTIVE_POLICY", d.console.file.getvalue())
        d.console.file.seek(0)
        d.console.file.truncate()
        self.candidate(d, 1, name="ACTIVE_POLICY", status="evaluated", score=7)
        d.console.print(d.render())
        output = d.console.file.getvalue()
        self.assertIn("ACTIVE_POLICY", output)
        self.assertNotIn("evaluating", output)
        self.assertLessEqual(len(output.splitlines()), 24)

    def test_only_description_and_evaluation_name_grow_with_terminal(self):
        positions, description_lengths = [], []
        for width in (140, 200):
            d = self.display(width=width)
            self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=1)
            self.candidate(d, 0, name="N" * 60, description="D" * 200)
            self.candidate(d, 0, name="N" * 60, status="evaluated", score=123.456, duration=12)
            d.console.print(d.render())
            lines = d.console.file.getvalue().splitlines()
            header = next(line for line in lines if "Time" in line and "Duration" in line)
            proposal_start = header.index("Time")
            evaluation_start = header.index("Time", proposal_start + 4)
            positions.append(
                [
                    header.index(label, evaluation_start) - evaluation_start
                    for label in ("Time", "ID", "Name", "Score", "Duration")
                ]
            )
            description_lengths.append(max(line.count("D") for line in lines))
            self.assertIn("N" * 15 + "…", "\n".join(lines))
        self.assertEqual(positions[0][:3], positions[1][:3])
        self.assertEqual(positions[0][4] - positions[0][3], positions[1][4] - positions[1][3])
        self.assertEqual(positions[1][3] - positions[0][3], 30)
        self.assertGreater(description_lengths[1], description_lengths[0])

    def test_layout_custom_columns_safe_text_and_invalid_record(self):
        from rich.table import Column

        for width, height in ((120, 45), (80, 24)):
            d = self.display(width, height)
            self.send(
                d,
                "search_started",
                optimizer="Test",
                total_candidates=50,
                columns={"operation": Column("Operation")},
                resumed=False,
            )
            self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=50)
            for i in range(20):
                self.candidate(d, i, name="[bold] literal\x1b[2J")
            self.send(
                d,
                "leaderboard",
                rows=[
                    dict(
                        id="abc",
                        name="First",
                        description="test",
                        score=3,
                        generation=1,
                        extras={"operation": "Remix"},
                    )
                ],
            )
            self.send(
                d,
                "candidate",
                batch_id="1",
                attempt_id="bad",
                revision=-1,
                status="bogus",
                proposal_done=True,
            )
            self.assertEqual(d.completed, 20)
            d.console.print(d.render())
            output = d.console.file.getvalue()
            for label in ("w/ Test", "PROPOSALS", "EVALUATIONS", "EVENT LOG"):
                self.assertIn(label, output)
            self.assertNotIn("\x1b", output)
            self.assertTrue(any("Invalid progress" in line.plain for line in d.logs))
            self.assertLessEqual(max(map(len, output.splitlines())), width)
            self.assertLessEqual(len(output.splitlines()), height)
            if width == 120:
                self.assertIn("Remix", output)
                self.assertIn("[bold] literal", output)

    def test_event_log_types_have_fixed_prefixes_and_preserve_error_details(self):
        from rsikit.progress import _event_lines

        console = self.display().console
        for level, status, event, label, color in (
            (logging.INFO, None, None, "INFO", "cyan"),
            (logging.DEBUG, None, None, "DEBUG", "grey50"),
            (logging.WARNING, "generated", None, "WARNING", "yellow"),
            (logging.ERROR, "completed", None, "ERROR", "red"),
            (logging.INFO, "failed", None, "ERROR", "red"),
            (logging.INFO, "repairing", None, "WARNING", "yellow"),
            (logging.INFO, "evaluated", None, "SUCCESS", "green"),
            (logging.INFO, "completed", None, "SUCCESS", "green"),
            (logging.INFO, None, "policy_generated", "SUCCESS", "green"),
        ):
            record = logging.makeLogRecord(
                dict(
                    msg="Message [bold]\x1b[2J\n  error details",
                    name="research.internal.module",
                    levelno=level,
                    progress=dict(status=status),
                    event=event,
                    exc_info=(ValueError, ValueError("broken"), None),
                )
            )
            lines = list(_event_lines(record, "2026-09-28T12:34:56.789Z"))
            self.assertEqual(lines[0].plain, f"12:34:56  {label:7}  Message [bold]")
            self.assertTrue(lines[0].get_style_at_offset(console, 0).dim)
            self.assertEqual(lines[0].get_style_at_offset(console, 10).color.name, color)
            self.assertEqual(lines[1].plain, " " * 19 + "  error details")
            self.assertEqual(lines[2].plain, " " * 19 + "ValueError: broken")

    def test_tables_reserve_fixed_height_and_leave_remaining_space_to_event_log(self):
        for width, height in ((120, 45), (120, 60), (80, 24)):
            d = self.display(width=width, height=height)
            self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=None)
            d.console.print(d.render())
            empty_lines = d.console.file.getvalue().splitlines()
            empty_log_start = next(i for i, line in enumerate(empty_lines) if "EVENT LOG" in line)
            self.assertEqual(empty_log_start, 41 if width >= 100 else 20)
            for attempt in range(50):
                self.candidate(d, attempt)
                self.candidate(d, attempt, status="evaluated", score=attempt)
            d.logs.extend([Text("Long log message " * 20)] * 100 + [Text("Newest log entry")])
            d.console.file.seek(0)
            d.console.file.truncate()
            d.console.print(d.render())
            lines = d.console.file.getvalue().splitlines()
            log_start = next(i for i, line in enumerate(lines) if "EVENT LOG" in line)
            self.assertEqual(log_start, empty_log_start)
            self.assertEqual(len(lines), height)
            self.assertIn("Newest log entry", lines[-2])
            headers = [
                segment
                for segment in d.console.render(d.render())
                if segment.text.strip() == "Time"
            ]
            self.assertEqual(len(headers), 2)
            self.assertTrue(all(segment.style.dim for segment in headers))

    def test_leaderboard_reserves_configured_elites_capped_at_ten(self):
        for size in (5, 10, 50):
            d = self.display()
            self.send(d, "search_started", total_candidates=100, leaderboard_size=size)
            reserved = min(size, 10)
            self.assertEqual(d.render().renderables[0].height, reserved + 3)
            self.send(
                d,
                "leaderboard",
                rows=[
                    dict(
                        id=str(i),
                        name=f"Elite {i}",
                        description="",
                        score=100 - i,
                        generation=1,
                        extras={},
                    )
                    for i in range(size)
                ],
            )
            self.assertEqual(d.render().renderables[0].height, reserved + 3)
            d.console.print(d.render())
            output = d.console.file.getvalue()
            self.assertIn(f"Elite {reserved - 1}", output)
            self.assertNotIn(f"Elite {reserved} ", output)
            self.assertEqual(len(output.splitlines()), 45)

    def test_leaderboard_borders_show_run_metadata_and_generation_progress(self):
        d = self.display(width=160)
        self.send(d, "environment", name="BipedalWalker-v3")
        self.send(
            d, "search_started", optimizer="EliteSearch", total_candidates=4, total_generations=2
        )
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=2)
        self.candidate(d, 0, status="evaluated")
        self.candidate(d, 1, status="evaluated")
        self.send(d, "batch_started", batch_id="2", label="Generation 2", total_candidates=2)
        self.candidate(d, 2, batch_id="2", status="evaluated")
        with patch("rsikit.progress.monotonic", return_value=d.started + 75):
            d.console.print(d.render())
        lines = d.console.file.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("╭"))
        self.assertIn("BipedalWalker-v3 w/ EliteSearch", lines[0])
        self.assertIn("1/2 generations", lines[0])
        self.assertIn("00:01:15", lines[0])
        footer = next(line for line in lines if str(d.path) in line)
        self.assertTrue(footer.startswith("╰"))
        self.assertTrue(footer.endswith("╯"))
        self.assertEqual(len(lines), 45)
        self.assertNotIn("Run:", "\n".join(lines))
        self.candidate(d, 3, batch_id="2", status="evaluated")
        d.console.file.seek(0)
        d.console.file.truncate()
        d.console.print(d.render())
        self.assertIn("2/2 generations", d.console.file.getvalue().splitlines()[0])

    def test_repeated_render_does_not_mutate_columns_and_rejects_invalid_ids(self):
        from rich.table import Column

        d = self.display()
        column = Column("Operation")
        self.send(
            d,
            "search_started",
            optimizer="Test",
            total_candidates=1,
            columns={"operation": column},
            resumed=False,
        )
        self.send(d, "batch_started", batch_id="1", label="Generation 1", total_candidates=1)
        self.send(
            d,
            "leaderboard",
            rows=[
                dict(
                    id="abcdef",
                    name="Test",
                    description="test",
                    score=1,
                    generation=1,
                    extras={"operation": "New"},
                )
            ],
        )
        for _ in range(3):
            d.console.print(d.render())
        self.assertEqual(column._cells, [])
        self.candidate(d, 1, policy_id=5)
        self.assertEqual(d.completed, 0)
        d.console.print(d.render())


class RunLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_error_reaches_plain_output_and_durable_log(self):
        from rsikit import Run

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with Run.create(
                name="error", path=Path(directory) / "run", console=Console(file=output)
            ) as run:
                logger = logging.getLogger("research.test")
                logger.info(
                    "Starting",
                    extra={
                        "progress": dict(
                            kind="batch_started", batch_id="1", label="Batch 1", total_candidates=1
                        )
                    },
                )
                logger.error(
                    "Failed policy",
                    extra={
                        "progress": dict(
                            kind="candidate",
                            batch_id="1",
                            attempt_id="1",
                            revision=0,
                            status="failed",
                            proposal_done=True,
                            error="UNIQUE FAILURE CONTENT\n" + "x" * 9000,
                        )
                    },
                )
                saved = (run.path / "run.log").read_text()
                self.assertIn("UNIQUE FAILURE CONTENT", saved)
                self.assertIn("x" * 9000, saved)
                self.assertIn("UNIQUE FAILURE CONTENT", output.getvalue())

    async def test_standalone_generation_advances_past_settled_batches(self):
        from research.alphaevolve.original import AlphaEvolve, Config
        from research.shinkaevolve import ShinkaEvolve
        from rsikit import Run
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        for cls, templates in (
            (AlphaEvolve, "research/alphaevolve"),
            (ShinkaEvolve, "research/shinkaevolve/prompts"),
        ):
            with (
                self.subTest(optimizer=cls.__name__),
                tempfile.TemporaryDirectory() as directory,
                patch.object(prompts, "TEMPLATE_ROOT", Path(templates)),
                Run.create(
                    name="standalone",
                    path=Path(directory) / "run",
                    console=Console(file=io.StringIO()),
                ),
            ):
                agent = cls(
                    "test",
                    ScriptedProvider([program(0), program(1)]),
                    **({"config": Config(mode="rewrite")} if cls is AlphaEvolve else {}),
                )
                policies = await agent.generate(1)
                if cls is AlphaEvolve:
                    agent.update_scores({policies[0].id: 1})
                else:
                    agent.update({policies[0].id: 1})
                policies = await agent.generate(1)
                agent.evaluation_started(policies)
                display = _current_run.get()["display"]
                self.assertEqual(display.current_batch, "2")
                self.assertEqual(display.batches["1"]["status"], "completed")
                self.assertEqual(len(display.candidates), 1)

    async def test_repeated_shinka_search_counts_prior_attempts(self):
        from research.shinkaevolve import ShinkaEvolve
        from research.shinkaevolve.search import run_search
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1") as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/shinkaevolve/prompts")),
            recorded_run(
                name="shinka",
                path=Path(directory) / "run",
                environment=env,
                executor=fake_executor(evaluation=FakeEvaluation()),
                console=Console(file=io.StringIO()),
            ) as (run, rollouts),
        ):
            agent = ShinkaEvolve("test", ScriptedProvider([program(0), program(1)]))
            for completed in (2, 4):
                await run_search(agent, run, rollouts, generations=1, batch_size=1)
                display = _current_run.get()["display"]
                self.assertEqual((display.completed, display.total), (completed, completed))

    async def test_native_paper_resume_replays_only_known_completions(self):
        from research.alphaevolve import paper
        from research.alphaevolve.paper.evaluation import EvaluationResult
        from rsikit import Run
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        async def evaluate(policies):
            return {p.id: EvaluationResult({"reward": 1}) for p in policies}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/alphaevolve")),
        ):
            for invocation in range(2):
                with Run.create(
                    name="native",
                    path=Path(directory) / str(invocation),
                    console=Console(file=io.StringIO()),
                ):
                    agent = paper.AlphaEvolve(
                        "test",
                        ScriptedProvider([program(invocation)]),
                        database_path=Path(directory) / "population.sqlite",
                    )
                    try:
                        await paper.search(agent, evaluate, proposals=1)
                        display = _current_run.get()["display"]
                        self.assertEqual(
                            (display.completed, display.total),
                            (2 * (invocation + 1), 2 * (invocation + 1)),
                        )
                    finally:
                        agent.close()

    async def test_invalid_payload_is_diagnosed_in_plain_output_and_run_log(self):
        from rsikit import Run

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with Run.create(
                name="invalid", path=Path(directory) / "run", console=Console(file=output)
            ) as run:
                logger = logging.getLogger("research.test")
                logger.info(
                    "Starting", extra={"progress": dict(kind="search_started", total_candidates=1)}
                )
                logger.info("Original event", extra={"progress": dict(kind="candidate")})
                self.assertIn("Invalid progress record", output.getvalue())
                self.assertIn("Invalid progress record", (run.path / "run.log").read_text())
                self.assertIn("Original event", (run.path / "run.log").read_text())

    async def test_inner_loop_counts_policies_across_multiple_seeds(self):
        from examples import inner_loop
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        report = inner_loop.write_report

        def checked_report(run):
            display = _current_run.get()["display"]
            self.assertEqual((display.completed, display.total), (10, 10))
            self.assertEqual(len(display.leaders), 5)
            self.assertEqual(display.status, "completed")
            return report(run)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                inner_loop, "Executor", return_value=fake_executor(evaluation=FakeEvaluation())
            ),
            patch.object(inner_loop, "write_report", side_effect=checked_report),
            patch("rsikit.progress.Console", return_value=Console(file=io.StringIO())),
            patch.object(prompts, "TEMPLATE_ROOT", Path("rsikit/generation/prompts")),
        ):
            rows = await inner_loop.run_demo(
                ScriptedProvider([program(i) for i in range(5)]),
                Path(directory) / "run",
                seeds=(0, 1),
            )
            self.assertEqual(len(rows), 5)

    async def test_exception_and_cancellation_restore_terminal_and_file_errors_surface(self):
        from rsikit import Run
        from rsikit.progress import _current_run

        for error in (ValueError("broken"), asyncio.CancelledError()):
            with tempfile.TemporaryDirectory() as directory:
                output = io.StringIO()
                with self.assertRaises(type(error)):
                    with Run.create(
                        name="failure",
                        path=Path(directory) / "run",
                        console=Console(file=output, force_terminal=True, width=120, height=45),
                    ):
                        logger = logging.getLogger("research.test")
                        logger.info(
                            "Starting",
                            extra={
                                "progress": dict(
                                    kind="search_started",
                                    optimizer="Test",
                                    total_candidates=1,
                                    columns={},
                                    resumed=False,
                                )
                            },
                        )
                        display = _current_run.get()["display"]
                        raise error
                self.assertEqual(
                    display.status,
                    "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                )
                self.assertIn("\x1b[?25h", output.getvalue())
        with (
            tempfile.TemporaryDirectory() as directory,
            Run.create(name="disk", path=Path(directory) / "run"),
        ):
            binding = _current_run.get()
            with patch.object(binding["file"], "write", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    logging.getLogger("rsikit.test").info("persist this")

    async def test_generic_generation_activates_progress_and_retains_attempt_identity(self):
        import rsikit.generation as generation
        from rsikit import generate
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1") as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path(generation.__file__).parent / "prompts"),
            recorded_run(
                name="generic",
                path=Path(directory) / "run",
                environment=env,
                console=Console(file=io.StringIO()),
            ),
        ):
            provider = ScriptedProvider([program(0), program(0)])
            first = await generate("test", provider=provider)
            second = await generate("test", provider=provider)
            display = _current_run.get()["display"]
            self.assertTrue(display.active)
            self.assertEqual(display.completed, 2)
            self.assertEqual(len(display.candidates), 2)
            self.assertEqual(first.id, second.id)

    async def test_paper_resume_replays_history_before_new_proposals(self):
        from research.alphaevolve import paper
        from research.alphaevolve.search import run_search
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        with (
            tempfile.TemporaryDirectory() as directory,
            gym.make("CartPole-v1") as env,
            patch.object(prompts, "TEMPLATE_ROOT", Path("research/alphaevolve")),
        ):
            path = Path(directory) / "run"
            for resumed in (False, True):
                with recorded_run(
                    name=None if resumed else "resume",
                    path=path,
                    environment=env,
                    executor=fake_executor(evaluation=FakeEvaluation()),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts):
                    provider = ScriptedProvider([program(int(resumed))])
                    original = provider.acall

                    async def generate(*args, **kwargs):
                        if resumed:
                            display = _current_run.get()["display"]
                            self.assertEqual(display.completed, 2)
                            self.assertEqual(display.total, 4)
                            self.assertTrue(display.leaders)
                        return await original(*args, **kwargs)

                    provider.acall = generate
                    agent = paper.AlphaEvolve(
                        "test", provider, database_path=path / "population.sqlite"
                    )
                    try:
                        await run_search(agent, run, rollouts, generations=1, batch_size=1)
                        display = _current_run.get()["display"]
                        self.assertEqual(display.completed, 4 if resumed else 2)
                    finally:
                        agent.close()

    async def test_research_loops_report_all_alpha_and_shinka_variants(self):
        from research.alphaevolve import improved, original, paper
        from research.alphaevolve.search import run_search as alpha_search
        from research.shinkaevolve import Config, ShinkaEvolve
        from research.shinkaevolve.search import run_search as shinka_search
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program

        variants = [
            (name, cls, alpha_search, {})
            for name, cls in (
                ("original", original.AlphaEvolve),
                ("improved", improved.AlphaEvolve),
                ("paper", paper.AlphaEvolve),
            )
        ]
        variants.append(("shinka", ShinkaEvolve, shinka_search, {"config": Config()}))
        for name, cls, search, options in variants:
            with (
                tempfile.TemporaryDirectory() as directory,
                gym.make("CartPole-v1") as env,
                patch.object(
                    prompts,
                    "TEMPLATE_ROOT",
                    Path("research/shinkaevolve/prompts")
                    if name == "shinka"
                    else Path("research/alphaevolve"),
                ),
                recorded_run(
                    name=name,
                    path=Path(directory) / "run",
                    environment=env,
                    executor=fake_executor(evaluation=FakeEvaluation()),
                    console=Console(file=io.StringIO()),
                ) as (run, rollouts),
            ):
                agent = cls("test", ScriptedProvider([program(0)]), **options)
                try:
                    await search(agent, run, rollouts, generations=1, batch_size=1, seeds=(0, 1))
                    display = _current_run.get()["display"]
                    self.assertEqual((display.completed, display.total), (2, 2), name)
                    self.assertEqual(display.leaders[0]["score"], 7, name)
                    self.assertIn("island", display.columns)
                    self.assertEqual(display.status, "completed")
                finally:
                    if name == "paper":
                        agent.close()

    async def test_native_optimizer_events_without_example_setup(self):
        from research.elitesearch import Config, EliteSearch, Measurement
        from research.lineagesearch import Config as LineageConfig
        from research.lineagesearch import LineageSearch
        from rsikit import Run
        from rsikit.progress import _current_run
        from tests.test_elitesearch import program
        from tests.test_lineagesearch import experiments, families

        for kind in ("elitesearch", "lineagesearch"):
            responses = (
                [program(0)] if kind == "elitesearch" else [families(), experiments(0), program(0)]
            )

            async def evaluate(policies):
                display = _current_run.get()["display"]
                self.assertTrue(display.active)
                self.assertEqual(display.completed, 1)
                self.assertTrue(
                    any(r["status"] == "evaluating" for r in display.candidates.values())
                )
                return {p.id: Measurement({0: 3, 1: 5}) for p in policies}

            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(prompts, "TEMPLATE_ROOT", Path("research") / kind / "prompts"),
            ):
                async with Run.create(
                    name=kind, path=Path(directory) / "run", console=Console(file=io.StringIO())
                ):
                    provider = ScriptedProvider(responses)
                    agent = (
                        EliteSearch(
                            "test",
                            provider,
                            evaluate,
                            config=Config(population_size=1, generations=1),
                        )
                        if kind == "elitesearch"
                        else LineageSearch(
                            "test",
                            provider,
                            evaluate,
                            config=LineageConfig(
                                families=1, initial_per_family=1, max_attempts=1, decomposition_k=1
                            ),
                        )
                    )
                    await agent.run()
                    display = _current_run.get()["display"]
                    self.assertEqual(display.completed, 2)
                    self.assertEqual(display.total, 2)
                    self.assertEqual(display.leaders[0]["score"], 4)
                    self.assertTrue(display.columns)

    async def test_automatic_scoped_logging_and_cleanup(self):
        from rsikit import Run

        logger = logging.getLogger("research.test")
        parent = logging.getLogger("research")
        original = (parent.level, list(parent.handlers), parent.propagate)
        output = io.StringIO()
        console = Console(file=output, force_terminal=False)
        with tempfile.TemporaryDirectory() as directory:
            with Run.create(name="quiet", path=Path(directory) / "quiet", console=console):
                pass
            self.assertEqual(output.getvalue(), "")
            ready = asyncio.Event()

            async def work(name):
                async with Run.create(name=name, path=Path(directory) / name, console=console):
                    logger.info(
                        "starting %s",
                        name,
                        extra={
                            "progress": dict(
                                kind="search_started",
                                optimizer=name,
                                total_candidates=0,
                                columns={},
                                resumed=False,
                            )
                        },
                    )
                    ready.set()
                    await asyncio.sleep(0)
                    logger.info("private %s", name)
                    try:
                        raise ValueError(f"error {name}")
                    except ValueError:
                        logger.exception("failure %s", name)

            await asyncio.gather(work("one"), work("two"))
            for name, other in [("one", "two"), ("two", "one")]:
                text = (Path(directory) / name / "run.log").read_text()
                self.assertIn(f"private {name}", text)
                self.assertNotIn(f"private {other}", text)
                self.assertIn("Traceback", text)
            self.assertNotIn("\x1b", output.getvalue())
            self.assertEqual((parent.level, parent.handlers, parent.propagate), original)

    async def test_close_is_idempotent_and_late_tasks_do_not_write(self):
        from rsikit import Run

        with tempfile.TemporaryDirectory() as directory:
            release = asyncio.Event()

            async def later():
                await release.wait()
                logging.getLogger("rsikit.test").warning("too late")

            run = Run.create(
                name="closed", path=Path(directory) / "run", console=Console(file=io.StringIO())
            )
            async with run:
                task = asyncio.create_task(later())
            run.close()
            release.set()
            await task
            self.assertNotIn("too late", (run.path / "run.log").read_text())
