"""Rich dashboard components and presentation helpers."""

import logging
import re
from collections import deque
from dataclasses import replace
from itertools import product
from random import SystemRandom
from statistics import fmean

from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Column, Table
from rich.text import Text

_CONTROLS = re.compile(
    r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]"
)
_ID_COLORS = tuple(
    f"#{red:02x}{green:02x}{blue:02x}"
    for red, green, blue in product((0, 95, 135, 175, 215, 255), repeat=3)
    if max(red, green, blue) >= 215 and max(red, green, blue) - min(red, green, blue) >= 95
)


def _text(value):
    return Text(_CONTROLS.sub("", str(value if value is not None else "—")))


def _duration(seconds):
    if seconds is None:
        return "—"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _message(record):
    message = logging.Formatter().format(record)
    payload = getattr(record, "progress", None)
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, str) and error and error not in message:
        message += "\n" + error
    return message


def _event_lines(record, stamp):
    payload = getattr(record, "progress", None)
    status = payload.get("status") if isinstance(payload, dict) else None
    if record.levelno >= logging.ERROR or status == "failed":
        label, color = "ERROR", "red"
    elif record.levelno >= logging.WARNING or status in {"discarded", "repairing", "cancelled"}:
        label, color = "WARNING", "yellow"
    elif record.levelno < logging.INFO:
        label, color = "DEBUG", "grey50"
    elif status in {"generated", "evaluated", "completed"} or getattr(record, "event", None) in {
        "policy_generated",
        "policy_evaluated",
    }:
        label, color = "SUCCESS", "green"
    else:
        label, color = "INFO", "cyan"
    for index, line in enumerate(_message(record).splitlines()):
        yield Text.assemble(
            (stamp[11:19] if index == 0 else " " * 8, "dim"),
            "  ",
            (f"{label:7}" if index == 0 else " " * 7, f"bold {color}"),
            "  ",
            _text(line[:4096]),
        )


class LeaderboardPanel:
    """Ranked policies, custom columns, and the run summary borders."""

    def render(self, model, *, height, title, path, color):
        leaders = Table(box=None, expand=True, padding=(0, 1), header_style="bold dim")
        for column_title, size in [
            ("Rank", 4),
            ("ID", 6),
            ("Name", 16),
            ("Description", None),
            ("Score", 7),
            ("Gen", 3),
        ]:
            leaders.add_column(
                column_title,
                width=size,
                no_wrap=True,
                overflow="ellipsis",
                ratio=1 if size is None else None,
            )
        leaders = Table(
            *leaders.columns,
            *(replace(column, _cells=[]) for column in model.columns.values()),
            box=None,
            expand=True,
            padding=(0, 1),
            header_style="bold dim",
        )
        limit = max(0, height - 3)
        for rank, row in enumerate(model.leaders[:limit], 1):
            leaders.add_row(
                str(rank),
                _text(row["id"][:6]),
                _text(row["name"]),
                _text(row["description"]),
                _text(
                    f"{row['score']:.6g}"
                    if isinstance(row["score"], (int, float))
                    else row["score"]
                ),
                _text(row["generation"]),
                *[_text(row["extras"].get(key)) for key in model.columns],
                style=color(row["id"]),
            )
        return Panel(
            leaders,
            title=title,
            title_align="left",
            height=height,
            subtitle=_text(path),
            subtitle_align="left",
        )


class ProposalsPanel:
    """Newest completed proposals for the current generation."""

    def render(self, model, *, height, title, color):
        proposals = Table(
            Column("Time", width=8),
            Column("ID", width=6),
            Column("Name", width=16),
            Column("Description", ratio=1),
            box=None,
            expand=True,
            header_style="bold dim",
        )
        for column in proposals.columns:
            column.no_wrap, column.overflow = True, "ellipsis"
        for row in list(model.proposals)[: max(0, height - 3)]:
            proposals.add_row(
                _text(row["time"]),
                _text(row.get("policy_id", "—")[:6]),
                _text(row.get("name") or row["attempt_id"]),
                _text(row.get("description", "").replace("\n", " ")),
                style=color(row.get("policy_id")),
            )
        return Panel(
            proposals,
            title=title,
            title_align="left",
            height=height,
        )


class EvaluationsPanel:
    """Newest completed evaluations, including scores and durations."""

    def render(self, model, *, height, title, color):
        evaluations = Table(
            Column("Time", width=8),
            Column("ID", width=6),
            Column("Name", ratio=1),
            Column("Score", width=7),
            Column("Duration", width=8),
            box=None,
            expand=True,
            header_style="bold dim",
        )
        for column in evaluations.columns:
            column.no_wrap, column.overflow = True, "ellipsis"
        limit = max(0, height - 3)
        for row in list(model.evaluations)[:limit]:
            evaluations.add_row(
                _text(row["time"]),
                _text(row.get("policy_id", "—")[:6]),
                _text(row.get("name") or row["attempt_id"]),
                _text(f"{row['score']:.6g}" if row.get("score") is not None else None),
                _text(_duration(row.get("duration"))),
                style=color(row.get("policy_id")),
            )
        return Panel(
            evaluations,
            title=title,
            title_align="left",
            height=height,
        )


class EventLogPanel:
    """Bounded event text with worker activity above the visible tail."""

    def __init__(self):
        self.lines = deque(maxlen=100)

    def append(self, record, stamp):
        self.lines.extend(_event_lines(record, stamp))

    def render(self, workers, *, console, width, height):
        log_lines = Text("\n").join(self.lines).wrap(console, max(1, width - 4))
        worker_line = (
            [
                Text(
                    " | ".join(
                        f"{name} {p['active']}/{p['limit']} q{p['queued']} done{p['finished']}"
                        for name, p in workers.items()
                    ),
                    style="bold cyan",
                    no_wrap=True,
                    overflow="ellipsis",
                )
            ]
            if workers
            else []
        )
        return Panel(
            Group(*worker_line, *log_lines[-max(1, height - 2 - len(worker_line)) :]),
            title="EVENT LOG",
            title_align="left",
            height=height,
        )


class DashboardView:
    """Arrange the panels and share one policy palette across them."""

    def __init__(self, path, console):
        self.path, self.console = path, console
        self.leaderboard = LeaderboardPanel()
        self.proposals = ProposalsPanel()
        self.evaluations = EvaluationsPanel()
        self.events = EventLogPanel()
        self._colors, self._used_colors = {}, set()
        self._color_rng = SystemRandom()
        self._available_colors = list(_ID_COLORS)
        self._color_rng.shuffle(self._available_colors)

    def color(self, policy_id):
        if not policy_id or policy_id == "—":
            return None
        if policy_id not in self._colors:
            # Use distinct 256-color swatches first, then unused RGB colors for longer runs.
            if self._available_colors:
                color = self._available_colors.pop()
            else:
                while True:
                    rgb = tuple(self._color_rng.randrange(256) for _ in range(3))
                    color = "#%02x%02x%02x" % rgb
                    if (
                        max(rgb) >= 215
                        and max(rgb) - min(rgb) >= 95
                        and color not in self._used_colors
                    ):
                        break
            self._colors[policy_id] = color
            self._used_colors.add(color)
        return self._colors[policy_id]

    def render(self, model):
        width, height = self.console.size
        height = max(20, height)
        current = model.current_batch
        batch = model.batches.get(current, {})
        counts = model.batch_counts(current) if current else (0, None)
        population = None if counts[1] is None else counts[1] // 2
        # Reserve rows for this generation, capped at 25, with room for the event log.
        sections = 2 if width >= 100 else 3
        row_limit = max(0, (height - 4) // sections - 3)
        leader_height = min(model.leaderboard_size, 10, row_limit) + 3
        work_limit = max(0, (height - 4 - leader_height) // (sections - 1) - 3)
        panel_height = evaluation_height = (
            min(25, population if population is not None else 25, work_limit) + 3
        )
        work_height = panel_height * (sections - 1)
        log_height = height - leader_height - work_height
        elapsed = model.elapsed

        def bar(done, total):
            return Text.assemble(
                *(
                    (segment.text, segment.style)
                    for segment in self.console.render(
                        ProgressBar(total=total, completed=done, width=10)
                    )
                )
            )

        generations = sum(row["status"] == "completed" for row in model.batches.values())
        unit = "batches" if batch.get("label", "").startswith("Batch") else "generations"
        header = Text.assemble(
            _text(f"{model.environment} w/ {model.optimizer} "),
            bar(generations, model.total_generations),
            f" {generations}/{model.total_generations if model.total_generations is not None else '—'} {unit}",
            f" · {_duration(elapsed)} · {model.status}",
            " · resumed" if model.resumed else "",
            _text(f" ({model.reason})".replace("\n", " ")) if model.reason else "",
            no_wrap=True,
            overflow="ellipsis",
        )
        leaders_panel = self.leaderboard.render(
            model, height=leader_height, title=header, path=self.path, color=self.color
        )
        proposed, evaluated = model.work_counts(current) if current is not None else (0, 0)

        def title(name, done):
            return Text.assemble(
                _text(f"{name} · {batch.get('label', 'Waiting')} "),
                bar(done, population),
                f" {done}/{population if population is not None else '—'}",
            )

        proposal = self.proposals.render(
            model, height=panel_height, title=title("PROPOSALS", proposed), color=self.color
        )
        evaluation = self.evaluations.render(
            model, height=evaluation_height, title=title("EVALUATIONS", evaluated), color=self.color
        )
        if width >= 100:
            work = Table.grid(expand=True, padding=(0, 1))
            work.add_column(ratio=1)
            work.add_column(ratio=1)
            work.add_row(proposal, evaluation)
        else:
            work = Group(proposal, evaluation)
        logs = self.events.render(
            model.workers, console=self.console, width=width, height=log_height
        )
        return Group(leaders_panel, work, logs)


def show_scores(policies, run, console, *, seeds=None):
    table = Table("Policy", "Description", "Score")
    seeds = None if seeds is None else tuple(dict.fromkeys(seeds))
    for policy in policies:
        scores = run.scores(policy)
        values = list(scores.values()) if seeds is None else [scores.get(seed) for seed in seeds]
        score = (
            f"{fmean(values):.1f}"
            if values and all(v is not None for v in values)
            else "unfinished"
        )
        table.add_row(Text(policy.name), Text(policy.description), score)
    console.print(table)
