"""Common search progress events and policy score display."""

import asyncio
import logging
import math
import re
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from random import SystemRandom
from statistics import fmean
from threading import RLock
from time import monotonic

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Column, Table
from rich.text import Text

_TERMINAL = {"evaluated", "discarded", "failed"}
_STATUSES = _TERMINAL | {
    "planned",
    "generating",
    "generated",
    "queued",
    "evaluating",
    "repairing",
    "cancelled",
}
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


class RunDisplay:
    """Consume domain log records; neither runs nor optimizers push UI snapshots."""

    def __init__(self, path, console):
        self.path, self.console = path, console
        self.environment = self.optimizer = "—"
        self.columns, self.batches, self.candidates = {}, {}, {}
        self.leaders = []
        self.leaderboard_size = 10
        self.proposals, self.evaluations = deque(maxlen=50), deque(maxlen=50)
        self.logs = deque(maxlen=100)
        self._colors, self._used_colors = {}, set()
        self._color_rng = SystemRandom()
        self._available_colors = list(_ID_COLORS)
        self._color_rng.shuffle(self._available_colors)
        self.started = monotonic()
        self.finished = None
        self.status = "running"
        self.reason = ""
        self.resumed = self.active = False
        self.total = self.total_generations = None
        self.archived = 0
        self.restored_units = self.prior_completed = 0
        self.lock = RLock()

    def _color(self, policy_id):
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

    def _log_candidate(self, row, previous, stamp):
        if row.get("restored") or all(
            row.get(key) == previous.get(key) for key in ("status", "revision", "policy_id")
        ):
            return
        if row["status"] == "evaluated":
            self.evaluations.appendleft(dict(row, time=stamp[11:19]))
        elif row["status"] == "generated":
            self.proposals.appendleft(dict(row, time=stamp[11:19]))

    @property
    def completed(self):
        return (
            self.archived
            + max(0, self.prior_completed - self.restored_units)
            + sum(
                int(row["proposal_done"]) + int(row["status"] in _TERMINAL)
                for row in self.candidates.values()
            )
        )

    @property
    def current_batch(self):
        return next(
            (key for key, row in self.batches.items() if row["status"] == "running"),
            next(reversed(self.batches), None),
        )

    def batch_counts(self, batch_id):
        batch = self.batches[batch_id]
        if "completed" in batch:
            return batch["completed"], batch["total"]
        return sum(
            int(r["proposal_done"]) + int(r["status"] in _TERMINAL)
            for (bid, _), r in self.candidates.items()
            if bid == batch_id
        ), batch["total"]

    def consume(self, record):
        stamp = (
            datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        with self.lock:
            self.logs.extend(_event_lines(record, stamp))
            payload = getattr(record, "progress", None)
            if payload is None:
                return
            try:
                self._consume(payload, stamp)
            except (KeyError, TypeError, ValueError) as exc:
                diagnostic = f"Invalid progress record: {exc}"
                self.logs.extend(
                    _event_lines(
                        logging.makeLogRecord(dict(msg=diagnostic, levelno=logging.ERROR)), stamp
                    )
                )
                return diagnostic

    def _consume(self, p, stamp=""):
        kind = p["kind"]
        if self.finished is not None and kind != "search_started":
            return
        if kind == "environment":
            self.environment = str(p["name"])
        elif kind in ("search_started", "batch_started"):
            total = p["total_candidates"]
            if total is not None and (type(total) is not int or total < 0):
                raise ValueError("total_candidates must be nonnegative or None")
            generations = p.get("total_generations")
            if generations is not None and (type(generations) is not int or generations < 0):
                raise ValueError("total_generations must be nonnegative or None")
            leaderboard_size = p.get("leaderboard_size", self.leaderboard_size)
            if type(leaderboard_size) is not int or leaderboard_size < 0:
                raise ValueError("leaderboard_size must be nonnegative")
            prior = p.get("completed_candidates_before", 0)
            if type(prior) is not int or prior < 0 or (total is not None and prior > total):
                raise ValueError("invalid historical completion count")
            columns = p.get("columns", self.columns)
            if not isinstance(columns, dict) or any(
                not isinstance(k, str)
                or k.lower() in {"rank", "id", "name", "description", "score", "gen", "generation"}
                or not isinstance(v, Column)
                for k, v in columns.items()
            ):
                raise ValueError("invalid custom columns")
            if kind == "batch_started":
                key = p["batch_id"]
                if not isinstance(key, str) or not isinstance(p["label"], str):
                    raise ValueError("batch ID and label must be strings")
                if key not in self.batches:
                    self.proposals.clear()
                    self.evaluations.clear()
                self.batches.setdefault(
                    key,
                    dict(
                        total=None if total is None else total * 2,
                        label=p["label"],
                        status="running",
                    ),
                )
                if self.batches[key]["total"] is None and total is not None:
                    self.batches[key]["total"] = total * 2
            else:
                self.finished, self.status, self.reason = None, "running", ""
                self.total = None if total is None else total * 2
                self.total_generations = generations
                self.prior_completed = max(self.prior_completed, prior * 2)
                self.resumed = bool(p.get("resumed", False))
            self.optimizer = str(p.get("optimizer", self.optimizer))
            self.columns = columns.copy()
            self.leaderboard_size = leaderboard_size
            self.active = True
            if kind == "batch_started" and total == 0:
                self._consume(dict(kind="batch_finished", batch_id=key, status="completed"))
        elif kind == "candidate":
            key = (p["batch_id"], p["attempt_id"])
            if not all(isinstance(v, str) for v in key):
                raise ValueError("candidate identity must be strings")
            if type(p["revision"]) is not int or p["revision"] < 0 or p["status"] not in _STATUSES:
                raise ValueError("invalid revision or status")
            if type(p["proposal_done"]) is not bool:
                raise ValueError("proposal_done must be boolean")
            for field in ("policy_id", "name", "description"):
                if field in p and not isinstance(p[field], str):
                    raise ValueError(f"{field} must be text")
            for field in ("score", "duration"):
                value = p.get(field)
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    raise ValueError(f"invalid {field}")
            batch = self.batches[key[0]]
            if batch["status"] != "running":
                return
            previous = self.candidates.get(key, {})
            if p["revision"] < previous.get("revision", 0):
                return
            if previous.get("status") in _TERMINAL and p["revision"] == previous["revision"]:
                return
            row = {**previous, **p}
            for field in ("name", "description", "error"):
                if isinstance(row.get(field), str):
                    row[field] = row[field][:8192]
            row["proposal_done"] = p["proposal_done"] or previous.get("proposal_done", False)
            if p["status"] == "evaluating":
                row.setdefault("evaluation_started", monotonic())
            if p["status"] in _TERMINAL:
                row["proposal_done"] = True
                if not p.get("restored", False):
                    if "evaluation_started" in row:
                        row.setdefault("duration", monotonic() - row["evaluation_started"])
            self.candidates[key] = row
            self._log_candidate(row, previous, stamp)
            done, total = self.batch_counts(key[0])
            if total is not None and done == total:
                self._consume(
                    dict(
                        kind="batch_finished",
                        batch_id=key[0],
                        status="completed",
                        restored=all(
                            r.get("restored", False)
                            for (bid, _), r in self.candidates.items()
                            if bid == key[0]
                        ),
                    )
                )
        elif kind == "leaderboard":
            rows = p["rows"]
            if not isinstance(rows, list) or any(
                not isinstance(row, dict)
                or not {"id", "name", "description", "score", "generation", "extras"} <= row.keys()
                for row in rows
            ):
                raise ValueError("invalid leaderboard rows")
            if any(not isinstance(row["extras"], dict) for row in rows):
                raise ValueError("invalid leaderboard extras")
            if any(
                not isinstance(row[key], str)
                for row in rows
                for key in ("id", "name", "description")
            ):
                raise ValueError("leaderboard identity and descriptions must be text")
            self.leaders = [dict(row) for row in rows[:100]]
        elif kind == "batch_finished":
            batch_id = p["batch_id"]
            batch = self.batches[batch_id]
            status = p["status"]
            if status not in {"completed", "stopped", "failed", "cancelled"}:
                raise ValueError("invalid batch outcome")
            if batch["status"] != "running":
                return
            batch["completed"] = self.batch_counts(batch_id)[0]
            rows = [row for (bid, _), row in self.candidates.items() if bid == batch_id]
            batch["proposed"] = sum(row["proposal_done"] for row in rows)
            batch["evaluated"] = sum(row["status"] in _TERMINAL for row in rows)
            batch["status"] = status
            self.archived += batch["completed"]
            if p.get("restored", False):
                self.restored_units += batch["completed"]
            self.candidates = {
                key: row for key, row in self.candidates.items() if key[0] != batch_id
            }
        elif kind == "search_finished":
            status = p["status"]
            if status not in {"completed", "stopped", "failed", "cancelled"}:
                raise ValueError("invalid search outcome")
            self.status, self.reason = status, str(p["reason"])
            self.finished = monotonic()

    def render(self):
        with self.lock:
            return self._render()

    def _render(self):
        width, height = self.console.size
        height = max(20, height)
        current = self.current_batch
        batch = self.batches.get(current, {})
        counts = self.batch_counts(current) if current else (0, None)
        population = None if counts[1] is None else counts[1] // 2
        # Reserve rows for this generation, capped at 25, with room for the event log.
        sections = 2 if width >= 100 else 3
        row_limit = max(0, (height - 4) // sections - 3)
        leader_height = min(self.leaderboard_size, 10, row_limit) + 3
        work_limit = max(0, (height - 4 - leader_height) // (sections - 1) - 3)
        panel_height = evaluation_height = (
            min(25, population if population is not None else 25, work_limit) + 3
        )
        work_height = panel_height * (sections - 1)
        log_height = height - leader_height - work_height
        elapsed = (self.finished or monotonic()) - self.started

        def bar(done, total):
            return Text.assemble(
                *(
                    (segment.text, segment.style)
                    for segment in self.console.render(
                        ProgressBar(total=total, completed=done, width=10)
                    )
                )
            )

        generations = sum(row["status"] == "completed" for row in self.batches.values())
        unit = "batches" if batch.get("label", "").startswith("Batch") else "generations"
        header = Text.assemble(
            _text(f"{self.environment} w/ {self.optimizer} "),
            bar(generations, self.total_generations),
            f" {generations}/{self.total_generations if self.total_generations is not None else '—'} {unit}",
            f" · {_duration(elapsed)} · {self.status}",
            " · resumed" if self.resumed else "",
            _text(f" ({self.reason})".replace("\n", " ")) if self.reason else "",
            no_wrap=True,
            overflow="ellipsis",
        )
        leaders = Table(box=None, expand=True, padding=(0, 1), header_style="bold dim")
        for title, size in [
            ("Rank", 4),
            ("ID", 6),
            ("Name", 16),
            ("Description", None),
            ("Score", 7),
            ("Gen", 3),
        ]:
            leaders.add_column(
                title,
                width=size,
                no_wrap=True,
                overflow="ellipsis",
                ratio=1 if size is None else None,
            )
        leaders = Table(
            *leaders.columns,
            *(replace(column, _cells=[]) for column in self.columns.values()),
            box=None,
            expand=True,
            padding=(0, 1),
            header_style="bold dim",
        )
        limit = max(0, leader_height - 3)
        for rank, row in enumerate(self.leaders[:limit], 1):
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
                *[_text(row["extras"].get(key)) for key in self.columns],
                style=self._color(row["id"]),
            )
        leaders_panel = Panel(
            leaders,
            title=header,
            title_align="left",
            height=leader_height,
            subtitle=_text(self.path),
            subtitle_align="left",
        )
        rows = [row for (bid, _), row in self.candidates.items() if bid == current]
        proposed = batch.get("proposed", sum(row["proposal_done"] for row in rows))
        evaluated = batch.get("evaluated", sum(row["status"] in _TERMINAL for row in rows))

        def title(name, done):
            return Text.assemble(
                _text(f"{name} · {batch.get('label', 'Waiting')} "),
                bar(done, population),
                f" {done}/{population if population is not None else '—'}",
            )

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
        for row in list(self.proposals)[: max(0, panel_height - 3)]:
            proposals.add_row(
                _text(row["time"]),
                _text(row.get("policy_id", "—")[:6]),
                _text(row.get("name") or row["attempt_id"]),
                _text(row.get("description", "").replace("\n", " ")),
                style=self._color(row.get("policy_id")),
            )
        proposal = Panel(
            proposals,
            title=title("PROPOSALS", proposed),
            title_align="left",
            height=panel_height,
        )
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
        limit = max(0, evaluation_height - 3)
        for row in list(self.evaluations)[:limit]:
            evaluations.add_row(
                _text(row["time"]),
                _text(row.get("policy_id", "—")[:6]),
                _text(row.get("name") or row["attempt_id"]),
                _text(f"{row['score']:.6g}" if row.get("score") is not None else None),
                _text(_duration(row.get("duration"))),
                style=self._color(row.get("policy_id")),
            )
        evaluation = Panel(
            evaluations,
            title=title("EVALUATIONS", evaluated),
            title_align="left",
            height=evaluation_height,
        )
        if width >= 100:
            work = Table.grid(expand=True, padding=(0, 1))
            work.add_column(ratio=1)
            work.add_column(ratio=1)
            work.add_row(proposal, evaluation)
        else:
            work = Group(proposal, evaluation)
        log_lines = Text("\n").join(self.logs).wrap(self.console, max(1, width - 4))
        logs = Panel(
            Group(*log_lines[-(log_height - 2) :]),
            title="EVENT LOG",
            title_align="left",
            height=log_height,
        )
        return Group(leaders_panel, work, logs)

    def close(self):
        if self.finished is None:
            self.finished = monotonic()
            self.status = "stopped"
            self.reason = "run closed"


_current_run = ContextVar("rsikit_log_run", default=None)
_bindings = []
_logger_settings = {}
_live_streams = set()


class ProgressHandler(logging.Handler):
    def emit(self, record):
        binding = _current_run.get()
        if binding is None or binding["closed"]:
            return
        display = binding["display"]
        stamp = (
            datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        message = f"{stamp} {record.levelname} [{record.name}] {_message(record)}"
        binding["file"].write(message + "\n")
        binding["file"].flush()
        diagnostic = display.consume(record)
        if diagnostic:
            binding["file"].write(diagnostic + "\n")
            binding["file"].flush()
            message += "\n" + diagnostic
        if not display.active:
            return
        if binding["live"] is None and display.finished is None and display.console.is_terminal:
            stream = display.console.file
            try:
                stream_key = stream.fileno()
            except (AttributeError, OSError):
                stream_key = id(stream)
            if stream_key not in _live_streams:
                live = Live(
                    console=display.console, get_renderable=display.render, refresh_per_second=4
                )
                live.start()
                binding["live"], binding["stream_key"] = live, stream_key
                _live_streams.add(stream_key)
        if binding["live"] is None:
            display.console.print(_text(message), soft_wrap=True)
        elif display.finished is not None:
            binding["live"].stop()
            _live_streams.discard(binding["stream_key"])
            binding["live"] = None


_run_handler = ProgressHandler()


@contextmanager
def bind_run(path, console=None):
    """Bind logging to the Run's task context; lazily acquire a terminal display."""
    display = RunDisplay(path, console or Console())
    binding = dict(
        display=display,
        file=(path / "run.log").open("a", encoding="utf-8"),
        live=None,
        closed=False,
        stream_key=None,
    )
    if not _bindings:
        for name in ("rsikit", "research"):
            logger = logging.getLogger(name)
            _logger_settings[name] = logger.level
            logger.addHandler(_run_handler)
            logger.setLevel(logging.INFO)
    _bindings.append(binding)
    token = _current_run.set(binding)
    try:
        yield display
    except BaseException as exc:
        if display.active:
            cancelled = isinstance(exc, asyncio.CancelledError)
            logging.getLogger("rsikit").log(
                logging.INFO if cancelled else logging.ERROR,
                "Run cancelled" if cancelled else "Run failed",
                exc_info=not cancelled,
                extra={
                    "progress": dict(
                        kind="search_finished",
                        status="cancelled" if cancelled else "failed",
                        reason="cancelled" if cancelled else str(exc),
                    )
                },
            )
        raise
    finally:
        binding["closed"] = True
        try:
            display.close()
            if binding["live"] is not None:
                binding["live"].stop()
                _live_streams.discard(binding["stream_key"])
        finally:
            binding["file"].close()
            _current_run.reset(token)
            _bindings.remove(binding)
            if not _bindings:
                for name, level in _logger_settings.items():
                    logger = logging.getLogger(name)
                    logger.removeHandler(_run_handler)
                    logger.setLevel(level)
                _logger_settings.clear()


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
