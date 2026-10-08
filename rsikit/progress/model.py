"""Validated progress events and run accounting, independent of rendering."""

import math
from collections import deque
from time import monotonic

from rich.table import Column

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


class ProgressState:
    """Track accepted transitions and completion history for one run."""

    def __init__(self):
        self.environment = self.optimizer = "—"
        self.columns, self.batches, self.candidates = {}, {}, {}
        self.leaders = []
        self.workers = {}
        self.leaderboard_size = 10
        self.proposals, self.evaluations = deque(maxlen=50), deque(maxlen=50)
        self.started = monotonic()
        self.finished = None
        self.status = "running"
        self.reason = ""
        self.resumed = self.active = False
        self.total = self.total_generations = None
        self.archived = 0
        self.restored_units = self.prior_completed = 0

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

    @property
    def elapsed(self):
        return (self.finished or monotonic()) - self.started

    def work_counts(self, batch_id):
        batch = self.batches[batch_id]
        if "proposed" in batch:
            return batch["proposed"], batch["evaluated"]
        rows = [row for (bid, _), row in self.candidates.items() if bid == batch_id]
        return sum(row["proposal_done"] for row in rows), sum(
            row["status"] in _TERMINAL for row in rows
        )

    def apply(self, p, stamp=""):
        kind = p["kind"]
        if self.finished is not None and kind != "search_started":
            return
        if kind == "environment":
            self.environment = str(p["name"])
        elif kind == "workers":
            self._update_workers(p)
        elif kind in ("search_started", "batch_started"):
            self._start(p)
        elif kind == "candidate":
            self._update_candidate(p, stamp)
        elif kind == "leaderboard":
            self._update_leaderboard(p)
        elif kind == "batch_finished":
            self._finish_batch(p)
        elif kind == "search_finished":
            self._finish_search(p)

    def _update_workers(self, p):
        pools = p["pools"]
        if not isinstance(pools, dict) or any(
            not isinstance(name, str)
            or not isinstance(pool, dict)
            or any(
                type(pool.get(k)) is not int or pool[k] < 0
                for k in ("active", "limit", "queued", "finished")
            )
            or pool["active"] > pool["limit"]
            for name, pool in pools.items()
        ):
            raise ValueError("invalid worker counts")
        self.workers = {name: dict(pool) for name, pool in pools.items()}

    def _start(self, p):
        kind = p["kind"]
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
            self._finish_batch(dict(kind="batch_finished", batch_id=key, status="completed"))

    def _update_candidate(self, p, stamp):
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
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
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
            self._finish_batch(
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

    def _update_leaderboard(self, p):
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
            not isinstance(row[key], str) for row in rows for key in ("id", "name", "description")
        ):
            raise ValueError("leaderboard identity and descriptions must be text")
        self.leaders = [dict(row) for row in rows[:100]]

    def _finish_batch(self, p):
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
        self.candidates = {key: row for key, row in self.candidates.items() if key[0] != batch_id}

    def _finish_search(self, p):
        status = p["status"]
        if status not in {"completed", "stopped", "failed", "cancelled"}:
            raise ValueError("invalid search outcome")
        self.status, self.reason = status, str(p["reason"])
        self.finished = monotonic()

    def close(self):
        if self.finished is None:
            self.finished = monotonic()
            self.status = "stopped"
            self.reason = "run closed"
