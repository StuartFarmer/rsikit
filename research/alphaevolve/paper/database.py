"""Persistent island/MAP-Elites archive; selection rules are explicit local choices."""

import ast
import hashlib
import json
import math
import sqlite3
from dataclasses import field
from numbers import Real

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from rsikit.policy import PolicyDefinition

from .evaluation import FiniteNumber, NamedValues, SeedScores


@dataclass(
    frozen=True, config=ConfigDict(strict=True, extra="forbid", revalidate_instances="always")
)
class Candidate:
    policy: PolicyDefinition
    score: FiniteNumber
    metrics: NamedValues
    features: NamedValues
    feedback: str = ""
    seed_scores: SeedScores = field(default_factory=dict)


_CANDIDATE = TypeAdapter(Candidate)


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class Database:
    """Keep one winner per maximized metric and descriptor cell on each island.

    Descriptor ranges are inclusive; overflow clamps to the nearest edge bin.
    Ties keep the earlier winner. With no descriptors, each metric has one cell.
    All unique evaluated programs survive in history, including displaced elites.
    Syntax-equivalent registrations reuse the first evaluation and its lineage.
    This synchronous connection is owned by one controller/event-loop thread.
    """

    def __init__(
        self,
        path=":memory:",
        *,
        islands=4,
        objective="reward",
        features=None,
        elite_fraction=0.2,
        exploration=0.3,
    ):
        if isinstance(islands, bool) or not isinstance(islands, int) or islands < 1:
            raise ValueError("islands must be a positive integer")
        if not isinstance(objective, str) or not objective:
            raise ValueError("objective must be a nonempty metric name")
        if not _finite(elite_fraction) or not 0 < elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not _finite(exploration) or not 0 <= exploration <= 1:
            raise ValueError("exploration must be in [0, 1]")
        self.features = {}
        if features is not None and not isinstance(features, dict):
            raise ValueError("features must map descriptor names to (low, high, bins)")
        for name, bounds in (features or {}).items():
            if not isinstance(name, str) or not name or not isinstance(bounds, (list, tuple)):
                raise ValueError("invalid descriptor configuration")
            if len(bounds) != 3:
                raise ValueError("descriptor bounds must contain (low, high, bins)")
            low, high, bins = bounds
            if (
                not _finite(low)
                or not _finite(high)
                or low >= high
                or not math.isfinite(high - low)
                or isinstance(bins, bool)
                or not isinstance(bins, int)
                or bins < 1
            ):
                raise ValueError("descriptor bounds must be finite, increasing, with positive bins")
            self.features[name] = (float(low), float(high), bins)
        self.islands = islands
        self.objective = objective
        self.elite_fraction = elite_fraction
        self.exploration = exploration
        self._policies = {}
        config = json.dumps(
            {
                "version": 1,
                "islands": islands,
                "objective": objective,
                "features": self.features,
                "elite_fraction": elite_fraction,
                "exploration": exploration,
            },
            sort_keys=True,
        )
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS programs (
                    id TEXT PRIMARY KEY, syntax TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL, source TEXT NOT NULL, description TEXT NOT NULL,
                    score REAL NOT NULL, metrics TEXT NOT NULL, features TEXT NOT NULL,
                    feedback TEXT NOT NULL, seed_scores TEXT NOT NULL,
                    island INTEGER NOT NULL, parent_id TEXT REFERENCES programs(id)
                );
                CREATE TABLE IF NOT EXISTS cells (
                    island INTEGER NOT NULL, niche TEXT NOT NULL, metric TEXT NOT NULL,
                    program_id TEXT NOT NULL REFERENCES programs(id), value REAL NOT NULL,
                    PRIMARY KEY (island, niche, metric)
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            with self._connection:
                stored = self._connection.execute(
                    "SELECT value FROM metadata WHERE key = 'config'"
                ).fetchone()
                if stored is not None and json.loads(stored[0]) != json.loads(config):
                    raise ValueError("database configuration differs from its saved configuration")
                self._connection.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('config', ?)", (config,)
                )
        except BaseException:
            self._connection.close()
            raise

    def _island(self, island):
        if (
            isinstance(island, bool)
            or not isinstance(island, int)
            or not 0 <= island < self.islands
        ):
            raise ValueError("island index is out of range")

    def _validate(self, candidate):
        _CANDIDATE.validate_python(candidate)
        if candidate.metrics.get(self.objective) != candidate.score:
            raise ValueError("score must equal the required objective metric")
        if set(candidate.features) != set(self.features):
            raise ValueError("descriptors must match the configured schema exactly")
        schema = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'metrics'"
        ).fetchone()
        if schema is not None and sorted(candidate.metrics) != json.loads(schema[0]):
            raise ValueError("metrics must match the first evaluated program's schema")

    def validate(self, candidate: Candidate, island, parent_id=None) -> None:
        """Check registration inputs against saved configuration without changing state."""
        self._island(island)
        self._validate(candidate)
        if (
            parent_id is not None
            and self._connection.execute(
                "SELECT 1 FROM programs WHERE id = ?", (parent_id,)
            ).fetchone()
            is None
        ):
            raise ValueError("parent_id must identify an evaluated program")

    def _niche(self, candidate):
        bins = []
        for name, (low, high, count) in sorted(self.features.items()):
            value = min(high, max(low, candidate.features[name]))
            bins.append(min(count - 1, int((value - low) / (high - low) * count)))
        return json.dumps(bins)

    def _place(self, candidate, island):
        changed = False
        for metric, value in candidate.metrics.items():
            cursor = self._connection.execute(
                """
                INSERT INTO cells VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(island, niche, metric) DO UPDATE SET
                    program_id = excluded.program_id, value = excluded.value
                WHERE excluded.value > cells.value
            """,
                (island, self._niche(candidate), metric, candidate.policy.id, value),
            )
            changed = bool(cursor.rowcount) or changed
        return changed

    def register(self, candidate: Candidate, island, parent_id=None) -> Candidate:
        """Register without executing source; duplicate syntax keeps its canonical evaluation."""
        self.validate(candidate, island, parent_id)
        source = candidate.policy.source
        syntax = hashlib.sha256(
            ast.dump(ast.parse(source), include_attributes=False).encode()
        ).hexdigest()
        with self._connection:
            existing = self._connection.execute(
                "SELECT * FROM programs WHERE syntax = ?", (syntax,)
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    "INSERT INTO programs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        candidate.policy.id,
                        syntax,
                        candidate.policy.name,
                        source,
                        getattr(candidate.policy, "description", ""),
                        float(candidate.score),
                        json.dumps({k: float(v) for k, v in candidate.metrics.items()}),
                        json.dumps({k: float(v) for k, v in candidate.features.items()}),
                        candidate.feedback,
                        json.dumps({k: float(v) for k, v in candidate.seed_scores.items()}),
                        island,
                        parent_id,
                    ),
                )
                self._connection.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('metrics', ?)",
                    (json.dumps(sorted(candidate.metrics)),),
                )
            else:
                candidate = self._decode(existing)
            self._place(candidate, island)
        self._policies[candidate.policy.id] = candidate.policy
        return candidate

    def _decode(self, row):
        if row["id"] not in self._policies:
            self._policies[row["id"]] = PolicyDefinition.from_text(
                row["source"], name=row["name"], description=row["description"]
            )
        return Candidate(
            self._policies[row["id"]],
            row["score"],
            json.loads(row["metrics"]),
            json.loads(row["features"]),
            row["feedback"],
            {int(k): v for k, v in json.loads(row["seed_scores"]).items()},
        )

    def members(self, island) -> list[Candidate]:
        """Return the union of the island's metric/niche winners."""
        self._island(island)
        return [
            self._decode(row)
            for row in self._connection.execute(
                """
            SELECT * FROM programs WHERE id IN (
                SELECT program_id FROM cells WHERE island = ?
            ) ORDER BY rowid
        """,
                (island,),
            )
        ]

    def all(self) -> list[Candidate]:
        return [
            self._decode(row)
            for row in self._connection.execute("SELECT * FROM programs ORDER BY rowid")
        ]

    @property
    def champions(self) -> list[Candidate | None]:
        return [
            max(self.members(i), key=lambda c: c.score, default=None) for i in range(self.islands)
        ]

    @property
    def best(self) -> Candidate | None:
        row = self._connection.execute(
            "SELECT * FROM programs ORDER BY score DESC, rowid LIMIT 1"
        ).fetchone()
        return self._decode(row) if row is not None else None

    def sample(self, rng, inspirations=3) -> tuple[int, Candidate, list[Candidate]]:
        """Choose a populated island uniformly, then explore or exploit a random metric.

        Exploitation uses descending-rank weights within ceil(elite_fraction * size).
        Inspirations prioritize different descriptor cells across all islands, then
        fill from remaining retained programs; the parent is always excluded.
        """
        if isinstance(inspirations, bool) or not isinstance(inspirations, int) or inspirations < 0:
            raise ValueError("inspirations must be a nonnegative integer")
        populations = [(i, self.members(i)) for i in range(self.islands)]
        populated = [(i, members) for i, members in populations if members]
        if not populated:
            raise ValueError("cannot sample an empty population")
        island, members = rng.choice(populated)
        if rng.random() < self.exploration:
            parent = rng.choice(members)
        else:
            metric = rng.choice(sorted(members[0].metrics))
            elite = sorted(members, key=lambda c: c.metrics[metric], reverse=True)
            elite = elite[: max(1, math.ceil(len(elite) * self.elite_fraction))]
            parent = rng.choices(elite, weights=range(len(elite), 0, -1), k=1)[0]
        others = {
            c.policy.id: c
            for _, group in populated
            for c in group
            if c.policy.id != parent.policy.id
        }
        remaining = list(others.values())
        rng.shuffle(remaining)
        selected = []
        seen = {self._niche(parent)}
        for candidate in remaining:
            niche = self._niche(candidate)
            if niche not in seen and len(selected) < inspirations:
                selected.append(candidate)
                seen.add(niche)
        chosen = {c.policy.id for c in selected}
        selected.extend(c for c in remaining if c.policy.id not in chosen)
        return island, parent, selected[:inspirations]

    def migrate(self, rng, count=1) -> list[dict]:
        """Attempt count uniform member transfers; replace only improved target cells."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("migration count must be a nonnegative integer")
        events = []
        if self.islands < 2:
            return events
        with self._connection:
            for _ in range(count):
                sources = [i for i in range(self.islands) if self.members(i)]
                if not sources:
                    break
                source = rng.choice(sources)
                target = rng.choice([i for i in range(self.islands) if i != source])
                candidate = rng.choice(self.members(source))
                if self._place(candidate, target):
                    events.append(
                        {"source": source, "target": target, "policy_id": candidate.policy.id}
                    )
        return events

    def save_state(self, key, value):
        with self._connection:
            self._connection.execute(
                "INSERT INTO state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, allow_nan=False)),
            )

    def load_state(self, key, default=None):
        row = self._connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else default

    def close(self):
        self._connection.close()
