"""Persist the configuration, checkpoints and outputs of one optimizer run."""

import fcntl
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Column, inspect
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .episode import Episode
from .policy import PolicyDefinition
from .progress import bind_run


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64] or "policy"


class _Settings(SQLModel, table=True):
    __tablename__ = "settings"
    name: str = Field(primary_key=True)
    export: bool = True


class _StoredPolicy(SQLModel, table=True):
    __tablename__ = "policy"
    id: str = Field(primary_key=True)
    name: str
    description: str = ""
    implementation: str
    scores: dict[str, float | None] = Field(default_factory=dict, sa_column=Column(JSON))


class Run:
    """A directory containing policies and scores. Unfinished scores are None."""

    @classmethod
    def create(
        cls,
        *,
        name: str,
        path: str | Path | None = None,
        export: bool = True,
        console=None,
    ) -> "Run":
        settings = _Settings(name=name, export=export)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            Path(path)
            if path is not None
            else Path("runs") / f"{_slug(name)}-{stamp}-{uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory, settings, console=console)

    @classmethod
    def open(cls, path: str | Path, *, console=None) -> "Run":
        directory = Path(path)
        if not (directory / "run.sqlite").is_file():
            raise FileNotFoundError(directory / "run.sqlite")
        return cls(directory, console=console)

    def __init__(
        self,
        path: Path,
        settings: _Settings | None = None,
        *,
        console=None,
    ):
        self.path = path.resolve()
        self._console = console
        self._progress_scope = None
        self._lock = (self.path / ".lock").open("a")
        self._engine = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._engine = create_engine(f"sqlite:///{self.path / 'run.sqlite'}")
            if settings is not None:
                SQLModel.metadata.create_all(
                    self._engine, tables=[_Settings.__table__, _StoredPolicy.__table__]
                )
            if "description" not in {
                column["name"] for column in inspect(self._engine).get_columns("policy")
            }:
                with self._engine.begin() as connection:
                    connection.exec_driver_sql(
                        "ALTER TABLE policy ADD COLUMN description TEXT NOT NULL DEFAULT ''"
                    )
            with Session(self._engine) as session:
                if settings is not None:
                    session.add(settings)
                    session.commit()
                self._settings = session.exec(select(_Settings)).one()
                for stored in session.exec(select(_StoredPolicy)):
                    self._export(stored)
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return self._settings.name

    def __enter__(self) -> "Run":
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        if self._progress_scope is None:
            self._progress_scope = bind_run(self.path, self._console)
            self._progress_scope.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            if self._progress_scope is not None:
                scope, self._progress_scope = self._progress_scope, None
                scope.__exit__(*exc)
        finally:
            self.close()

    async def __aenter__(self) -> "Run":
        return self.__enter__()

    async def __aexit__(self, *exc):
        self.__exit__(*exc)

    async def aclose(self) -> None:
        self.close()

    def close(self) -> None:
        if self._progress_scope is not None:
            scope, self._progress_scope = self._progress_scope, None
            scope.__exit__(None, None, None)
        if self._engine is not None:
            self._engine.dispose()
        self._lock.close()

    def save(self, *records: SQLModel) -> None:
        """Save typed records together, inferring and creating their tables.

        Primary keys identify records to update. Database-generated primary keys
        are copied back onto the supplied objects after a successful commit.
        """
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        tables = list(dict.fromkeys(record.__table__ for record in records))
        SQLModel.metadata.create_all(self._engine, tables=tables)
        with Session(self._engine, expire_on_commit=False) as db:
            saved = [db.merge(record) for record in records]
            db.commit()
            for record, stored in zip(records, saved):
                mapper = inspect(record).mapper
                for column in mapper.primary_key:
                    key = mapper.get_property_by_column(column).key
                    setattr(record, key, getattr(stored, key))

    def database(self) -> Session:
        """Open a native SQLModel session for queries against this Run's SQLite database."""
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        return Session(self._engine)

    def _export(self, policy: _StoredPolicy) -> None:
        if self._settings.export:
            destination = self.path / "exports" / f"{policy.id}_{_slug(policy.name)}.py"
            if not destination.exists():
                destination.parent.mkdir(exist_ok=True)
                PolicyDefinition.from_text(
                    policy.implementation, name=policy.name, description=policy.description
                ).to_file(destination)

    def policies(self) -> list[PolicyDefinition]:
        with self.database() as session:
            return [
                PolicyDefinition.from_text(
                    row.implementation, name=row.name, description=row.description
                )
                for row in session.exec(select(_StoredPolicy))
            ]

    def scores(self, policy: PolicyDefinition) -> dict[int, float | None]:
        with self.database() as session:
            row = session.get(_StoredPolicy, policy.id)
            return {} if row is None else {int(seed): score for seed, score in row.scores.items()}

    def save_policy(
        self, policy: PolicyDefinition, *, scores: dict[int, float | None] | None = None
    ) -> None:
        """Persist source and optimizer-supplied measurements without evaluating anything."""
        if scores is not None and any(
            type(seed) is not int
            or (score is not None and (type(score) not in (int, float) or not math.isfinite(score)))
            for seed, score in scores.items()
        ):
            raise ValueError("Scores require integer seeds and finite numbers or None")
        with self.database() as db:
            stored = db.get(_StoredPolicy, policy.id)
            if stored is None:
                stored = _StoredPolicy(
                    id=policy.id,
                    name=policy.name,
                    description=policy.description,
                    implementation=policy.source,
                )
            elif (stored.name, stored.implementation) != (policy.name, policy.source):
                raise ValueError("A stored policy cannot change under the same ID")
            stored.scores = {**stored.scores, **{str(k): v for k, v in (scores or {}).items()}}
            db.add(stored)
            db.commit()
            self._export(stored)

    def _output_path(self, relative: Path) -> Path:
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        destination = (self.path / relative).resolve()
        if not destination.is_relative_to(self.path):
            raise ValueError("Output path must stay inside the run")
        return destination

    def save_episode(self, policy: PolicyDefinition, seed: int, episode: Episode) -> None:
        """Save raw evidence and artifacts; fitness is supplied separately by the optimizer."""
        if type(seed) is not int:
            raise ValueError("Episode seed must be an integer")
        if not isinstance(episode, Episode):
            raise ValueError("Expected an Episode")
        data = json.dumps(episode.encode(), allow_nan=False).encode()
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Episode exceeds 64 MiB")
        self.save_policy(policy)
        root = self._output_path(Path("artifacts") / policy.id / str(seed))
        outputs = []
        for name, value in episode.artifacts.items():
            destination = (root / name).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Artifact path must stay inside the evaluation directory")
            outputs.append((destination, value))
        outputs.append((self._output_path(Path("episodes") / policy.id / f"{seed}.json"), data))
        for destination, value in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(value)
            temporary.replace(destination)

    def load_episode(self, policy: PolicyDefinition, seed: int) -> Episode | None:
        if type(seed) is not int:
            raise ValueError("Episode seed must be an integer")
        path = self._output_path(Path("episodes") / policy.id / f"{seed}.json")
        if not path.exists():
            return None
        with path.open("rb") as stream:
            data = stream.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Episode exceeds 64 MiB")
        return Episode.from_data(json.loads(data))
