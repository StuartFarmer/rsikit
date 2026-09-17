"""Store generated policies and their scores in one local SQLite database."""

import asyncio
import fcntl
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .episode import run_episode
from .policy import Policy
from .sandbox import _policy_class


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64] or "policy"


class _Settings(SQLModel, table=True):
    __tablename__ = "settings"
    name: str = Field(primary_key=True)
    environment: str
    environment_kwargs: dict = Field(default_factory=dict, sa_column=Column(JSON))
    max_steps: int | None = None
    instructions: str | None = None
    call_timeout: float = 10.0
    export: bool = True
    record_video: bool = False


class _StoredPolicy(SQLModel, table=True):
    __tablename__ = "policy"
    id: str = Field(primary_key=True)
    name: str
    implementation: str
    scores: dict[str, float | None] = Field(default_factory=dict, sa_column=Column(JSON))


class Run:
    """A directory containing policies and scores. Unfinished scores are None."""

    @classmethod
    def create(
        cls,
        *,
        name: str,
        environment: str,
        path: str | Path | None = None,
        environment_kwargs: dict | None = None,
        max_steps: int | None = None,
        instructions: str | None = None,
        call_timeout: float = 10.0,
        export: bool = True,
        record_video: bool = False,
    ) -> "Run":
        settings = _Settings(
            name=name,
            environment=environment,
            environment_kwargs=environment_kwargs or {},
            max_steps=max_steps,
            instructions=instructions,
            call_timeout=call_timeout,
            export=export,
            record_video=record_video,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            Path(path)
            if path is not None
            else Path("runs") / f"{_slug(name)}-{stamp}-{uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory, settings)

    @classmethod
    def open(cls, path: str | Path) -> "Run":
        directory = Path(path)
        if not (directory / "run.sqlite").is_file():
            raise FileNotFoundError(directory / "run.sqlite")
        return cls(directory)

    def __init__(self, path: Path, settings: _Settings | None = None):
        self.path = path.resolve()
        self._lock = (self.path / ".lock").open("a")
        self._engine = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._engine = create_engine(f"sqlite:///{self.path / 'run.sqlite'}")
            if settings is not None:
                SQLModel.metadata.create_all(
                    self._engine, tables=[_Settings.__table__, _StoredPolicy.__table__]
                )
            with Session(self._engine) as session:
                if settings is not None:
                    session.add(settings)
                    session.commit()
                self._settings = session.exec(select(_Settings)).one()
                for stored in session.exec(select(_StoredPolicy)):
                    self._export(stored)
            self._busy = asyncio.Lock()
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return self._settings.name

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
        self._lock.close()

    def _export(self, policy: _StoredPolicy) -> None:
        if self._settings.export:
            destination = self.path / "exports" / f"{policy.id}_{_slug(policy.name)}.py"
            if not destination.exists():
                destination.parent.mkdir(exist_ok=True)
                temporary = destination.with_suffix(".py.tmp")
                temporary.write_text(policy.implementation, encoding="utf-8")
                temporary.replace(destination)

    def policies(self) -> list[type[Policy]]:
        with Session(self._engine) as session:
            return [
                _policy_class(row.name, row.implementation)
                for row in session.exec(select(_StoredPolicy))
            ]

    def scores(self, policy: type[Policy]) -> dict[int, float | None]:
        with Session(self._engine) as session:
            row = session.get(_StoredPolicy, policy.id)
            return {} if row is None else {int(seed): score for seed, score in row.scores.items()}

    async def evaluate(self, policy: type[Policy], *, seeds=(0,)) -> dict[int, float]:
        seeds = tuple(dict.fromkeys(seeds))
        async with self._busy:
            if self._lock.closed:
                raise RuntimeError("Run is closed")
            with Session(self._engine) as session:
                stored = session.get(_StoredPolicy, policy.id)
                if stored is None:
                    stored = _StoredPolicy(
                        id=policy.id, name=policy.name, implementation=policy._implementation
                    )
                elif (stored.name, stored.implementation) != (policy.name, policy._implementation):
                    raise ValueError("A stored policy cannot change under the same ID")
                stored.scores = {**dict.fromkeys(map(str, seeds)), **stored.scores}
                session.add(stored)
                # Save requested work before executing it; exceptions leave None scores.
                session.commit()
                session.refresh(stored)
                self._export(stored)
                for seed in seeds:
                    if stored.scores[str(seed)] is not None:
                        continue

                    def make_env():
                        kwargs = dict(self._settings.environment_kwargs)
                        if self._settings.record_video:
                            kwargs["render_mode"] = "rgb_array"
                        env = gym.make(self._settings.environment, **kwargs)
                        if self._settings.record_video:
                            try:
                                env = gym.wrappers.RecordVideo(
                                    env,
                                    str(self.path / "videos" / policy.id / str(seed)),
                                    episode_trigger=lambda _: True,
                                )
                            except BaseException:
                                env.close()
                                raise
                        return env

                    *_, info = await run_episode(
                        make_env,
                        partial(policy, call_timeout=self._settings.call_timeout),
                        env_seed=seed,
                        policy_seed=seed,
                        max_steps=self._settings.max_steps,
                        instructions=self._settings.instructions,
                    )
                    stored.scores = {**stored.scores, str(seed): float(info["episode"]["r"])}
                    session.add(stored)
                    session.commit()
                return {seed: stored.scores[str(seed)] for seed in seeds}

    async def resume(self) -> None:
        """Retry unfinished evaluations. Errors propagate; completed scores are reused."""
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        for policy in self.policies():
            await self.evaluate(policy, seeds=self.scores(policy))
