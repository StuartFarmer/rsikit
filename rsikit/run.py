"""A local run: generated policies, durable episode jobs, and optional artifacts."""

import asyncio
import fcntl
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .episode import PolicyError
from .policy import Policy
from .sandbox import run_policy


def _now():
    return datetime.now(timezone.utc)


def _slug(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64] or "policy"


class _RunRecord(SQLModel, table=True):
    __tablename__ = "run"
    id: str = Field(primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=_now)
    environment: str
    environment_kwargs: dict = Field(default_factory=dict, sa_column=Column(JSON))
    max_steps: int | None = None
    instructions: str | None = None
    call_timeout: float = 10.0
    export: bool = True
    record_video: bool = False
    schema_version: int = 1


class _PolicyRecord(SQLModel, table=True):
    __tablename__ = "policy"
    id: str = Field(primary_key=True)
    name: str
    summary: str = ""
    implementation: str
    created_at: datetime = Field(default_factory=_now)

    def policy(self):
        value = Policy(id=self.id, name=self.name, summary=self.summary)
        value._implementation = self.implementation
        return value


class Execution(SQLModel, table=True):
    """One persisted episode; terminal results are reused when a run resumes."""

    __tablename__ = "execution"
    __table_args__ = (UniqueConstraint("policy_id", "env_seed", "policy_seed"),)
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    policy_id: str = Field(foreign_key="policy.id", index=True)
    env_seed: int
    policy_seed: int
    status: str = "pending"
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = 0
    reward: float | None = None
    length: int | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Run:
    """One owner per run directory. Resume episodes, never optimizer state."""

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
    ):
        record = _RunRecord(
            id=uuid4().hex,
            name=name,
            environment=environment,
            environment_kwargs=environment_kwargs or {},
            max_steps=max_steps,
            instructions=instructions,
            call_timeout=call_timeout,
            export=export,
            record_video=record_video,
        )
        directory = (
            Path(path)
            if path is not None
            else Path("runs")
            / (f"{_slug(name)}-{record.created_at:%Y%m%dT%H%M%SZ}-{record.id[:8]}")
        )
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory, record)

    @classmethod
    def open(cls, path: str | Path):
        directory = Path(path)
        if not (directory / "run.sqlite").is_file():
            raise FileNotFoundError(directory / "run.sqlite")
        return cls(directory)

    def __init__(self, path: Path, record: _RunRecord | None = None):
        self.path = path.resolve()
        self._lock = (self.path / ".lock").open("a")
        self._engine = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._engine = create_engine(f"sqlite:///{self.path / 'run.sqlite'}")
            if record is not None:
                SQLModel.metadata.create_all(
                    self._engine,
                    tables=[_RunRecord.__table__, _PolicyRecord.__table__, Execution.__table__],
                )
            with Session(self._engine) as session:
                if record is not None:
                    session.add(record)
                    session.commit()
                self._record = session.exec(select(_RunRecord)).one()
                if self._record.schema_version != 1:
                    raise ValueError("Unsupported run database version")
                for execution in session.exec(
                    select(Execution).where(Execution.status == "running")
                ):
                    execution.status = "pending"
                    execution.error = "Interrupted before completion was committed"
                    session.add(execution)
                session.commit()
                session.refresh(self._record)
                for stored in session.exec(select(_PolicyRecord)):
                    self._export(stored)
            self._busy = asyncio.Lock()
        except BaseException:
            self.close()
            raise

    @property
    def name(self):
        return self._record.name

    @property
    def id(self):
        return self._record.id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
        self._lock.close()

    def _export(self, policy):
        if self._record.export:
            destination = self.path / "exports" / f"{policy.id}_{_slug(policy.name)}.py"
            if not destination.exists():
                destination.parent.mkdir(exist_ok=True)
                temporary = destination.with_suffix(".py.tmp")
                temporary.write_text(policy.implementation, encoding="utf-8")
                temporary.replace(destination)

    def policies(self) -> list[Policy]:
        with Session(self._engine) as session:
            return [
                row.policy()
                for row in session.exec(
                    select(_PolicyRecord).order_by(_PolicyRecord.created_at, _PolicyRecord.id)
                )
            ]

    def executions(self, policy: Policy | None = None) -> list[Execution]:
        query = select(Execution).order_by(Execution.created_at, Execution.id)
        if policy is not None:
            query = query.where(Execution.policy_id == policy.id)
        with Session(self._engine) as session:
            return list(session.exec(query))

    async def evaluate(self, policy: Policy, *, seeds=(0,)) -> list[Execution]:
        seeds = tuple(dict.fromkeys(seeds))
        async with self._busy:
            if self._lock.closed:
                raise RuntimeError("Run is closed")
            if not policy._implementation:
                raise ValueError("Pass a policy returned by generation or loaded from a run")
            with Session(self._engine) as session:
                stored = session.get(_PolicyRecord, policy.id)
                if stored is None:
                    stored = _PolicyRecord(
                        id=policy.id,
                        name=policy.name,
                        summary=policy.summary,
                        implementation=policy._implementation,
                    )
                    session.add(stored)
                elif (stored.name, stored.summary, stored.implementation) != (
                    policy.name,
                    policy.summary,
                    policy._implementation,
                ):
                    raise ValueError("A stored policy cannot change under the same ID")
                for seed in seeds:
                    existing = session.exec(
                        select(Execution).where(
                            Execution.policy_id == policy.id,
                            Execution.env_seed == seed,
                            Execution.policy_seed == seed,
                        )
                    ).first()
                    if existing is None:
                        session.add(Execution(policy_id=policy.id, env_seed=seed, policy_seed=seed))
                # Persist the complete requested work before starting any episode.
                session.commit()
                session.refresh(stored)
                self._export(stored)
            jobs = [job for job in self.executions(policy) if job.env_seed in seeds]
            await self._execute(jobs)
            return [job for job in self.executions(policy) if job.env_seed in seeds]

    async def resume(self) -> list[Execution]:
        async with self._busy:
            if self._lock.closed:
                raise RuntimeError("Run is closed")
            await self._execute(self.executions())
            return self.executions()

    async def _execute(self, jobs):
        for job in jobs:
            if job.status != "pending":
                continue
            job.status, job.error = "running", None
            job.started_at = _now()
            job.attempts += 1
            with Session(self._engine) as session:
                policy = session.get(_PolicyRecord, job.policy_id).policy()
                session.add(job)
                session.commit()
                session.refresh(job)
            artifact_dir = self.path / "artifacts" / job.id / str(job.attempts)

            def make_env():
                kwargs = dict(self._record.environment_kwargs)
                if self._record.record_video:
                    kwargs["render_mode"] = "rgb_array"
                env = gym.make(self._record.environment, **kwargs)
                if self._record.record_video:
                    try:
                        env = gym.wrappers.RecordVideo(
                            env,
                            str(artifact_dir),
                            episode_trigger=lambda _: True,
                        )
                    except BaseException:
                        env.close()
                        raise
                return env

            try:
                _, _, job.terminated, job.truncated, info = await run_policy(
                    policy,
                    make_env,
                    env_seed=job.env_seed,
                    policy_seed=job.policy_seed,
                    max_steps=self._record.max_steps,
                    instructions=self._record.instructions,
                    call_timeout=self._record.call_timeout,
                )
                job.reward, job.length = info["episode"]["r"], info["episode"]["l"]
                job.status = "completed"
            except PolicyError as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            except BaseException as exc:
                job.status, job.error = "pending", f"{type(exc).__name__}: {exc}"
                with Session(self._engine) as session:
                    session.add(job)
                    session.commit()
                raise
            job.finished_at = _now()
            job.artifacts = [
                str(p.relative_to(self.path)) for p in sorted(artifact_dir.glob("*.mp4"))
            ]
            with Session(self._engine) as session:
                session.add(job)
                session.commit()
