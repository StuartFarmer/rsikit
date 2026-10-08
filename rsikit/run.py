"""Persist the configuration, checkpoints and outputs of one optimizer run."""

import asyncio
import fcntl
import json
import logging
import math
import pickle
import re
from contextlib import AsyncExitStack, aclosing
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from sqlalchemy import JSON, Column, inspect
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .episode import Episode
from .evaluation import PolicyError, episode_error, episode_scores
from .execution import Executor, Job
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
    """Persist policy source, scores, episodes, and artifacts in a local directory.

    Use ``create`` for a new directory or ``open`` for an existing trusted run.
    Both acquire an exclusive directory lock immediately. Synchronous and async
    context managers bind run logging and release the lock on exit; otherwise
    call ``close`` explicitly.

    A run must use one fixed environment and evaluation configuration. Cached
    successful episodes are identified only by policy ID and seed, so changing
    environment settings requires a new run. Failed episodes are retried.

    Attributes:
        path: Absolute run directory containing ``run.sqlite``, episodes, artifacts,
            and optional exported policy source.
    """

    @classmethod
    def create(
        cls,
        *,
        name: str,
        path: str | Path | None = None,
        export: bool = True,
        console=None,
    ) -> "Run":
        """Create and lock a new run directory.

        Args:
            name: Human-readable run name.
            path: New directory to create. Defaults to a unique directory beneath
                ``runs/`` using the name and UTC timestamp.
            export: Whether to export stored policies as Python files in ``exports/``.
            console (rich.console.Console | None): Optional Rich console used when entering the run's context.

        Returns:
            An open run. Use a context manager or explicitly close it.

        Raises:
            FileExistsError: The requested directory already exists.
            OSError: Directory creation or lock acquisition fails.
        """
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
        """Open and exclusively lock an existing trusted run.

        Missing enabled policy exports are regenerated on opening. Legacy policy
        tables are migrated to add the description field when needed.

        Args:
            path: Directory containing ``run.sqlite``.
            console (rich.console.Console | None): Optional Rich console used when entering the run's context.

        Returns:
            An open run with its saved settings.

        Raises:
            FileNotFoundError: The run database does not exist.
            BlockingIOError: Another run instance holds the directory lock.
            OSError: Files or the lock cannot be accessed.
        """
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
        self._evaluation_locks = {}
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
        """Human-readable name saved when the run was created."""
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
        """Close this run; asynchronous convenience wrapper around ``close``."""
        self.close()

    def close(self) -> None:
        """Release logging scope, database connections, and the directory lock.

        Safe to call more than once. A closed run cannot be entered again or used
        for database and episode operations.
        """
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

        Args:
            *records: SQLModel table instances to merge in one transaction.

        Raises:
            RuntimeError: The run is closed.

        Note:
            SQLAlchemy schema and transaction errors propagate to the caller.
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
        """Open a native SQLModel session against the run database.

        Returns:
            A new session owned by the caller; use ``with run.database() as db``
                to close it after queries or transactions.

        Raises:
            RuntimeError: The run is closed.
        """
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
        """Return stored policy definitions without executing or validating their source.

        Raises:
            RuntimeError: The run is closed.
        """
        with self.database() as session:
            return [
                PolicyDefinition.from_text(
                    row.implementation, name=row.name, description=row.description
                )
                for row in session.exec(select(_StoredPolicy))
            ]

    def scores(self, policy: PolicyDefinition) -> dict[int, float | None]:
        """Read recorded fitness values for a policy.

        Args:
            policy: Definition whose ID selects the stored record.

        Returns:
            Seed IDs mapped to scores or ``None`` for unscored entries. Returns
                an empty mapping when the policy has not been stored.

        Raises:
            RuntimeError: The run is closed.
        """
        with self.database() as session:
            row = session.get(_StoredPolicy, policy.id)
            return {} if row is None else {int(seed): score for seed, score in row.scores.items()}

    def save_policy(
        self, policy: PolicyDefinition, *, scores: dict[int, float | None] | None = None
    ) -> None:
        """Persist source and merge supplied scores without evaluating the policy.

        Args:
            policy: Definition to store by its content-derived ID.
            scores: Optional mapping of integer seeds to finite scores or ``None``.
                Supplied seeds replace existing scores; omitted seeds are retained.

        Raises:
            ValueError: Scores are invalid or a stored ID conflicts with the source.
            RuntimeError: The run is closed.
            OSError: An enabled source export cannot be written.
        """
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
        """Save a validated episode and its artifact files for one policy and seed.

        Source is saved too. Episode and artifact files are individually replaced
        atomically; the collection is not a single transaction. This does not
        calculate or update fitness scores.

        Args:
            policy: Policy definition identifying the evidence.
            seed: Integer seed ID used in the episode filename.
            episode: Complete successful or failed trajectory to serialize as pickle.

        Raises:
            ValueError: Inputs or trajectory are invalid, the encoded episode
                exceeds 64 MiB, or an artifact path escapes its evaluation directory.
            RuntimeError: The run is closed.
            OSError: A policy export, episode, or artifact cannot be written.
        """
        if type(seed) is not int:
            raise ValueError("Episode seed must be an integer")
        if not isinstance(episode, Episode):
            raise ValueError("Expected an Episode")
        data = pickle.dumps(episode.encode(), protocol=pickle.HIGHEST_PROTOCOL)
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
        outputs.append((self._output_path(Path("episodes") / policy.id / f"{seed}.pkl"), data))
        for destination, value in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(value)
            temporary.replace(destination)

    def load_episode(self, policy: PolicyDefinition, seed: int) -> Episode | None:
        """Load saved evidence from a trusted run directory.

        Pickle loading may execute Python; never use an untrusted run directory.
        Legacy JSON episodes are accepted if the pickle file is absent.

        Args:
            policy: Definition whose ID locates the saved episode.
            seed: Integer seed ID identifying the episode.

        Returns:
            A validated episode, or ``None`` if no saved episode exists.

        Raises:
            ValueError: Seed, stored size, or decoded episode is invalid.
            RuntimeError: The run is closed.
            OSError: The saved file cannot be read.

        Note:
            Pickle and JSON decoding errors propagate to the caller.
        """
        if type(seed) is not int:
            raise ValueError("Episode seed must be an integer")
        path = self._output_path(Path("episodes") / policy.id / f"{seed}.pkl")
        if not path.exists():
            path = self._output_path(Path("episodes") / policy.id / f"{seed}.json")
        if not path.exists():
            return None
        with path.open("rb") as stream:
            data = stream.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Episode exceeds 64 MiB")
        return Episode.from_data(pickle.loads(data) if path.suffix == ".pkl" else json.loads(data))

    async def collect(self, policies, *, environment, executor: Executor, seeds=(0,)):
        """Yield saved successes and execute missing or previously failed episodes.

        Args:
            policies (Iterable[PolicyDefinition]): Definitions, deduplicated by policy ID.
            environment (gymnasium.Env): Caller-owned environment copied by the executor.
                Use the same configuration throughout this run.
            executor: Executor inside its active async context.
            seeds (Iterable[int]): Nonempty seed IDs, deduplicated in order.
                New jobs require nonnegative seeds.

        Yields:
            result (tuple[str, int, Episode]): ``(policy_id, seed, episode)`` tuples. New evidence is saved
                before yielding. No score calculation is performed.

        Raises:
            ValueError: Seeds or executor results are invalid.
            RuntimeError: The run or executor is closed, or expected results are missing.
            InfrastructureError: Execution or result transport fails.

        Note:
            Cached successes use only policy ID and seed as the key. Concurrent
            calls on this run serialize evaluation of overlapping policy IDs.
            Use ``contextlib.aclosing`` if stopping iteration early, so locks and
            pending executor work are released before either context exits.
        """
        seeds = tuple(seeds)
        if not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("Evaluation requires at least one seed; seed IDs must be integers")
        seeds = tuple(dict.fromkeys(seeds))
        policies = {policy.id: policy for policy in policies}
        name = getattr(getattr(environment, "spec", None), "id", None) or type(environment).__name__
        logging.getLogger(__name__).info(
            "Environment: %s", name, extra={"progress": dict(kind="environment", name=name)}
        )
        async with AsyncExitStack() as stack:
            for policy_id in sorted(policies):
                await stack.enter_async_context(
                    self._evaluation_locks.setdefault(policy_id, asyncio.Lock())
                )
            jobs = []
            for policy in policies.values():
                self.save_policy(policy)
                for seed in seeds:
                    episode = self.load_episode(policy, seed)
                    if episode is None or episode.error is not None:
                        jobs.append(Job(policy, environment, seed=seed))
                    else:
                        yield policy.id, seed, episode
            requested = {(job.policy.id, job.seed) for job in jobs}
            async with aclosing(executor.iterate(jobs)) as results:
                async for job in results:
                    policy_id, seed, episode = job.policy.id, job.seed, job.result
                    if type(seed) is not int or (policy_id, seed) not in requested:
                        raise ValueError("Executor returned an unexpected or duplicate result")
                    self.save_episode(policies[policy_id], seed, episode)
                    requested.remove((policy_id, seed))
                    yield policy_id, seed, episode
            if requested:
                raise RuntimeError("Executor finished without returning all requested results")

    async def evaluate(
        self, policies, *, environment, executor: Executor, seeds=(0,)
    ) -> dict[str, dict[int, Episode]]:
        """Collect an evaluation panel and persist successful fitness values.

        Args:
            policies (Iterable[PolicyDefinition]): Definitions, deduplicated by ID.
            environment (gymnasium.Env): Fixed caller-owned environment for this run.
            executor: Executor inside its active async context.
            seeds (Iterable[int]): Nonempty seeds, deduplicated in order.
                New jobs require nonnegative seeds.

        Returns:
            Policy IDs mapped to seed-keyed episodes, including failures. Successful
                scores use final ``info["fitness"]`` when present, otherwise total reward.
                Newly requested scores start as ``None`` until successfully evaluated.

        Raises:
            ValueError: Seeds, episodes, or successful fitness values are invalid.
            RuntimeError: The run or executor is closed, or expected results are missing.
            InfrastructureError: Execution or result transport fails.

        Note:
            Uses ``collect`` caching rules; policy failures remain in the returned
            episodes rather than raising ``PolicyError``.
        """
        seeds = tuple(seeds)
        if not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("Evaluation requires at least one seed; seed IDs must be integers")
        seeds = tuple(dict.fromkeys(seeds))
        policies = {policy.id: policy for policy in policies}
        results = {policy_id: {} for policy_id in policies}
        for policy in policies.values():
            stored = self.scores(policy)
            self.save_policy(policy, scores={seed: None for seed in seeds if seed not in stored})
        logging.getLogger(__name__).info(
            "Evaluating %s episodes (%s workers)",
            len(policies) * len(seeds),
            executor.concurrency,
            extra={"event": "evaluation_started", "total": len(policies) * len(seeds)},
        )
        async with aclosing(
            self.collect(policies.values(), environment=environment, executor=executor, seeds=seeds)
        ) as episodes:
            async for policy_id, seed, episode in episodes:
                results[policy_id][seed] = episode
                if episode.error is None:
                    score = episode_scores({seed: episode})[seed]
                    self.save_policy(policies[policy_id], scores={seed: score})
                    logging.getLogger(__name__).info(
                        "%s: score=%g (seed=%s)",
                        policies[policy_id].name,
                        score,
                        seed,
                        extra={"event": "policy_evaluated", "policy_id": policy_id, "seed": seed},
                    )
        return results

    async def mean_scores(
        self, policies, *, environment, executor: Executor, seeds=(0,)
    ) -> dict[str, float]:
        """Evaluate policies and average fitness across the requested seed panel.

        Args:
            policies (Iterable[PolicyDefinition]): Policy definitions.
            environment (gymnasium.Env): Fixed caller-owned environment for this run.
            executor: Executor inside its active async context.
            seeds (Iterable[int]): Nonempty seeds; new jobs require nonnegative seeds.

        Returns:
            Policy IDs mapped to arithmetic mean fitness across their episodes.

        Raises:
            PolicyError: Any episode failed. The exception's ``failures`` mapping
                contains diagnostics keyed by failed policy ID.
            ValueError: Seeds, episodes, or fitness values are invalid.
            RuntimeError: The run or executor is closed, or expected results are missing.
            InfrastructureError: Execution or result transport fails.
        """
        results = await self.evaluate(
            policies, environment=environment, executor=executor, seeds=seeds
        )
        errors = {
            id: error for id, episodes in results.items() if (error := episode_error(episodes))
        }
        if errors:
            error = PolicyError("\n".join(errors.values()))
            error.failures = errors
            raise error
        return {id: fmean(episode_scores(episodes).values()) for id, episodes in results.items()}
