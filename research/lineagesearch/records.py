"""Inspectable research history saved by the caller with Run.save()."""

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Study(SQLModel, table=True):
    __tablename__ = "lineagesearch_study"
    id: int = Field(default=1, primary_key=True)
    task: str
    context: str = ""
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    reason: str = "ready"
    phase: str = "ready"
    attempts: int = 0
    batches: int = 0
    error: str | None = None
    calls: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    decompositions: list[dict] = Field(default_factory=list, sa_column=Column(JSON))


class Family(SQLModel, table=True):
    __tablename__ = "lineagesearch_family"
    id: int = Field(primary_key=True)
    name: str
    mechanism: str
    initial_approaches: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = "active"
    best_id: int | None = None
    checkpoint_id: int | None = None
    frontier: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    batches: int = 0
    stale_batches: int = 0
    failed_batches: int = 0
    pivot_tested: bool = False
    last_gain: float = 0


class Trial(SQLModel, table=True):
    __tablename__ = "lineagesearch_trial"
    id: int = Field(primary_key=True)
    family_id: int = Field(foreign_key="lineagesearch_family.id")
    parent_id: int | None = Field(default=None, foreign_key="lineagesearch_trial.id")
    batch: int
    kind: str
    hypothesis: str = ""
    mechanism: str = ""
    change: str = ""
    test: str = ""
    status: str = "planning"
    policy_id: str | None = None
    name: str = ""
    description: str = ""
    implementation: str = ""
    score: float | None = None
    seed_scores: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    feedback: str = ""
    error: str | None = None
    repairs: int = 0
    revisions: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
