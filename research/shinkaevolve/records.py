"""ShinkaEvolve's experiment records; Run infers their SQLite tables."""

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Evaluation(SQLModel, table=True):
    __tablename__ = "shinkaevolve_evaluation"
    generation: int = Field(primary_key=True)
    attempt: int = Field(primary_key=True)
    revision: int = Field(default=0, primary_key=True)
    island: int
    model: int
    patch: str
    parents: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    policy_id: str | None = None
    status: str = "generating"
    score: float | None = None
    proposals: int = 0
    repairs: int = 0
    error: str | None = None
    similarity: float | None = None
    nearest_id: str | None = None
    novelty_reason: str | None = None
    model_gain: str | None = None


class Generation(SQLModel, table=True):
    __tablename__ = "shinkaevolve_generation"
    number: int = Field(primary_key=True)
    complete: bool = False
    seeds: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    islands: list[list[dict]] = Field(default_factory=list, sa_column=Column(JSON))
    events: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    recommendations: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    model_weights: list[float] = Field(default_factory=list, sa_column=Column(JSON))
    model_counts: list[int] = Field(default_factory=list, sa_column=Column(JSON))
