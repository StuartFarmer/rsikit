"""Optimizer-owned evaluation records and generation snapshots for plotting."""

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Evaluation(SQLModel, table=True):
    """One proposed policy version, including failed and unevaluated proposals."""

    __tablename__ = "alphaevolve_evaluation"
    generation: int = Field(primary_key=True)
    attempt: int = Field(primary_key=True)
    revision: int = Field(default=0, primary_key=True)
    island: int | None = None
    policy_id: str | None = None
    parent_id: str | None = None
    status: str
    score: float | None = None
    repairs: int = 0
    error: str | None = None


class Generation(SQLModel, table=True):
    """Island champions and resets at a generation boundary; no recovery state."""

    __tablename__ = "alphaevolve_generation"
    number: int = Field(primary_key=True)
    optimizer: str
    complete: bool = False
    seeds: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    islands: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    resets: list[dict] = Field(default_factory=list, sa_column=Column(JSON))


def history_records(
    generator,
    *,
    generation: int,
    attempt_start: int,
    event_start: int,
    seeds,
    complete: bool = False,
    failures: dict[str, str] | None = None,
) -> list[SQLModel]:
    """Build typed records for the current batch and archive without writing storage."""
    failures = failures or {}
    records = []
    for record in generator.attempts[attempt_start:]:
        policy, parent = record.get("policy"), record.get("parent")
        policy_id = None if policy is None else policy.id
        diagnostic = failures.get(policy_id)
        status = record["status"]
        if diagnostic and status != "discarded":
            status = "failed"
        records.append(
            Evaluation(
                generation=generation,
                attempt=record["id"],
                revision=record.get("revision", 0),
                island=record.get("island"),
                policy_id=policy_id,
                parent_id=None if parent is None else parent.policy.id,
                status=status,
                score=record.get("score"),
                repairs=len(record.get("repairs", [])),
                error=record.get("error") or diagnostic,
            )
        )
    records.append(
        Generation(
            number=generation,
            optimizer=type(generator).__module__,
            complete=complete,
            seeds=list(seeds),
            islands=[
                {
                    "island": i,
                    "policy_id": None if champion is None else champion.policy.id,
                    "score": None if champion is None else champion.score,
                }
                for i, champion in enumerate(generator.islands)
            ],
            resets=generator.events[event_start:],
        )
    )
    return records
