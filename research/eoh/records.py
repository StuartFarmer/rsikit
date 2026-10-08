"""Durable EoH events; source and diagnostics stay available after interruption."""

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class EoHEvent(SQLModel, table=True):
    __tablename__ = "eoh_event"
    id: int | None = Field(default=None, primary_key=True)
    kind: str
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))
