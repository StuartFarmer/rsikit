"""Legacy score records retained for unintegrated search research modules."""

from typing import Annotated

from pydantic import BaseModel, Field, StrictBool

Metric = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Evaluation(BaseModel, extra="forbid", frozen=True):
    """Correctness is a gate; named finite measurements determine selection."""

    valid: StrictBool
    metrics: dict[str, Metric] = Field(default_factory=dict)
    feedback: str = ""


class EvaluationError(RuntimeError):
    """The evaluator failed to supply its promised measurement contract."""
