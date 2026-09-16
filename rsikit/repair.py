"""Bounded source repair, independent of proposal generation and search."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .edits import ProposalRejected
from .evaluation import Evaluation

Check = Callable[[str], Awaitable[Evaluation]]
Repair = Callable[[str, Evaluation], Awaitable[str]]


@dataclass(frozen=True)
class RepairAttempt:
    source: str
    evaluation: Evaluation


@dataclass(frozen=True)
class RepairResult:
    """Checked revisions, including the initial source and the final check."""

    attempts: tuple[RepairAttempt, ...]

    @property
    def source(self) -> str:
        return self.attempts[-1].source

    @property
    def valid(self) -> bool:
        return self.attempts[-1].evaluation.valid

    def require_valid_source(self) -> str:
        if not self.valid:
            raise RepairExhausted(self)
        return self.source


class RepairExhausted(ProposalRejected):
    """Repair stopped with invalid source; retain all completed checks."""

    def __init__(self, result: RepairResult):
        self.result = result
        super().__init__(
            result.source,
            f"Repair exhausted after {len(result.attempts) - 1} repairs: "
            f"{result.attempts[-1].evaluation.feedback}",
        )


async def repair_until_valid(
    source: str, *, check: Check, repair: Repair, max_repairs: int = 3
) -> RepairResult:
    """Check once, then repair/recheck at most max_repairs times.

    Only an explicit invalid check triggers repair. Exceptions and cancellation
    propagate. Callers own deadlines, execution isolation, and the checks needed
    for their task; validity here makes no claim beyond those checks.
    """
    attempts = []
    while True:
        evaluation = await check(source)
        attempts.append(RepairAttempt(source, evaluation))
        if evaluation.valid or len(attempts) > max_repairs:
            return RepairResult(tuple(attempts))
        source = await repair(source, evaluation)
