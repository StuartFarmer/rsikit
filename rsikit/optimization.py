"""The proposal/feedback boundary shared by policy optimizers."""

from collections.abc import Iterable
from typing import Protocol

from .episode import Episode
from .policy import Policy


class Optimizer(Protocol):
    async def propose(self, n: int) -> list[type[Policy]]:
        """Create policy definitions; execution creates fresh instances per episode."""
        ...

    def update(self, results: Iterable[tuple[type[Policy], Episode]]) -> None:
        """Consume paired definitions and episodes, identifying policies by their IDs."""
        ...
