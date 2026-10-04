"""The proposal/feedback boundary shared by policy optimizers."""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from .evaluation import Measurement
from .policy import Policy


class Optimizer(Protocol):
    @property
    def done(self) -> bool: ...

    @property
    def best(self) -> type[Policy] | None: ...

    async def propose(self) -> list[type[Policy]]:
        """Create policy definitions; execution creates fresh instances per episode."""
        ...

    def update(self, results: Mapping[str, Measurement]) -> None:
        """Consume exactly one complete round, including rejected and failed candidates."""
        ...


def validate_results(results, pending, *, seed_panel=None):
    """Validate a complete round before any optimizer state changes."""
    if not pending or not isinstance(results, Mapping) or set(results) != set(pending):
        raise ValueError("Feedback must contain exactly the outstanding policy IDs")
    panel = seed_panel
    for result in results.values():
        if not isinstance(result, Measurement):
            raise ValueError("Evaluator must return Measurement values")
        if result.accepted and result.scores:
            if panel is not None and set(result.scores) != panel:
                raise ValueError("All candidates must use the same seed panel")
            panel = set(result.scores)
    return panel


async def search(
    optimizer: Optimizer,
    evaluate: Callable[[Sequence[type[Policy]]], Awaitable[Mapping[str, Measurement]]],
    *,
    on_checkpoint: Callable[[Optimizer], None] | None = None,
) -> type[Policy] | None:
    """Drive complete proposal rounds; the caller owns evaluation resources."""

    def checkpoint():
        if on_checkpoint is not None:
            on_checkpoint(optimizer)

    try:
        while not optimizer.done:
            policies = await optimizer.propose()
            checkpoint()
            if not policies:
                if not optimizer.done:
                    raise RuntimeError("Empty proposal round without optimizer completion")
                break
            ids = [p.id for p in policies]
            if len(set(ids)) != len(ids):
                raise ValueError("Proposal round contains duplicate policy IDs")
            results = await evaluate(policies)
            validate_results(results, ids)
            optimizer.update(results)
            checkpoint()
        return optimizer.best
    except BaseException:
        try:
            checkpoint()
        except BaseException:
            logging.getLogger(__name__).exception("Checkpoint failed while handling search error")
        raise
