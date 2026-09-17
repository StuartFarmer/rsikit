"""Snapshot migration between independently selected populations and archives."""

from collections.abc import Callable, Iterable, Sequence

from .archives import EliteArchive, QDArchive, SteppingStoneArchive
from .strategies import Candidate

Archive = EliteArchive | SteppingStoneArchive | QDArchive


def migrate(
    islands: Sequence[Archive],
    routes: Iterable[tuple[int, int]],
    *,
    select: Callable[[tuple[Candidate, ...]], Sequence[Candidate]],
) -> list[dict]:
    """Copy selected candidates across directed routes using one pre-migration snapshot.

    select receives each route's source snapshot; destinations decide admission.
    Migrants preserve run-global IDs and lineage. No selection reward is assigned.
    Selection failures precede any writes; admission errors propagate and may leave
    earlier routes committed. Callers own topology, timing and retry policy.
    """
    snapshots = [island.candidates for island in islands]
    outgoing = [(source, target, tuple(select(snapshots[source]))) for source, target in routes]
    events = []
    for source, target, candidates in outgoing:
        for candidate in candidates:
            admitted = islands[target].add(candidate)
            events.append(
                dict(source=source, target=target, candidate_id=candidate.id, admitted=admitted)
            )
    return events
