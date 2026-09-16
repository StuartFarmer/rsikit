"""Abstract proposal contract and its Slick implementation."""

from abc import ABC, abstractmethod

from slick import prompt
from slick.providers import Provider

from .edits import InvalidCandidate, validate_source
from .evaluation import Evaluation
from .repair import Check, Repair, RepairResult, repair_until_valid
from .strategies import Candidate, Propose


class Proposer(ABC):
    """Generate replacement source from a parent and strategy-supplied history."""

    @abstractmethod
    async def __call__(self, parent: Candidate, history: tuple[Candidate, ...]) -> str:
        """Return complete replacement source; the strategy validates it."""
        raise NotImplementedError


class RepairingProposer(Proposer):
    """Compose generation with bounded repair before search sees a candidate.

    Check edit boundaries against the original parent before each task check.
    Completed repair runs (including exhaustion) belong to this object's history.
    Failed/cancelled calls propagate without adding a completed run.
    """

    def __init__(self, propose: Propose, check: Check, repair: Repair, *, max_repairs: int = 3):
        self.propose, self.check, self.repair = propose, check, repair
        self.max_repairs = max_repairs
        self.history: list[RepairResult] = []

    async def __call__(self, parent: Candidate, history: tuple[Candidate, ...]) -> str:
        source = await self.propose(parent, history)

        async def check(source: str) -> Evaluation:
            try:
                validate_source(source, parent.source)
            except InvalidCandidate as exc:
                return Evaluation(valid=False, feedback=str(exc))
            return await self.check(source)

        result = await repair_until_valid(
            source, check=check, repair=self.repair, max_repairs=self.max_repairs
        )
        self.history.append(result)
        return result.require_valid_source()


class SlickProposer(Proposer):
    """Configure Slick's template root to this package's prompts/ before use.

    Task instructions must describe the objective, its direction, and interface.
    Each proposal is an independent model call, with no shared Session or tools.
    """

    def __init__(self, task: str, provider: Provider):
        self.task, self.provider = task, provider

    async def __call__(self, parent: Candidate, history: tuple[Candidate, ...]) -> str:
        # ponytail: eight recent outcomes; add retrieval when longer histories help.
        return await self.revise(parent, history[-8:], provider=self.provider)

    @prompt(template="revise.j2")
    async def revise(
        self, parent: Candidate, history: tuple[Candidate, ...], *, generated: str
    ) -> str:
        """Return raw replacement source; the strategy validates it."""
        return generated
