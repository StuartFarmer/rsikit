"""Stateful search strategies; callers own evaluation and the experiment loop."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from .edits import InvalidCandidate, ProposalRejected, validate_source
from .evaluation import Evaluation, EvaluationError


class Candidate(BaseModel, frozen=True):
    """Source and lineage; completed candidates also carry measured feedback."""

    id: int
    source: str
    parent_id: int | None = None
    parent_ids: tuple[int, ...] = ()
    evaluation: Evaluation | None = None


Propose = Callable[[Candidate, tuple[Candidate, ...]], Awaitable[str]]


class SequentialStrategy(ABC):
    """Share single-candidate bookkeeping; subclasses own selection and admission.

    Initialize with source and its already measured evaluation. Use one operation
    at a time per instance and evaluate/update each generated batch before the next
    generation. This strategy generates one candidate per batch. It performs no
    file I/O, evaluation, retries, logging, or timeout management.
    """

    def __init__(
        self,
        initial: str,
        evaluation: Evaluation,
        propose: Propose,
        *,
        objective: str = "score",
        maximize: bool = True,
    ):
        self.propose, self.objective, self.maximize = propose, objective, maximize
        validate_source(initial)
        if not evaluation.valid:
            raise InvalidCandidate("Initial program failed evaluation")
        self._score(evaluation)
        self.best = Candidate(id=0, source=initial, evaluation=evaluation)
        self.history = [self.best]
        self.pending: Candidate | None = None
        self.context: dict = {}
        self.selections: list[dict] = []

    @abstractmethod
    def select_parent(self) -> Candidate:
        """Select a parent and prepare context for the proposal callable."""
        raise NotImplementedError

    def observe(self, candidate: Candidate) -> None:
        """Let a concrete strategy incorporate a completed or rejected attempt."""

    def _record(self, candidate: Candidate) -> None:
        self.observe(candidate)
        self.history.append(candidate)
        self.selections.append(
            {
                "id": candidate.id,
                "operation": self.context.get("operation", "mutate"),
                "parents": [c.id for c in self.context.get("parents", ())],
                "inspirations": [c.id for c in self.context.get("inspirations", ())],
                "island": self.context.get("island"),
            }
        )

    async def generate(self) -> list[Candidate]:
        """Propose from the selected parent; remember invalid edits and return no work.

        Explicit proposal rejections are recorded as invalid attempts. Other
        failed/cancelled calls propagate without admitting a candidate or changing
        completed history. Parent sampling may advance RNG and proposal context.
        Every returned candidate must receive an evaluation through update().
        """
        if self.pending is not None:
            raise RuntimeError("Update the pending candidate before generating again")
        parent = self.select_parent()
        parent_ids = tuple(c.id for c in self.context.get("parents", (parent,)))
        try:
            source = await self.propose(parent, tuple(self.history))
        except ProposalRejected as exc:
            self._record(
                Candidate(
                    id=len(self.history),
                    source=exc.source,
                    parent_id=parent.id,
                    parent_ids=parent_ids,
                    evaluation=Evaluation(valid=False, feedback=str(exc)),
                )
            )
            return []
        candidate = Candidate(
            id=len(self.history), source=source, parent_id=parent.id, parent_ids=parent_ids
        )
        try:
            validate_source(source, parent.source)
        except InvalidCandidate as exc:
            evaluation = Evaluation(valid=False, feedback=str(exc))
            self._record(candidate.model_copy(update={"evaluation": evaluation}))
            return []
        self.pending = candidate
        return [candidate]

    async def update(
        self, candidates: Sequence[Candidate], evaluations: Sequence[Evaluation]
    ) -> None:
        """Incorporate a complete batch; ties and invalid results keep the incumbent.

        Mismatched candidates/results and missing objectives leave pending state
        intact, allowing the caller to correct or retry evaluation. An empty batch
        is a no-op only when no candidate is pending.
        """
        expected = [] if self.pending is None else [self.pending]
        if list(candidates) != expected or len(evaluations) != len(expected):
            raise ValueError("Update must match the pending candidates and evaluation count")
        if not expected:
            return
        evaluation = evaluations[0]
        improved = False
        if evaluation.valid:
            score, incumbent = self._score(evaluation), self._score(self.best.evaluation)
            improved = score > incumbent if self.maximize else score < incumbent
        completed = self.pending.model_copy(update={"evaluation": evaluation})
        self._record(completed)
        if improved:
            self.best = completed
        self.pending = None

    def _score(self, evaluation: Evaluation) -> float:
        if self.objective not in evaluation.metrics:
            raise EvaluationError(f"Valid evaluation is missing objective {self.objective!r}")
        return evaluation.metrics[self.objective]


class HillClimb(SequentialStrategy):
    """Always propose from the best candidate and retain strict improvements."""

    def select_parent(self) -> Candidate:
        self.context = {"operation": "mutate", "parents": (self.best,)}
        return self.best
