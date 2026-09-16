"""Bounded run-local guidance derived from measured candidate outcomes."""

from slick import prompt
from slick.providers import Provider

from .evaluation import EvaluationError
from .proposer import _RecordedProvider
from .strategies import Candidate


class ReflectionMemory:
    """Keep completed reflections separately from measurements and failed calls.

    Configure the checkout template root once. Observe only completed attempts;
    the caller commits search state first and owns deadlines and call allowances.
    """

    def __init__(self, task: str, provider: Provider, *, max_items: int = 8):
        self.task, self.provider, self.max_items = task, provider, max_items
        self.records: list[dict] = []
        self.attempts: list[dict] = []

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(record["text"] for record in self.records)

    async def observe(
        self,
        parent: Candidate,
        candidate: Candidate,
        *,
        objective: str = "score",
        maximize: bool = True,
    ) -> None:
        evaluation = candidate.evaluation
        if evaluation is None or parent.evaluation is None:
            raise EvaluationError("Reflection requires completed evaluations")
        record = {
            "parent_id": parent.id,
            "candidate_id": candidate.id,
            "parent": parent.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "objective": objective,
            "maximize": maximize,
        }
        if evaluation.valid:
            if not parent.evaluation.valid:
                raise EvaluationError("Comparison parent must have a valid evaluation")
            try:
                left, right = parent.evaluation.metrics[objective], evaluation.metrics[objective]
            except KeyError as exc:
                raise EvaluationError(f"Missing reflection objective {objective!r}") from exc
            if left == right:
                return
            improved = right > left if maximize else right < left
            worse, better = (parent, candidate) if improved else (candidate, parent)
            record.update(worse_id=worse.id, better_id=better.id)
            operation, args = self.reflect_pair, (worse, better, objective)
        else:
            if not evaluation.feedback.strip():
                return
            operation, args = self.reflect_failure, (parent, candidate)
        self.attempts.append(record)
        try:
            text = await operation(*args, provider=_RecordedProvider(self.provider, record))
            record["text"] = text
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        self.records.append(record)
        self.records[:] = self.records[-self.max_items :] if self.max_items else []

    @prompt(template="rsikit/prompts/reflection/pair.j2")
    async def reflect_pair(self, worse, better, objective, *, generated: str) -> str:
        if not generated.strip():
            raise ValueError("Reflection must not be blank")
        return generated

    @prompt(template="rsikit/prompts/reflection/failure.j2")
    async def reflect_failure(self, parent, candidate, *, generated: str) -> str:
        if not generated.strip():
            raise ValueError("Reflection must not be blank")
        return generated
