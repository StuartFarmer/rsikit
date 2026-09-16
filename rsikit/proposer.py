"""Abstract proposal contract and its Slick implementation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError, field_validator
from slick import prompt
from slick.providers import Provider

from .edits import InvalidCandidate, ProposalRejected, validate_source
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


class Draft(BaseModel, extra="forbid"):
    """A generated idea and its complete replacement source."""

    description: str
    source: str

    @field_validator("description", "source")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Draft fields must not be blank")
        return value


def recent_context(history: tuple[Candidate, ...], *, limit: int = 8) -> tuple[Candidate, ...]:
    return history[-limit:] if limit else ()


class _RecordedProvider:
    """Capture raw API outcomes before generated-output parsing, including cancellation."""

    def __init__(self, provider, record):
        self.provider, self.record = provider, record

    async def acall(self, context, *, tools=None, tool_results=None):
        call = {"prompt": context}
        self.record.setdefault("calls", []).append(call)
        try:
            response, requests = await self.provider.acall(
                context, tools=tools, tool_results=tool_results
            )
            call["response"] = response
            if requests:
                raise ValueError("Prompt operations require text, not tool requests")
            return response, requests
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise


class PromptProposer(Proposer):
    """Operation-aware generation with editable text and explicit, independent context.

    Configure Slick's root to the checkout root once. The strategy callback stays
    (parent, history); a caller closure may additionally pass strategy.context.
    One operation at a time per instance. Instructions are rendered as plain text.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        *,
        instructions: Mapping[str, str] | None = None,
        history_size: int = 8,
    ):
        self.task, self.provider = task, provider
        self.instructions, self.history_size = dict(instructions or {}), history_size
        self.records: list[dict] = []
        self.thoughts: dict[int, str] = {}
        self.diagnoses: dict[int, str] = {}

    async def __call__(
        self, parent: Candidate, history: tuple[Candidate, ...], *, context: Mapping | None = None
    ) -> str:
        context = context or {}
        operation = context.get("operation", "mutate")
        parents = context.get("parents", (parent,))
        inspirations = context.get("inspirations", ())
        instruction = self.instructions.get(operation, "")
        record = {
            "attempt": len(self.records),
            "candidate_id": len(history),
            "operation": operation,
            "parent_ids": [c.id for c in parents],
            "inspiration_ids": [c.id for c in inspirations],
            "island": context.get("island"),
            "instruction": instruction,
        }
        self.records.append(record)
        data = {
            "instruction": instruction,
            "parents": [self._describe(c) for c in parents],
            "inspirations": [self._describe(c) for c in inspirations],
            "outcomes": [
                c.model_dump(mode="json") for c in recent_context(history, limit=self.history_size)
            ],
            "guidance": context.get("guidance", ()),
            "evidence": context.get("evidence", ()),
        }
        provider = _RecordedProvider(self.provider, record)
        operations = {
            "mutate": self.mutate,
            "INIT": self.initialize,
            "E1": self.explore_diverse,
            "E2": self.explore_shared,
            "M1": self.modify_structure,
            "M2": self.tune_settings,
            "M3": self.simplify,
            "alphaevolve": self.use_inspirations,
        }
        try:
            if operation == "dgm-archive":
                diagnosis = await self.diagnose(data, provider=provider)
                record["diagnosis"] = diagnosis
                self.diagnoses[len(history)] = diagnosis
                draft = await self.modify(data, diagnosis, provider=provider)
            else:
                draft = await operations[operation](data, provider=provider)
            record["draft"] = draft.model_dump(mode="json")
            self.thoughts[len(history)] = draft.description
            return draft.source
        except ValidationError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            call = record.get("calls", [{}])[-1]
            if "error" in call:
                raise  # Provider-side schema errors are infrastructure failures.
            raw = call.get("response", "")
            raise ProposalRejected(raw, f"Invalid draft: {exc}") from exc
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _describe(self, candidate):
        return {
            **candidate.model_dump(mode="json"),
            "description": self.thoughts.get(candidate.id, ""),
        }

    async def repair(self, source: str, evaluation: Evaluation) -> str:
        record = {
            "attempt": len(self.records),
            "operation": "repair",
            "source": source,
            "evaluation": evaluation.model_dump(mode="json"),
        }
        self.records.append(record)
        try:
            return await self.fix(
                source, evaluation, provider=_RecordedProvider(self.provider, record)
            )
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    @prompt(template="rsikit/prompts/operations/mutate.j2", output_type=Draft)
    async def mutate(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/initialize.j2", output_type=Draft)
    async def initialize(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/diverse.j2", output_type=Draft)
    async def explore_diverse(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/shared.j2", output_type=Draft)
    async def explore_shared(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/structure.j2", output_type=Draft)
    async def modify_structure(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/settings.j2", output_type=Draft)
    async def tune_settings(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/simplify.j2", output_type=Draft)
    async def simplify(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/inspirations.j2", output_type=Draft)
    async def use_inspirations(self, data, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/diagnose.j2")
    async def diagnose(self, data, *, generated: str) -> str:
        if not generated.strip():
            raise ProposalRejected(generated, "Blank diagnosis")
        return generated

    @prompt(template="rsikit/prompts/operations/modify.j2", output_type=Draft)
    async def modify(self, data, diagnosis, *, generated: Draft) -> Draft:
        return generated

    @prompt(template="rsikit/prompts/operations/repair.j2")
    async def fix(self, source, evaluation, *, generated: str) -> str:
        return generated
