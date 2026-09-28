"""A self-contained policy repair agent; callers own budgets and acceptance."""

from slick import parse, render
from slick.providers import Provider

from rsikit.generation import WORKER_LIBRARIES
from rsikit.generation.edits import Program


class SelfHealer:
    def __init__(
        self, task: str, provider: Provider, *, context: str = "", libraries: str = WORKER_LIBRARIES
    ):
        self.task, self.provider, self.context, self.libraries = task, provider, context, libraries

    async def repair(
        self, reference: str, failed: str, diagnostic: str, *, provider=None, record=None
    ) -> Program:
        schema = Program.model_json_schema()
        context = render(
            "repair.j2",
            instance=self,
            schema=schema,
            reference=reference,
            failed=failed,
            diagnostic=diagnostic,
        )
        raw, _ = await (self.provider if provider is None else provider).acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, Program)
