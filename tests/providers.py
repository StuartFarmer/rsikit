"""One scripted provider shared by all agent tests."""

from pydantic import BaseModel


class ScriptedProvider:
    """Record requests and consume explicit responses or exceptions in order."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.calls.append(context)
        try:
            response = next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("Scripted provider exhausted") from exc
        if isinstance(response, Exception):
            raise response
        return response.model_dump_json() if isinstance(response, BaseModel) else response, []
