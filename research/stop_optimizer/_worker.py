"""Container-only improver entry point; host validates every capability request."""

import asyncio
import json
import os
import sys

MAX_FRAME = 1024 * 1024


def main():
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def send(message):
        frame = json.dumps(message, allow_nan=False)
        if len(frame.encode()) + 1 > MAX_FRAME:
            raise ValueError("Worker output exceeds 1 MiB")
        print(frame, file=protocol, flush=True)

    def receive():
        frame = sys.stdin.buffer.readline(MAX_FRAME + 1)
        if not frame or len(frame) > MAX_FRAME:
            raise ValueError("Missing or oversized host frame")
        return json.loads(frame)

    class Capabilities:
        def __init__(self, generations, evaluations):
            self.generations_left, self.evaluations_left = generations, evaluations
            self.request_id = 0

        async def request(self, method, source, guidance=""):
            request_id = self.request_id
            self.request_id += 1
            send(dict(id=request_id, method=method, source=source, guidance=guidance))
            response = receive()
            if response.get("id") != request_id:
                raise ValueError("Mismatched host response")
            if "error" in response:
                raise RuntimeError(response["error"])
            return response["result"]

        async def suggest(self, source, guidance=""):
            self.generations_left -= 1
            return await self.request("suggest", source, guidance)

        async def evaluate(self, source):
            self.evaluations_left -= 1
            return await self.request("evaluate", source)

    try:
        request = receive()
        namespace = {"__name__": "stop_generated_improver"}
        exec(compile(request["source"], "<improver>", "exec"), namespace)
        result = asyncio.run(
            namespace["improve"](
                request["initial"], Capabilities(request["generations"], request["evaluations"])
            )
        )
        if not isinstance(result, str):
            raise ValueError("Improver must return source text")
        send(dict(result=result))
    except BaseException as exc:
        send(dict(error=f"{type(exc).__name__}: {str(exc)[:4000]}"))


if __name__ == "__main__":
    main()
