"""JSON worker for the optional subprocess runner; never imported by the optimizer."""

import contextlib
import json
import os
import random
import sys


def main():
    request = json.load(sys.stdin)
    snapshot = json.dumps([request["population"], request["state"]], sort_keys=True)
    try:
        namespace = {"__name__": "evox_generated_strategy"}
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            exec(compile(request["code"], "<strategy>", "exec"), namespace)
            selection = namespace["select"](
                request["population"],
                request["state"],
                random.Random(request["seed"]),
            )
        if snapshot != json.dumps([request["population"], request["state"]], sort_keys=True):
            raise ValueError("strategy modified its population or state")
        output = json.dumps({"selection": selection}, allow_nan=False)
    except Exception as exc:
        # All exceptions inside generated code are candidate failures, not host failures.
        output = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    print(output)


if __name__ == "__main__":
    main()
