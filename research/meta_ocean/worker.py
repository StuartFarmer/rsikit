"""Container-only JSON protocol. No simulator, run directory or model credentials."""

import asyncio
import json
import os
import sys
import traceback

import gymnasium as gym
import numpy as np

from rsikit.policy import load_policy

MAX_MESSAGE = 8 * 1024 * 1024
INPUT = sys.stdin
OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def send(value):
    data = json.dumps(value, allow_nan=False)
    if len(data.encode()) > MAX_MESSAGE:
        raise ValueError("Worker message exceeds 2 MiB")
    OUTPUT.write(data + "\n")
    OUTPUT.flush()


def receive():
    data = INPUT.readline(MAX_MESSAGE + 1)
    if not data or len(data.encode()) > MAX_MESSAGE:
        raise ValueError("Missing or oversized worker message")
    return json.loads(data)


def oracle(op, value):
    send(dict(op=op, value=value))
    return receive()


async def policy_loop(request):
    obs = request["observation_space"]
    observation_space = gym.spaces.Box(
        np.array(obs["low"], dtype=obs["dtype"]),
        np.array(obs["high"], dtype=obs["dtype"]),
        dtype=obs["dtype"],
    )
    action_space = gym.spaces.MultiDiscrete(request["nvec"])
    policy = load_policy(request["source"], observation_space, action_space, "")
    await policy.reset(seed=0)
    send(dict(ready=True))
    while True:
        request = receive()
        action = await policy.act(np.array(request["observation"], dtype=obs["dtype"]))
        send(dict(action=np.asarray(action).tolist()))


def main():
    request = receive()
    if request["mode"] == "policy":
        asyncio.run(policy_loop(request))
    elif request["mode"] == "evolver":
        asyncio.run(evolver_loop(request))
    else:
        namespace = {"__name__": "controller"}
        exec(compile(request["source"], "controller.py", "exec"), namespace)
        namespace["search"](
            **request["context"],
            generate=lambda text: oracle("generate", text),
            evaluate=lambda source: oracle("evaluate", source),
            commit=lambda identifier: oracle("commit", identifier),
        )
        send(dict(op="done"))


async def evolver_loop(request):
    context = request["context"]
    evolver = load_policy(
        request["source"], gym.spaces.Dict({}), gym.spaces.Discrete(1), context["task"]
    )

    async def generate(prompts):
        reply = oracle("generate", prompts)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["responses"]

    evolver.generate = generate
    await evolver.reset(seed=context["seed"])
    results = []
    for generation in range(1, context["generations"] + 1):
        population = await evolver.act(dict(context, generation=generation, results=results))
        reply = oracle("population", dict(generation=generation, policies=population))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        results = reply["results"]
    send(dict(op="done"))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        send(dict(error=traceback.format_exc()[-8192:]))
