# Inner-loop API

A task is a regular Gymnasium environment. A program exports `Solution(Policy)`.
An episode connects a fresh policy instance to a fresh environment, records the
interaction, then closes both. No weights are trained and no optimizer runs here.

```python
from rsikit import Policy


class Solution(Policy):
    async def reset(self, *, seed=None):
        await super().reset(seed=seed)
        self.actions_taken = 0

    async def act(self, observation):
        self.actions_taken += 1
        return self.action_space.sample()
```

The inherited constructor accepts observation/action spaces and keyword-only
`instructions`. Spaces describe representation; instructions explain field
meanings, goals, constraints, rewards, and interaction rules. A mathematical policy
may ignore prose. A future LLM-backed policy may include it in requests to an
already-trained model. An organism is an evolutionary individual; Policy is the
executable interface, and model refers to the underlying model service.

`reset`, `act`, and `close` are asynchronous. Reset seeds `self.rng` and the
policy's private action space; subclasses clear their own memory. Each episode
gets a fresh instance. Close has an empty default implementation. The environment
retains Gymnasium's synchronous reset/step/close methods.

## Trusted classes

```python
import asyncio

from rsikit import run_episode
from rsikit.envs import CirclePackingEnv
from rsikit.examples.circle_packing.initial import Solution

episode = asyncio.run(
    run_episode(
        CirclePackingEnv,
        Solution,
        env_seed=1,
        policy_seed=2,
        max_steps=1,
    )
)
print(episode.status, episode.return_, episode.length)
```

`make_env()` must return a fresh environment. The policy factory receives copied
spaces and `instructions=...`; a concrete class can serve directly as the factory.
Instructions come from an explicit runner argument, otherwise from
`env.unwrapped.instructions`. Missing instructions raise a configuration error;
explicit empty text is allowed. Wrappers that change task semantics should pass
matching instructions explicitly. Gymnasium environments need no RSIKit base class
or custom methods: the CartPole example supplies its own text.

The runner records the resolved text unchanged. Only observations go to `act`;
`info` is evidence for the caller and may contain private diagnostic information.
Dynamic goals and feedback needed by the policy belong in observations. A policy's
own prompt can evolve independently of the authoritative task instructions.

## Generated programs

```python
from pathlib import Path

from rsikit import run_program

episode = await run_program(
    Path("solution.py"),
    CirclePackingEnv,
    env_seed=1,
    policy_seed=2,
    max_steps=1,
    call_timeout=10.0,
)
```

`run_program` reads source as data and executes it in Docker. Never import generated
Python before passing it to the runner. The environment, hidden state, and reward
computation stay on the host; the worker receives source, spaces, instructions,
seed, and observations. One class instance and one asyncio event loop survive
throughout an episode. An independent episode starts fresh.

The image has no network, credentials, or host mounts. It runs non-root with a
read-only filesystem and resource limits. Host deadlines stop blocked generated
code. This is a research isolation boundary, not a hostile multi-tenant
service. Local trusted policies have no hard execution deadline: blocking Python
cannot be interrupted safely by an asyncio timeout.

The isolated protocol supports Box, Discrete, Dict, Tuple, and Text spaces with
bounded numeric array and JSON payloads. Unsupported spaces fail explicitly before
program execution. The local runner can consume other Gymnasium spaces. There is
no pickle transport and no automatic fallback to executing source locally.

## Episode evidence

An `Episode` records seeds, task instructions, initial observation/reset info,
transitions, status, failure information, and any cleanup errors. Each transition
contains observation, action, next observation, reward, `terminated`, `truncated`,
and info. Values are copied to protect records from later array mutation.

`return_` sums recorded rewards; `length` counts completed transitions. A status of
`completed` can include task failure or TimeLimit truncation; it does not mean
success. In-space but overlapping circles, for example, complete with zero reward
and `info["feasible"] == False`.

Invalid actions yield `invalid_action` without calling env.step. Policy exceptions
yield `policy_error`; execution deadlines yield `timeout`. Their partial returns
must not be compared as normal fitness. No automatic retry, repair, ranking, or
cross-episode averaging occurs here.

Environment/backend failures raise `EpisodeError` with a partial `.episode`.
Its status can be `incomplete`. Caller configuration errors remain ordinary
exceptions. Cancellation propagates after cleanup. Cleanup failures are recorded
without overwriting an earlier failure.

## Development scope

The first tasks are one-step circle packing and native multi-step CartPole.
Generation, adaptive search, model API gateways, persistent cross-episode memory,
vectorization, and aggregate fitness are outside this core. The repository
contains only the inner loop and its examples/tests. The worker image provides
NumPy and Gymnasium; add other execution dependencies when a task requires them.
