# Inner-loop API

A task is a standard Gymnasium environment. RSIKit connects it to an async policy,
runs until termination or truncation, and closes both. No weights are trained.

```python
from rsikit import Policy, run_episode


class Solution(Policy):
    async def act(self, observation):
        return self.action_space.sample()


observation, reward, terminated, truncated, info = await run_episode(
    "CartPole-v1", Solution, env_seed=1, policy_seed=2
)
print(info["episode"])  # {"r": total reward, "l": number of steps, "t": seconds}
```

## Environments and results

Pass a registered environment ID, or a zero-argument factory returning a fresh
Gymnasium environment: `lambda: gym.make("FrozenLake-v1", is_slippery=False)`.
Custom problems implement the same `gym.Env` reset/step contract. No RSIKit
base class or extra environment methods are needed.

The runner uses Gymnasium's `RecordEpisodeStatistics` wrapper and returns the
final standard step tuple. `reward` is the last step's reward; total reward is
`info["episode"]["r"]`. No trajectory is stored. Leave the `episode` info key to
this wrapper; factories should not add another `RecordEpisodeStatistics` wrapper.

Existing Gymnasium time limits apply. Optional `max_steps` adds a `TimeLimit`
cap; it cannot extend a registered environment's limit. Set a cap for custom
environments that may never terminate. Seeds default to `None`.

## Policy lifecycle and instructions

The policy constructor receives copied observation/action spaces and optional
`instructions`. Each episode creates one instance, calls `await reset(seed=...)`,
then repeatedly calls `await act(observation)`, and finally `await close()`.
Inherited reset seeds `self.rng` and `self.action_space`; override it to also
clear policy memory. Close is an empty hook by default.

Spaces describe valid values. Instructions can explain the goal and observation
fields to a policy using an LLM API. Pass `instructions="..."` to the runner, or
set `instructions` on a custom environment. If neither is present, text defaults
to `""`. An explicit argument overrides the environment, including empty text.

Only copied observations reach `act`; diagnostic `info` stays with the caller.
Goals and feedback the policy needs during the episode belong in observations.

## Generated programs

```python
from pathlib import Path
from rsikit import run_program

result = await run_program(Path("solution.py"), "CartPole-v1", max_steps=100)
```

The source must export `Solution(Policy)`. `run_program` has the same environment,
seed, instruction, step-limit, and return contracts as `run_episode`. It reads
source as data and runs it in Docker; never import generated source on the host.
The environment and scoring stay on the host. One remote policy instance and
asyncio event loop persist for the episode.

Build the worker with:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
```

The worker runs non-root, without network or host mounts, with a read-only
filesystem and resource limits. `call_timeout` defaults to 10 seconds per policy
call. This is research isolation, not a hostile multi-tenant service. Trusted
local classes run directly and have no hard execution deadline.

The isolated protocol supports Box, Discrete, Dict, Tuple, and Text spaces with
bounded numeric array and JSON payloads. Other spaces fail explicitly before
source execution; the trusted runner can use other Gymnasium spaces. There is no
pickle transport or fallback to local execution.

## Failures

Exceptions propagate after cleanup. Invalid actions raise `PolicyError` before
`env.step`; sandbox failures raise `PolicyError`, `PolicyTimeout`, or
`InfrastructureError` from `rsikit.episode`. Environment and trusted-policy
exceptions retain their original types. Cancellation propagates. Secondary
cleanup failures are logged without replacing the original failure.

Termination does not imply success: task rewards and info define that. Failed
execution produces no normal episode result. Search and cross-episode aggregation live in the separate
[AlphaEvolve module](ALPHAEVOLVE.md); the runner stores no trajectory.
