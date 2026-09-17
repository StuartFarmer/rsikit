# RSIKit Gymnasium inner loop

Status: implemented; reflects the accepted Policy/instructions design.
Date: 2026-09-17.

## Scope

Build the contract between a trusted task environment and an executable policy,
plus one reusable episode runner. Implement circle packing and demonstrate an
unmodified CartPole environment. Do not redesign the adaptive agent, generation,
selection, mutation, crossover, islands, archives, or cross-episode memory yet.
The user explicitly permits deleting or rewriting the old API and breaking old
tests; compatibility is not an acceptance criterion. Implementation was subsequently authorized by the user.

## Design contract

An environment is an ordinary `gymnasium.Env`. There is no RSIKit environment
base class, custom environment registry, or mandatory evaluator method. Tasks
define observations, actions, rewards, and terminal conditions using Gymnasium.
Spaces describe representation; task instructions explain field meanings,
objectives, constraints, and interaction rules. RSIKit task environments expose
canonical instructions through an `instructions: str` attribute. This is an
RSIKit convenience, not an addition to Gymnasium's required environment API.

Use `Policy` for the executable interface. A stored program exports a `Solution`
policy class. Reserve `model` for an underlying trained model/API service;
"organism" remains optional vocabulary for individuals in evolutionary search.

A policy is a Python class implementing the following interface. A small
`Policy` base supplies spaces, task instructions, a seeded NumPy generator, and
default cleanup.

```python
class Policy:
    def __init__(self, observation_space, action_space, *, instructions: str): ...
    async def reset(self, *, seed: int | None = None) -> None: ...
    async def act(self, observation): ...
    async def close(self) -> None: ...
```

The generated module exports `Solution`, a subclass of `Policy`. It can define
private helpers and imports available in the execution image. Do not require
exactly one class in the module or AST-restrict its algorithm. Source is the
stored program; a fresh instance is the policy executing one episode.
Source/interface checks do not replace isolated execution. The worker image exposes
`rsikit.policy.Policy` so local and isolated Solution modules use the same import.

The constructor stores `self.observation_space`, `self.action_space`, and
`self.instructions`. Space access lets reusable policies inspect output dimensions
or sample valid actions; a specialized policy need not inspect them. Space
membership checks remain the runner's responsibility even if a policy uses them.
Do not build a service container or pass a live environment into the policy.
Asynchronous `reset`, `act`, and `close` give local and remote policies the same
lifecycle without blocking the runner. This refines the earlier synchronous
reset/close sketch. Asynchronous `act` also preserves a natural
future boundary for fixed-model API calls, but network/model-service injection
is outside this first implementation. Current offline sandbox networking stays
disabled; adding an API gateway will require a subsequent, explicit design.

Instance state persists between actions and is discarded between episodes.
The runner creates both environment and policy through factories, resets them,
and closes them even when execution fails. `reset` initializes the policy RNG;
subclasses also clear their episode memory. Environment, policy, and space
sampling seeds are recorded separately or deterministically derived. Environment
reset must use `super().reset(seed=seed)` and its `np_random` generator.

After construction, the runner passes only observations to `act`. It preserves
`info` for evidence but never automatically sends it into `act`. Task feedback necessary for acting,
including prior reward when relevant, belongs in the declared observation.
No training hook, `learn`, `update`, or mandatory transition callback is added.

## Task instructions and policy prompts

Both runner entry points accept keyword-only `instructions: str | None = None`.
An explicit string takes precedence, including an intentionally empty string.
Otherwise, read `env.unwrapped.instructions` before wrapping the fresh environment
with TimeLimit. If the attribute is absent, raise a configuration error asking
the caller to supply instructions; never infer the task from a Python docstring,
source code, or `info`. Existing Gymnasium environments remain unchanged: the
CartPole example passes its own instructions. Wrappers that change task semantics
must supply matching instructions explicitly instead of using the unwrapped fallback.

Resolve the text once per episode, preserve it exactly, store it on the episode
record, and pass it as `instructions=resolved_instructions` to the policy factory.
For isolated execution, include exactly that text in the worker's start request.
The string is configuration held by the trusted caller; a policy changing its own
copy cannot change environment rules, rewards, or the recorded task definition.

There are two uses for the same public task description:

- Generation context: a future adaptive agent supplies these instructions, space
  descriptions, the Policy interface, and allowed dependencies to its generator.
  This plan defines the data boundary but does not implement generation.
- Runtime context: a policy receives the instructions at construction. A numerical
  heuristic may ignore them; a future LLM-backed policy may include them in model
  requests. Model clients and the sandbox API gateway remain outside this slice.

Task instructions define what to accomplish. A policy's own prompt defines how
it approaches that task and is part of the program that can later evolve. Do not
merge these into one mutable prompt or automatically interpret task instructions
as a provider system message. Episode-varying goals and visible feedback belong
in observations, not in mutable constructor instructions. Prompts/instructions
contain only public task information, never held-out answers or hidden state.

## Public entry points

```python
async def run_episode(
    make_env,
    make_policy,
    *,
    env_seed: int,
    policy_seed: int,
    max_steps: int,
    instructions: str | None = None,
) -> Episode: ...

async def run_program(
    program: Path,
    make_env,
    *,
    env_seed: int,
    policy_seed: int,
    max_steps: int,
    instructions: str | None = None,
    call_timeout: float = 10.0,
) -> Episode: ...
```

`make_env()` returns a fresh Gymnasium environment. `make_policy(obs_space,
action_space, instructions=resolved_instructions)` returns a fresh trusted policy.
A concrete Policy subclass can serve as this factory. `run_episode` is the trusted
local path and contains the only rollout algorithm. `run_program` reads source as
data, starts the isolated policy client, and delegates to `run_episode`.
The isolated client implements the same async policy methods: `reset` starts
the worker, `act` exchanges one request, and `close` shuts it down. No generated
module is imported by the host.

`run_episode` wraps the environment with Gymnasium `TimeLimit(max_steps)` rather
than synthesizing a terminal transition. Existing environment time limits remain
effective. A finite-horizon task can still terminate according to its own rules.

Episode records contain:

- `env_seed`, `policy_seed`, initial observation and reset info (absent if reset fails);
- `instructions`: the exact resolved public task instructions;
- transitions with observation, action, next observation, reward, both ending
  flags, and info, copied when recorded to avoid later array mutation;
- `status`: `completed`, `policy_error`, `invalid_action`, or `timeout`;
  partial evidence attached to evaluator errors starts as `incomplete`;
- an optional failure message and attempted action when available, plus cleanup errors.

`return_` and `length` are derived from completed transitions. A failed episode's
partial return is evidence, not a rankable fitness score. Completion includes
ordinary termination and truncation; neither implies task success. No universal
success threshold, metric aggregation, or conversion to old `Evaluation` records.
Environment contract violations and infrastructure failures raise typed errors
carrying the partial episode, rather than being scored as policy failures.
Cancellation propagates after cleanup. Cleanup errors must not erase the original
failure. Validate action membership before stepping and finite scalar rewards
after stepping; an invalid environment observation is an environment error.

## Execution boundary

Keep the trusted environment and reward computation on the host. Evolve the
existing Docker backend from a one-function request into one persistent policy
per episode. Start a new container per episode for this first version; remove the
fork-per-action design because it would erase policy memory.

Retain non-root execution, disabled networking, no host mounts/credentials,
read-only filesystem, resource limits, bounded messages, and forced cleanup.
The worker loads source, constructs `Solution`, resets it, services sequential
actions using one persistent asyncio event loop, and closes it on shutdown.
Keep candidate stdout/stderr separate from the protocol channel. All constructor,
reset, action, and close execution is deadline-bounded by the host, which can kill
the container even if generated Python blocks the event loop. Container startup
failures remain infrastructure errors; generated code exceptions are policy
failures. A container is not a hostile multi-tenant security guarantee.

Use a small explicit JSON codec, never pickle or generated deserializers. Initially
support `Box`, `Discrete`, `Dict`, `Tuple`, and `Text`, preserving array dtype and
shape, dictionary keys, tuples, text charset/length, discrete start, and infinite
Box bounds. Tag nonfinite float values explicitly rather than emitting invalid
JSON. Restrict array dtype decoding to ordinary numeric/boolean types, and bound
message size and allocation. Unsupported spaces fail before launching a program.
The local runner remains usable with other Gymnasium spaces; isolated execution
claims support only for the documented subset. Codec errors in task configuration
are infrastructure errors; malformed candidate actions are candidate failures.

## First problems

`CirclePackingEnv(count=10)` owns all task rules in one class. Its `instructions`
describe the observation fields, `(x, y, radius)` action rows, positive radii,
unit-square containment, non-overlap, sum-of-radii objective, `1e-9` tolerance,
infeasible reward, and one-submission episode. Its observation is
a `Dict` containing the unit-square bounds and requested count. The count is fixed
for the lifetime of an environment instance, so spaces do not change on reset.
The action is a float64 `Box` of shape `(count, 3)`, with coordinates in `[0, 1]`
and radius in `[0, 0.5]`. The environment checks positive radii, containment, and
non-overlap with the existing `1e-9` tolerance. A feasible action receives the sum
of radii; infeasible but in-space geometry receives `0.0`. Both end the one-step
episode with `terminated=True`. Info records feasibility, violations, and raw
measurements. Terminal observation remains inside observation_space.

The baseline places ten radius-0.1 circles on a five-by-two grid and scores 1.0.
Its `Solution.act` returns an ndarray, not source text or a serialized result file.
Rendering is not required for the first environment; preserve reusable plotting
math but remove generation-history assumptions from the active example.

The second example uses `gym.make('CartPole-v1')` directly and supplies instructions
describing the four state coordinates, left/right actions, balance objective,
and standard reward/ending rules. A simple hand-written
`Solution` chooses a direction from pole angle/angular velocity. Performance is
not an acceptance requirement: completed multi-step interaction and preserved
policy state are. Both examples use exactly the same runner.

## Constraints and explicit deferrals

- Python >=3.10; retain the existing Ruff conventions.
- Add Gymnasium >=1.3,<2 and an explicit NumPy dependency >=1.24,<3.
- No LLM weight training or fine-tuning.
- Implementation authorized after the planning turn; this design is now implemented.
- No backwards-compatibility adapters for removed task functions or evaluator scripts.
- Do not weaken the existing isolation boundary to speed up the rewrite.
- Use small runnable smoke checks; comprehensive test hardening is a later phase.

No vectorization, parallel episodes, adaptive-agent integration, aggregate fitness,
API model gateway, persistent cross-episode memory, experiment database, rendering
framework, or universal artifact/plugin registry in this slice. Do not delete
useful outer-loop modules merely because they are outside this scope; remove their
root exports and document that integration awaits the inner-loop contract.

## Sources

Gymnasium defines the environment contract and spaces used here:
[Env](https://gymnasium.farama.org/api/env/),
[spaces](https://gymnasium.farama.org/api/spaces/), and
[custom environments](https://gymnasium.farama.org/introduction/create_custom_env/).
The runner follows [basic usage](https://gymnasium.farama.org/introduction/basic_usage/)
and uses [TimeLimit](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit).
Check environment conformance with [check_env](https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env).
[CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
supplies the existing multi-step task. The policy interface, isolation protocol,
record format, and packing formulation above are RSIKit design proposals.
