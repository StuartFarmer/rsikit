# Inner-loop API

`Evaluator` runs one episode using an existing Gymnasium environment and an
existing async `Policy`. The caller creates, seeds, resets, and closes both
instances, and is responsible for not reusing stateful objects accidentally.
For generated policies and durable batch evaluations, see [Run](RUNS.md).

`rsikit.evaluation.Evaluator` collects the rollout and returns an
`rsikit.episode.Episode`. Both are also exported directly from `rsikit`.
Episode execution returns raw trajectories. Search evaluators report neutral
`Measurement` values; optimizers interpret them. AlphaEvolve constructs its own
`EvaluationResult` inside `update()`.

```python
from copy import deepcopy

import gymnasium as gym
from rsikit import Evaluator, Policy


class RandomPolicy(Policy):
    async def act(self, observation):
        return self.action_space.sample()


with gym.make("LunarLander-v3", render_mode="human") as env:
    policy = RandomPolicy(deepcopy(env.observation_space), deepcopy(env.action_space))
    try:
        observation, info = env.reset(seed=42)
        await policy.reset(seed=42)
        evaluator = Evaluator(env, policy, max_steps=1000)
        episode = await evaluator.run(observation, info=info)
    finally:
        await policy.close()

print(episode.total_reward, len(episode))
observation, reward, terminated, truncated, info = episode.final_step
```

The code above runs inside an async function or a notebook supporting top-level
`await`. `run()` does not call `reset()` or `close()`, even on failure or
cancellation. Pass the observation returned by the caller's reset; supplying its
`info` is optional. The evaluator has no seed or factory parameters. It runs the
supplied objects directly and does not provide a sandbox.

Existing Gymnasium episode limits apply. `max_steps` is an optional positive
integer that marks the last recorded transition as truncated when the rollout
reaches that many steps. It neither resets nor wraps the supplied environment,
and cannot extend its existing limit. Like Gymnasium's `TimeLimit`, it can set
truncation on the same step as termination. Use a `TimeLimit` wrapper before
resetting if other environment wrappers also need to observe that limit.

## Episode histories

`run()` returns an `Episode`, with these ordinary Python lists:

| Field | Length after T steps | Meaning |
| --- | --- | --- |
| `observations` | T + 1 | Initial observation, then each step's next observation |
| `actions` | T | Validated actions passed to `env.step()` |
| `rewards` | T | Reward for each action |
| `terminations` | T | Task termination flags |
| `truncations` | T | Truncation flags, including the evaluator's step cap |
| `infos` | T + 1 | Initial info (or `{}`), then each step's diagnostic info |

Transition `t` is:

```python
observation = episode.observations[t]
action = episode.actions[t]
reward = episode.rewards[t]
next_observation = episode.observations[t + 1]
terminated = episode.terminations[t]
truncated = episode.truncations[t]
info = episode.infos[t + 1]
```

`episode.total_reward` is the undiscounted sum of rewards; `len(episode)` is the
number of transitions. `episode.final_step` provides the final Gymnasium tuple,
whose reward is only the last step's reward. No `info["episode"]` key is added by
the evaluator; use the summary properties or wrap the environment yourself.

Observations, actions, and infos are deep-copied so reused arrays or dictionaries
cannot rewrite earlier transitions. These values must support deep copying.
The result remains valid after caller cleanup. Full histories occupy memory
proportional to episode length and observation/info size, including image data.

This follows established RL practice: [RLlib's SingleAgentEpisode](https://docs.ray.io/en/latest/rllib/single-agent-episode.html)
retains an initial observation plus actions, rewards, subsequent observations,
and infos. [Stable-Baselines3 buffers](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
store transitions for off-policy learning and rollouts for PPO/A2C. PPO also needs
policy log probabilities and value estimates; this generic episode history does
not supply those algorithm-specific outputs or implement a replay buffer.

Keep termination and truncation separate: value targets generally bootstrap at
a time-limit truncation but not at a terminal state. See
[Gymnasium's time-limit explanation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
Histories also support reward plots, action analysis, and diagnosing failures.

## Policy contract

Construct the policy with observation/action spaces and optional `instructions`.
Copy the spaces if the policy should have independent sampling state. Inherited
`await policy.reset(seed=...)` seeds `self.rng` and `self.action_space`; overrides
should call it and clear their own episode memory. `close()` is an empty hook by
default. Supply instructions when constructing the policy; `Evaluator` does not
read or override them.

Only copied observations reach `act`; diagnostic info stays in the episode.
Goals and feedback the policy needs during the episode belong in observations.
No weights are trained by the evaluator.

## Generated programs

```python
from pathlib import Path
from rsikit import run_program

episode = await run_program(Path("solution.py"), "CartPole-v1", max_steps=100)
print(episode.total_reward, episode.actions)
```

`run_program` creates an environment template, then runs reset, policy execution,
steps and scoring in a fresh local child. The source must export `Solution(Policy)`.
`env_seed`, `policy_seed`, `instructions`, and `max_steps` apply inside that child.
The episode deadline defaults to 60 seconds and includes policy loading, reset,
actions, cleanup and result preparation. Direct `Evaluator` calls have no deadline.

Run application modules through `scripts/run MODULE [ARGS...]` to isolate the whole
application in Docker. Calling this Python API directly uses local processes.
Generated code shares application credentials, network, outputs and scoring state.
See [application execution](IN_PROCESS_SANDBOX.md).

Caller-provided environment templates use cloudpickle to reach the child; its
imports must be installed in the application image. Complete `Episode` objects
return over a multiprocessing connection, with a 64 MiB result ceiling. Saved
JSON serialization lives with `Episode` and retains the existing representation.

## Failures

Actions outside `action_space` raise `PolicyError` before `env.step()`.
Environments should raise `gymnasium.error.InvalidAction` for state-dependent
illegal actions; the evaluator converts this to `PolicyError`. Other environment
and trusted-policy exceptions retain their types. Cancellation propagates.
Failed execution returns no normal `Episode`.

The caller must clean up after `Evaluator.run()`, including on failure. The
execution helpers perform their own cleanup and log secondary cleanup
failures without replacing the original exception. Child execution failures raise
`PolicyError`, `PolicyTimeout`, or `InfrastructureError` from `rsikit.evaluation`.

Termination does not imply success: rewards and task-specific info define that.
Search and cross-episode aggregation remain outside the evaluator. `Run` persists
supplied scores, checkpoints, and raw episodes through `save_episode`/`load_episode`.

## One optimization loop

All three AlphaEvolve variants, ShinkaEvolve, EliteSearch, and LineageSearch
implement the structural `rsikit.Optimizer` protocol:

```python
from collections.abc import Mapping
from rsikit import Measurement, Policy


async def propose() -> list[type[Policy]]: ...
def update(results: Mapping[str, Measurement]) -> None: ...


# Read-only properties: done: bool; best: type[Policy] | None
```

Configure an optimizer using its algorithm's `Config`, then pass the same evaluator
function and runner to any implementation:

```python
from rsikit import Measurement, search
from research.rewards import measure_rewards


# rollouts is a caller-owned Rollouts(environment, executor, run).
async def evaluate(policies):
    return await measure_rewards(rollouts, policies, seeds=(0, 1, 2))


best = await search(optimizer, evaluate)
```

An evaluator object's bound method works too: `await search(optimizer, evaluator.evaluate)`.
No evaluator superclass is required. `search` returns the optimizer's best policy
definition, or `None`; histories and domain records remain on the optimizer and Run.
The caller owns environment, provider, evaluator, and database cleanup.

`Measurement({0: 0, 1: 10})` retains the individual seed scores. It is different
from `Measurement({0: 5, 1: 5})`: both means are five, but their variability differs.
Scores must be finite numbers, with integer seed keys. Use one fixed search panel;
keep validation and test panels out of optimizer feedback. All objectives maximize.
Named `metrics` and `features` carry independently measured evidence; their
aggregation and interpretation belong to the optimizer. Accepted feedback needs
scores or metrics, plus the evidence required by the chosen algorithm.

Return exactly one measurement per proposed policy ID, including failures:

```python
results = {
    good.id: Measurement({0: 3, 1: 7}, feedback="Completed both episodes"),
    broken.id: Measurement(failure="Invalid action"),
    screened.id: Measurement(accepted=False, feedback="Below screening threshold"),
}
optimizer.update(results)
```

A failure queues bounded repair; a screening rejection does not. The next
`propose()` generates replacements only for failed candidates. Repairs retain
original attempt identities and do not consume a new population generation.
Duplicate internal attempts may share a measurement, but returned proposal IDs
are unique. Missing, extra, repeated, or incompatible feedback raises before
selection changes. Calling `propose()` while feedback is outstanding raises.

The optimizer chooses round sizes and stopping limits. A round can be a batch,
a generation, a family sweep, or queued repairs. `update()` is synchronous and
performs no evaluation or model calls. An empty proposal list means completion;
`search` raises if an optimizer returns an empty list while `done` is false.
For manual orchestration, the same contract is:

```python
while not optimizer.done:
    policies = await optimizer.propose()
    if policies:
        optimizer.update(await evaluate(policies))
```

The shared runner also validates batches and invokes optional `on_checkpoint`
after proposals, after updates, and on exceptional exit. Algorithms keep any
additional checkpoints needed during generation. Cancellation, infrastructure
errors, and provider budget exhaustion propagate; they are interrupted outcomes.

### Scheduling and migration

New search manifests record `optimization_schedule: round-v1`. Generation
finishes before evaluation, and evaluation finishes before update. Model calls
within proposal generation and episodes within evaluation remain concurrent.
This replaces the previous paper AlphaEvolve streaming pipeline and EliteSearch
cross-stage overlap. It can change throughput and search trajectories; there is
no claim of benchmark equivalence.

Replace `propose(n)` with a configured batch size and `propose()`. Replace
policy–Episode pairs or scalar-score mappings with policy-ID–`Measurement`
mappings. `research.rewards.Measurement` re-exports the same core class.
EliteSearch/LineageSearch `run()` and historical application entry points remain
thin compatibility wrappers over `rsikit.search`; they retain their legacy
return shapes. No `fit()` method or alternate optimization loop is needed.
