# Inner-loop API

`Evaluator` runs one episode using an existing Gymnasium environment and an
existing async `Policy`. The caller creates, seeds, resets, and closes both
instances, and is responsible for not reusing stateful objects accidentally.
For generated policies and durable batch evaluations, see [Run](RUNS.md).

`rsikit.evaluation.Evaluator` collects the rollout and returns an
`rsikit.episode.Episode`. Both are also exported directly from `rsikit`.
Fitness and screening belong to the research optimizers. AlphaEvolve owns its
`EvaluationResult`; core execution returns raw episodes.

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

`run_program` creates, resets, and closes its instances and returns an `Episode`.
The source must export `Solution(Policy)`. It reads source as data and executes
it in Docker; never import generated source on the host. Its environment remains
on the host. One remote policy instance and asyncio event loop persist throughout
the episode. For execution with both instances in Docker, use `Executor`; see
[InProcessDockerSandbox](IN_PROCESS_SANDBOX.md).

Build the worker with:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
```

The worker runs non-root, without network or host mounts, with a read-only
filesystem and resource limits. `call_timeout` defaults to 10 seconds per policy
call in `run_program`. This is research isolation, not a hostile multi-tenant
service. Direct trusted policies have no hard execution deadline.

The isolated policy protocol supports Box, Discrete, Dict, Tuple, and Text spaces
with bounded numeric array and JSON payloads. Unsupported spaces fail before
source execution; direct evaluation can use other Gymnasium spaces. This policy
channel has no pickle transport or fallback to local execution.

## Failures

Actions outside `action_space` raise `PolicyError` before `env.step()`.
Environments should raise `gymnasium.error.InvalidAction` for state-dependent
illegal actions; the evaluator converts this to `PolicyError`. Other environment
and trusted-policy exceptions retain their types. Cancellation propagates.
Failed execution returns no normal `Episode`.

The caller must clean up after `Evaluator.run()`, including on failure. The
sandbox helpers perform their own cleanup and log secondary cleanup
failures without replacing the original exception. Sandbox failures raise
`PolicyError`, `PolicyTimeout`, or `InfrastructureError` from `rsikit.evaluation`.

Termination does not imply success: rewards and task-specific info define that.
Search and cross-episode aggregation remain outside the evaluator. `Run` persists
supplied scores, checkpoints, and raw episodes through `save_episode`/`load_episode`.
