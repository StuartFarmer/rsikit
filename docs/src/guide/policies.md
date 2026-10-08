# Policies and evaluation

## Source and runtime objects

Subclass `rsikit.Policy` for a live controller. Implement `async act(observation)`
and return an action inside the environment's `action_space`. Override
`async reset(seed=...)` to clear episode memory, calling
`await super().reset(seed=seed)` to initialize the policy RNG and seed its action
space. Release owned resources in `async close()`.

A `PolicyDefinition` stores Python source. Its static contract requires exactly one
top-level `Solution` class; worker loading checks that it subclasses `Policy`.
Prefer the inherited constructor;
if overridden, it must accept `(observation_space, action_space, *, instructions="")`.
The definition's `name` is display metadata, not the class name to instantiate.

```python
from rsikit import PolicyDefinition

policy = PolicyDefinition.from_file("winner.py")
policy.validate()
policy.to_file("winner-copy.py")
```

Loading and saving do not execute source. `validate()` checks syntax, the
`Solution` declaration, and constructor compatibility; it cannot prove correct
behavior or safe execution. Construction deliberately permits invalid Python so
an optimizer can retain it for repair. Identity hashes the exact source and name;
changing either changes the ID, while changing the description does not.

## Choose an evaluation interface

| Interface | Inputs | Result | Ownership |
| --- | --- | --- | --- |
| `evaluate` / `Evaluator.evaluate` | Live policy and environment | One `Episode` | Resets and mutates caller-owned instances; caller closes both. |
| `execute` | Iterable of `Job` objects | Original jobs with `result` attached | Opens workers for this batch, then closes them. |
| `Executor.execute` / `iterate` | Jobs inside an open async context | List / stream of completed jobs | Shares workers across batches; private input copies per job. |
| `Run.evaluate` | Definitions, environment, executor, seeds | `{policy_id: {seed: Episode}}` | Saves evidence, reuses successful episodes, records fitness. |
| `Run.mean_scores` | Same as `Run.evaluate` | `{policy_id: mean_fitness}` | Raises `PolicyError` if any requested episode failed. |

A `Job` accepts a live `Policy` or a `PolicyDefinition`. Inputs are snapshotted
with cloudpickle when enqueued, so worker changes do not update your objects.
Workers instantiate definitions, close their own policy/environment copies, and
collect artifacts. You still close the original environment and any original live
policy. Each `Job` can be submitted only once. Results are observed in readiness
order, not submission order.

Use `async with Executor(...)` for every executor. Finish evaluation coroutines
before leaving the context; use `contextlib.aclosing` around `iterate()` when
stopping consumption early. The [custom-system example](../examples/custom-system.md)
shows the full `Run` and `Executor` lifecycle.

## Seeds and trajectories

Every episode resets the environment and policy with the same seed. Use a fixed,
nonempty panel of distinct nonnegative integer seeds for comparisons. This
controls supported environment/policy randomness; it does not make external
model calls deterministic. Keep final test seeds separate from seeds used for
search or winner selection.

For `T` completed transitions, an episode has `T` actions, rewards, and ending
flags, plus `T + 1` observations and infos. The first info comes from reset.
`max_steps` forces truncation at the limit, and normal environment termination or
truncation ends the episode earlier. Failed episodes may contain only a partial
trajectory or no initial observation. Check `episode.error` before using scores
or `final_step`.

Episode storage supports ordinary nested values and real numeric/boolean NumPy
arrays, not arbitrary Python objects. Arrays are limited to 256 KiB each; worker
results and saved episodes are limited to 64 MiB. Environments and live policies
must be cloudpickle-serializable for worker execution.

## Failure and cancellation

Policy reset/action errors, invalid actions, and worker policy-loading errors
produce an `Episode` with `error` set. Worker episode timeouts also produce failed
episodes. Unexpected environment, serialization, queue, or worker failures raise
exceptions; infrastructure failures must not be turned into low policy scores.
Streaming execution can return successful siblings before raising an infrastructure
error. `Run.collect` persists completed episodes as they arrive.

Direct evaluation has no process deadline. `Executor.episode_timeout` applies to
worker episodes, including definition loading. Cancellation propagates;
queued work is revoked while already-running work finishes under the worker
timeout. Cleanup remains the caller's responsibility.

See [policy reference](../api/policies.md),
[evaluation reference](../api/evaluation.md), [worker execution](../api/execution.md),
and [run reuse](runs.md).
