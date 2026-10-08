# Execution

`execute` owns workers for one batch. An `Executor` async context keeps workers
available across batches. Both accept single-use `Job` objects and attach results
to those same objects in the caller's process.

```python
import gymnasium as gym
from rsikit import Executor, Job, PolicyDefinition

async def run_panel(source):
    policy = PolicyDefinition.from_text(source)
    environment = gym.make("CartPole-v1")
    try:
        async with Executor(concurrency=2, episode_timeout=30) as executor:
            jobs = await executor.execute(
                Job(policy, environment, seed=seed) for seed in (0, 1)
            )
        return {job.seed: job.result for job in jobs}
    finally:
        environment.close()
```

The caller closes the original environment; workers close their serialized copies.
For partial streaming, wrap `executor.iterate(jobs)` in `contextlib.aclosing` and
exit that iterator before leaving the executor context. Cancellation revokes
queued tasks; it does not stop tasks that are already running immediately.

See [installation](../guide/installation.md) for platform requirements and
[runs](../guide/runs.md) for persistent episode collection.

<!-- api: rsikit.execution.job.Job
{members: [policy, environment, seed, max_steps, instructions, result, task_id, done],
  inherited_members: false}
-->

<!-- api: rsikit.execution.executor.Executor
{members: [execute, iterate], inherited_members: false}
-->

<!-- api: rsikit.execution.executor.execute
{}
-->

