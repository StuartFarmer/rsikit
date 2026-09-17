# Local runs

`Policy` is the agent interface: `reset`, `act`, and `close`. `generate` returns a
named Policy definition without executing its implementation or choosing a backend.
`Run` chooses the executor and schedules evaluations.

```python
from rsikit import DockerExecutor, Run, generate

policies = [await generate(task, provider=provider) for _ in range(5)]
with Run.create(
    name="cartpole",
    environment="CartPole-v1",
    executor=DockerExecutor(),
    concurrency=4,
) as run:
    scores = await run.evaluate(*policies, seeds=[0, 1, 2])
    for policy in policies:
        print(policy.name, scores[policy.id])  # {0: score, 1: score, 2: score}
```

Configure Slick's template root once as shown in the README. The LLM supplies the
policy's name and implementation, validated by Pydantic. Generated definitions
retain the Policy class contract but remain unloaded and cannot be instantiated
on the host. The executor loads the actual generated `Solution(Policy)` class.
There is no sandbox subclass attached during generation.

## Execution

`Run.executor` is explicit. It defaults to `DockerExecutor()` when omitted.
`Run` registers the entire group before dispatching work, skips existing scores,
limits active evaluations with positive `concurrency` (default 1), and commits each
score as it arrives. `evaluate` always returns `{policy_id: {seed: score}}`, including
when only one policy is supplied. Repeated policies or seeds are evaluated once.

`DockerExecutor.evaluate` creates the Gymnasium environment and explicitly
constructs `SandboxPolicy` for the generated agent. Each evaluation gets a fresh
Docker container. The environment and scoring run on the host; observations and
actions cross the sandbox boundary. The executor closes both sides before returning
or raising. `DockerExecutor(image=...)` selects the worker image.

The `Executor` protocol has one async method, `evaluate`, returning one finite score.
Its inputs are the implementation text and ordinary serializable evaluation
arguments: environment ID, kwargs, seed, step limit, instructions, timeout, and
optional video directory. It receives no Run, ORM session, or dynamic Python class.
A process or cloud executor can implement this method without changing generation
or Run storage. Only the Docker implementation is included. A remote executor is
responsible for delivering requested videos to the supplied output directory.

All evaluations in a group are attempted. If any fail, successful scores remain
saved and `evaluate` raises the first error in submission order after the group
finishes. Cancellation cancels outstanding work and waits for executor cleanup.
Custom executors must honor cancellation and clean up their workers before raising.

## Storage

Each run has one `run.sqlite` database with two SQLModel tables:

- `settings`: the run name and evaluation settings.
- `policy`: ID, model-generated name, implementation, and a seed-to-score mapping.

There are no episode records or additional metrics. A score is the sum of Gymnasium
rewards. `None` means the requested evaluation has not completed. `run.scores(policy)`
returns the stored mapping; `run.policies()` reloads Policy definitions.
Policy IDs are hashes of the generated name and implementation, so identical
policies reuse their existing scores.

Python exports are enabled by default (`Run.create(export=True)`). Set `export=False`
to keep policies only in SQLite. Exports are ordinary `Solution(Policy)` modules.
Missing exports are recreated when opening a run; editing one does not change the
database. Generated names are sanitized for filenames.

```text
runs/cartpole-<timestamp>-<id>/
    run.sqlite
    .lock
    exports/<policy-id>_<name>.py
    videos/<policy-id>/<seed>/rl-video-episode-0.mp4
```

Pass `path=...` to choose a new directory. It must not already exist. Run settings
include a registered Gymnasium environment ID, JSON `environment_kwargs`, optional
`max_steps` and `instructions`, and the sandbox `call_timeout`. Register custom
environments in your application before using them.

## Resume

```python
with Run.open("runs/YOUR_RUN", executor=DockerExecutor(), concurrency=4) as run:
    await run.resume()
    for policy in run.policies():
        print(policy.name, run.scores(policy))
```

Requested seeds are saved before execution. Each score is committed immediately.
Completed scores are reused; missing scores are retried from the beginning with
the same seed for both environment and policy. Errors propagate without becoming
scores. This includes policy errors: resume will encounter the same error again
unless its cause has been resolved. There is no failure history or automatic repair.

One process owns a run directory at a time via an OS lock (macOS/Linux). Calls to
`evaluate` are serialized; evaluations within a group can run concurrently.
Close the run before moving or copying its directory.
Use the same environment code and dependencies when reopening it. Executors and
concurrency are runtime choices, not serialized state; supply them again to `open`.
The default executor on reopening is Docker.

Resume only evaluates already-stored policies. It does not generate more policies
or restore optimizer state. Databases from the earlier episode-record design are
not supported; create a new run.

## Video

`Run.create(record_video=True)` asks the executor to record video. The Docker
executor uses Gymnasium's `RecordVideo`. Install `.[video]`
for CartPole rendering. Videos live in the directory shown above and are not
tracked as database records. Retrying an unfinished seed replaces its video.
Rebuild the Docker image when upgrading to the restored `Policy` interface.
