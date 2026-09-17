# Local runs

A `Policy` is returned by `await generate(task, provider=...)`. Its name and summary
come from the model. Pass it directly to `await run.evaluate(policy, seeds=[...])`.
Its generated implementation is kept internal, never imported on the host.
The executed class implements `Controller`; `Policy` is the value used by callers.

## Create and evaluate

```python
with Run.create(
    name="cartpole-comparison",
    environment="CartPole-v1",
    max_steps=500,
    export=True,  # Default.
) as run:
    policy = await generate(task, provider=provider)
    results = await run.evaluate(policy, seeds=[0, 1, 2])
    print(run.path, policy.name)
```

Configure Slick's template root once at application startup as shown in the
README. `generate` owns its external template and uses a Pydantic response schema;
it returns a `Policy`, not the model's wire-format response. The returned object's
public fields are `id`, `name`, and `summary`. Keep the object returned by generation
or load it with `run.policies()`; its public JSON metadata is not a policy backup.

Each run owns one directory and three SQLModel tables: `run`, `policy`, and
`execution`. SQLite stores policy definitions and episode results. Names are
labels; generated UUIDs identify runs, policies, and episodes.

```text
runs/cartpole-comparison-<UTC timestamp>-<id>/
    run.sqlite
    .lock
    exports/<policy-id>_<safe-policy-name>.py
    artifacts/<execution-id>/<attempt>/rl-video-episode-0.mp4
```

Pass `path=...` to choose a new directory explicitly. Existing directories are
never silently overwritten. Exported filenames sanitize the model-generated name.
`export=False` retains the complete policy in SQLite without writing Python files.
Opening a run recreates missing exports when enabled; existing exports are left
alone. Editing an export does not change the stored policy.

Evaluation settings belong to the run: registered Gymnasium environment ID,
`environment_kwargs`, `max_steps`, `instructions`, and `call_timeout`. Register
custom Gymnasium environments in your application before creating or reopening
runs that use them. Only JSON environment arguments are persisted, not factories
or live environment instances. The run does not freeze your installed environment
code or dependencies; use the same software to reproduce an evaluation.

`evaluate` persists the policy and all requested seeds before starting any job.
It uses each seed for both environment and policy randomness. Repeating the same
policy ID and seed returns the stored terminal result; changing a policy under
an existing ID is rejected. Every new episode creates a fresh isolated worker.

Results are `Execution` SQLModel objects, including status, reward, length,
termination/truncation flags, error, timestamps, attempt count, and relative artifact
paths. `run.executions()` lists all results; `run.executions(policy)` filters them.
No transition history is stored.

## Resume

```python
with Run.open("runs/YOUR_RUN") as run:
    results = await run.resume()
    policies = run.policies()
```

Completed and failed episodes remain terminal. Pending jobs and interrupted
in-flight jobs restart from their recorded seeds. Each completed episode is
committed immediately. If the process dies after execution but before committing,
that episode is evaluated again. Mid-episode state is not restored.

Policy failures produce failed records with no reward. Infrastructure/environment
errors and cancellation leave the current job pending, preserve completed results,
and propagate to the caller. Retrying failed policies is deliberately not automatic.

Only one run owner may open a directory at a time; an OS file lock is released
when the context exits or the process dies. This local implementation targets
macOS/Linux. Calls to evaluate/resume on one Run are serialized; use separate run
directories for independent work. Close the run before copying/moving its directory.

Resume only finishes registered episode jobs. It neither generates missing
policies nor resumes AlphaEvolve, prompts, samplers, or other optimizer state.
There is no optimizer serialization/checkpoint contract.

## Videos and exports

`Run.create(record_video=True)` uses Gymnasium's `RecordVideo` on the host and
stores MP4 paths in each execution record. Install `.[video]` for CartPole rendering.
The controller remains in Docker. Videos for separate attempts have separate paths.
Python exports are ordinary `Solution(Controller)` modules and can be imported
in projects that have RSIKit installed, or passed to `run_program` for isolation.

The executable base was renamed from `Policy` to `Controller`. Existing saved
programs using the old base need their import and base class changed to Controller.
Rebuild the worker image after upgrading. Current examples and optimizer prompts
already use this interface. `run_episode` and `run_program` remain available as
stateless lower-level operations.
