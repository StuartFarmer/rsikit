# Local runs

The environment owns task configuration. The executor owns worker concurrency,
timeouts, and its sandbox. Run stores policies, scores, and returned artifacts.

```python
import gymnasium as gym
from rsikit import Executor, Run

executor = Executor(concurrency=4)
with gym.make("CartPole-v1", max_episode_steps=500) as environment:
    async with Run.create(name="comparison", environment=environment, executor=executor) as run:
        scores = await run.evaluate(policies)
        print(scores)  # {policy_id: score}
```

Policies come directly from `await generate(task, provider=provider)`. Their names
come from the model. Generation neither executes their implementations nor selects
a sandbox. Generation writes nothing to disk; `evaluate` is the first place
a policy is saved. `Policy` remains the agent's reset/act/close interface.

By default, each policy gets one episode with seed 0. A seed controls the random
starting conditions; giving every policy the same seeds makes comparisons fairer.
Pass `seeds=[0, 1, 2]` to evaluate multiple starts; the returned score is their mean.
The same seed also initializes policy randomness. `run.scores(policy)` exposes the
individual episode scores. Keep the seed set fixed while comparing generations.

## Environment

Pass a configured Gymnasium **instance**. Use `gym.make` arguments, `TimeLimit`,
custom environment attributes such as `instructions`, and native wrappers. Run has
no environment kwargs, step limit, instructions, render, or recording options.

The supplied environment is a template for independent evaluations. It is serialized
with cloudpickle and each worker loads its own instance inside the sandbox. The host
instance is not stepped or closed by Run; its caller owns it. Use an unstarted environment
that supports serialization. Required environment modules and dependencies must be
installed in the sandbox image; local classes that cloudpickle serializes by value
are also supported. Host and sandbox Python minor versions must match.

## Executor and sandbox

With `async with Run.create(...)` or `async with Run.open(...)`,
`Executor(concurrency=4)` defaults to `InProcessDockerSandbox` and lazily starts
**one Docker container for the run**, then launches up to four evaluation processes inside it.
Each process runs both its environment and generated agent, with no per-action
process communication. Completed
scores and artifacts stream back to Run. The same container serves later batches,
generations, policy repairs and held-out evaluations. It is removed when the async
Run context exits, including after model failures or interruption. Fully cached
runs start no container. All example CLIs use this persistent lifecycle.

Infrastructure failures or cancelled evaluations invalidate and remove the container;
an explicit retry starts a fresh one. Candidate errors and policy timeouts leave the
container available for repaired policies. Each episode uses a fresh process.

Direct executor users can use `async with Executor(...)`. `await run.aclose()` or
`await executor.aclose()` explicitly closes asynchronous resources. The older
synchronous `with Run(...)` API remains compatible, with per-batch sandbox cleanup;
use the async context for persistent evaluation. Synchronous `run.close()` releases
storage only and is not a replacement for async cleanup.

The default sandbox enforces a 60-second `episode_timeout` for the complete rollout;
`call_timeout` does not apply. To isolate the policy from environment state and enforce
per-call timeouts, pass `sandbox=DockerSandbox()` explicitly. See
[sandbox details](IN_PROCESS_SANDBOX.md). Cancellation waits for worker and sandbox
cleanup. Successful evaluations are persisted even when another evaluation fails;
the executor raises an error after delivering the successful results.

Sandbox implementations provide `start`, `evaluate`, and `close`. This keeps
Docker-specific process commands out of Run and Executor. Only Docker is implemented;
a future sandbox can use the same boundary. Shared-container workers share the
container's security boundary, rather than having one container boundary per policy.

Build the worker image using your Python minor version (the Dockerfile defaults to 3.14):

```sh
docker build --build-arg PYTHON_VERSION=3.14 -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
```

## Recording and artifacts

Configure Gymnasium normally:

```python
environment = gym.wrappers.RecordVideo(
    gym.make("CartPole-v1", max_episode_steps=500, render_mode="rgb_array"),
    video_folder="recordings",
    episode_trigger=lambda _: True,
)
```

The worker preserves recording triggers, frame rates, and lengths. It relocates
output paths into that evaluation's temporary directory so workers cannot overwrite
each other's recordings. Gymnasium creates and closes the recordings. The worker
returns those files and any custom `info["artifacts"]` mapping of relative names to
bytes. Run saves them under `artifacts/<policy-id>/<seed>/`. It performs no rendering.
Returned results contain only a score and artifact bytes; transport is limited to
64 MiB per evaluation. Larger artifacts need a streaming transport later.

For the AlphaEvolve example, record the best saved policies after closing the search:

```sh
.venv/bin/python -B -m examples.replay runs/YOUR_RUN --env LunarLander-v3 --top 3 --seeds 0 1 2
```

Specify the original environment and any original `--max-steps` override. Replay
selects policies by mean stored score, excluding incomplete evaluations. It uses
a fresh Run so cached scores do not suppress recording or change the original
results. The command prints MP4 paths under the new Run's `artifacts/` directory.
`--output` chooses a new directory. No model calls are made.

## Storage and resume

Each run directory contains `run.sqlite`, exported policies under `exports/`, and
returned files under `artifacts/`. SQLite starts with two core tables:

- `settings`: name and Python-export preference.
- `policy`: ID, generated name and description, implementation, and seed-to-score mapping.

`Run.create(export=True)` remains the default. Missing Python exports are recreated
when opening a run. `run.policies()` reloads definitions; `run.scores(policy)` returns
stored scores. `None` means unfinished. Identical policies and seeds reuse scores.
Older databases receive an empty description column when opened; their policies
and scores remain intact. There are no episode metrics, optimizer checkpoints, or
stored environment objects.

```python
async with Run.open("runs/YOUR_RUN", environment=environment, executor=executor) as run:
    await run.resume()
```

Supply the same configured environment when reopening; Run does not reconstruct it.
The entire group's requested seeds are persisted before dispatch. Each result and
its artifacts are saved as they arrive. Resume retries unfinished evaluations,
including failed ones. It does not regenerate policies or restore mid-episode state.
One process may own a run directory at a time (macOS/Linux file lock). Close it before
moving or copying the directory.


## Optimizer-defined records

An optimizer defines ordinary `SQLModel` table classes. Pass instances directly to
`run.save(record)` or save a group with `run.save(*records)`. Run infers the tables
from the records, creates missing tables, and commits the group in one transaction.
Saving an existing primary key updates that record. Generated primary keys are
copied back to the supplied objects, so saving the same object again updates it.
If a write fails, the group's record changes are rolled back.

```python
from sqlmodel import Field, SQLModel, select


class SearchEvaluation(SQLModel, table=True):
    __tablename__ = "search_evaluation"
    attempt: int = Field(primary_key=True)
    policy_id: str
    score: float
    strategy: str


evaluation = SearchEvaluation(attempt=1, policy_id=policy.id, score=score, strategy="mutate")
run.save(evaluation)

# Native SQLModel queries remain available for analysis.
with run.database() as db:
    evaluations = db.exec(select(SearchEvaluation).order_by(SearchEvaluation.attempt)).all()
```

`run.database()` opens a native SQLModel session for querying existing tables. It
takes no model arguments and does not create tables. Close query sessions before
closing the Run.

Schemas belong to the optimizer, including table names, fields, and any future
schema migrations. RSIKit does not interpret them. Keep optimizer table names
separate from the core `settings` and `policy` tables. This is native SQLModel
storage, not an optimizer checkpoint or an additional execution abstraction.

The AlphaEvolve CLI saves its own evaluation and generation records automatically;
see [experiment history](ALPHAEVOLVE.md#stored-experiment-history).
