# Runs and execution

A `Run` serializes one optimizer run: policies, measurements, raw episodes, and
optimizer-defined checkpoints. It has no environment, executor, evaluation loop,
or execution-resume method. Opening a Run never starts Docker.

```python
import gymnasium as gym
from rsikit import Executor, Run
from research.rollouts import Rollouts
from research.rewards import mean_rewards

# Inside an async function; policies are Policy definitions from generate/from_file.
with gym.make("CartPole-v1", max_episode_steps=500) as environment:
    async with Executor(concurrency=4) as executor, Run.create(name="comparison") as run:
        rollouts = Rollouts(environment, executor, run)
        scores = await mean_rewards(rollouts, policies, seeds=(0, 1, 2))
        print(scores)  # {policy_id: mean episode reward}
```

`Rollouts` and reward callbacks live in `research/`, alongside the experiments.
They connect execution to persistence and reuse saved episodes within a fixed
experiment environment/configuration. Each successful episode is saved before
fitness is computed; failed siblings do not erase it. Changing environment settings
requires a new run. Old scalar scores remain readable, but are not fabricated into
trajectories: missing episodes are executed again when requested.

`mean_rewards` explicitly selects cumulative reward as fitness and averages the
requested seeds. `measure_rewards` returns per-seed `Measurement` objects and
candidate diagnostics. AlphaEvolve's `research.alphaevolve.paper.evaluation.assess`
owns its richer `EvaluationResult`, descriptors, and screening thresholds.
Infrastructure errors and cancellation propagate; these are not low fitness.

## Core execution

`Evaluator(environment, policy).run(observation, info=info)` consumes caller-owned,
already-reset instances and returns an `Episode`. It never creates, resets, seeds,
or closes them. See [the inner-loop contract](INNER_LOOP.md).

`Executor.evaluate(jobs, environment)` is an async stream of
`(policy_id, seed, episode)`, where each job is `(policy_id, source, seed)`.
The environment is an unstarted, serializable Gymnasium instance used as a template.
Each sandbox episode gets fresh environment and policy instances, resets them,
uses `Evaluator`, and closes them. Host and sandbox Python minor versions must match.
The caller owns the host template's lifecycle.

Use `async with Executor(...)` to reuse a container across batches; its exit removes
the container. `Run` contexts only release storage. `InProcessDockerSandbox` is the
default: policy and environment share one fresh episode process inside Docker.
`DockerSandbox` puts them in separate processes to isolate environment state and
supports per-policy-call deadlines. Both return complete Episodes through a bounded,
data-only protocol. Episode deadlines default to 60 seconds.

Build the image before execution:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
```

Candidate errors preserve successful siblings and the service. Infrastructure
failures or cancellation invalidate and remove the service; an explicit retry
can start a new one. See [sandbox details](IN_PROCESS_SANDBOX.md).

## Storage and analysis

```python
from rsikit import Policy, Run

policy = Policy.from_file("solution.py")
with Run.create(name="experiment") as run:
    run.save_policy(policy)
    run.save_episode(policy, 42, episode)
    # Fitness is a caller decision, and need not equal total reward.
    run.save_policy(policy, scores={42: fitness})

with Run.open(run.path) as restored:
    episode = restored.load_episode(policy, 42)
    print(episode.observations, episode.actions, episode.rewards)
    print(restored.scores(policy))
```

`Policy.from_text`/`from_file` load source without validating or executing it.
Optimizers explicitly call `rsikit.policy.validate_policy(policy)` after
generation; this checks syntax and the construction interface without execution.
`to_text` and `to_file` preserve its name, description, source and ID, including
invalid proposals retained for diagnosis. Run exports definitions to
`exports/` by default; `export=False` disables that. Missing exports are recreated
on open. `policies()` reloads saved definitions.

`run.sqlite` contains core `settings` and `policy` tables plus optimizer-defined
records. Per-seed scores are explicit caller data; `None` denotes unfinished work.
Raw trajectories are stored under `episodes/<policy-id>/<seed>.json`, preserving
numeric arrays, tuples and byte payloads. Episode files are written atomically
after their artifacts. `load_episode` returns `None` when no episode was saved.
Both storage and sandbox transport limit each episode to 64 MiB.

Gymnasium recording wrappers run in the sandbox, with output relocated to an
isolated episode directory. Files and the final `info["artifacts"]` byte mapping
are returned as `episode.artifacts`. `save_episode` also writes these files under
`artifacts/<policy-id>/<seed>/`. Run performs no rendering.

## Resume

The optimizer/controller restores its configuration and checkpoint, opens storage
with `Run.open(path)`, constructs its environment and executor, and continues its
own search loop. Run does not reconstruct or advance an optimizer. Existing
AlphaEvolve paper and EliteSearch checkpoint paths remain optimizer-owned.
The inner-loop example restores its saved seed panel and step limit and requests
its saved policies again; Rollouts reuses completed episodes.

Only one process may own a run directory at a time (macOS/Linux file lock).
Close it before moving or copying the directory. Older databases retain policies
and scores and receive the existing description-column migration on open.

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
