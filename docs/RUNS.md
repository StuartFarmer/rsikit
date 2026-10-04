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
requested seeds. `measure_rewards` returns per-seed `Episode` objects and
candidate diagnostics. AlphaEvolve's `research.alphaevolve.paper.evaluation.assess`
returns raw episodes and applies screening thresholds. AlphaEvolve
constructs its richer `EvaluationResult` and derived descriptors during update.
Infrastructure errors and cancellation propagate; these are not low fitness.

## Core execution

`Evaluator(environment, policy).run(observation, info=info)` consumes caller-owned,
already-reset instances and returns an `Episode`. It never creates, resets, seeds,
or closes them. See [the inner-loop contract](INNER_LOOP.md).

`Executor.evaluate(jobs, environment)` is an async stream of
`(policy_id, seed, episode)`, where each job is `(policy_id, source, seed)`.
The environment is an unstarted, serializable Gymnasium instance used as a template.
Each episode gets fresh environment and policy instances in a local child process,
resets them, uses `Evaluator`, and closes them. The caller owns the template.

Use `async with Executor(concurrency=4, episode_timeout=60)` to bound evaluation
work. Context exit cancels and reaps outstanding episodes. `Run` contexts only
release storage. Candidate failures preserve successful siblings; runtime failures
and cancellation propagate. There is no backend selector or persistent service.

The launcher puts the entire application in one container:

```sh
./scripts/run examples.inner_loop --output runs/comparison
./scripts/run examples.inner_loop --resume runs/comparison
```

Rich renders in the main application process. Episode prints are saved as
`episode.log` artifacts; failures include a bounded log tail. See
[execution details](IN_PROCESS_SANDBOX.md).

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
Both storage and process transfer limit each episode to 64 MiB.

Gymnasium recording wrappers run in the episode process, with output relocated to an
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

## Automatic terminal progress

Entering a `Run` context automatically supplies scoped logging. Standard search loops report their work through Python logging, activating the shared Rich dashboard in a terminal. Redirected output stays plain text. Full messages and exception tracebacks are appended to `run.log`; the terminal shows a bounded recent tail.

The leaderboard’s top border shows `<environment> w/ <optimizer>`, a completed-generation progress bar and total elapsed time. Its bottom border shows the run directory. Searches without a fixed generation budget show an unknown total. Proposals, evaluations and recent events appear below. Progress bars live in the PROPOSALS and EVALUATIONS panel titles, each showing completed/total work for the current generation: 44 proposals and 26 evaluations out of a population of 50 show 44/50 and 26/50 respectively. Seeds and repairs do not create extra candidate slots. Discarded/failed work is visibly distinguished from successful evaluation. Elapsed time measures this invocation; resumed runs are labelled and recover counts from existing optimizer history.

The dashboard expands to the terminal height. Tables use dim grey column labels. The leaderboard reserves the configured elite count, capped at ten rows; proposal and evaluation tables each reserve the generation’s candidate count, capped at 25 rows (25 when the count is unknown). Missing rows remain blank, and small terminals may reserve fewer rows. Their height stays fixed as results arrive or a generation clears. The event log fills the remaining height and keeps the newest wrapped lines visible. Proposals and evaluations are tables of completed work, with each new completion inserted at the top and the oldest rows dropping off the bottom. Proposals show Time, ID, Name and Description; evaluations show Time, ID, Name, Score and Duration. Time is the completion timestamp in UTC (`HH:MM:SS`). Time, ID, Score and Duration have fixed widths. Description expands in the proposal table; Name expands in the evaluation table. Both tables clear when a new generation starts; completed rows remain visible until then. Generating/in-progress updates do not create table rows. In-flight, repair, and failed-attempt details remain in the event log. Each policy receives a random unused color for that Run, shared by its leaderboard, proposal and evaluation rows. The Docker launcher enables 256-color output and forwards the host's true-color setting when available. Very long runs use additional distinct RGB colors; terminals with only 256 colors may display some of those as the same shade.

Event log lines use `HH:MM:SS  TYPE     message`: dim UTC timestamps, green SUCCESS, red ERROR, yellow WARNING, cyan INFO and grey DEBUG labels. Types have a fixed seven-character width; multiline details and tracebacks align beneath the message. Completion events determine success without matching message text. Logger paths are omitted from the TUI, while `run.log` retains full timestamps, logger names and exception details. DEBUG records are styled when enabled; the default logging level remains INFO.

No UI setup is needed in application scripts:

```python
async with Run.create(name="experiment") as run:
    # Construct the environment, executor and optimizer as usual.
    await search(optimizer, evaluate, on_checkpoint=lambda agent: run.save(*agent.records()))
```

The optional `total_generations` field on `search_started` supplies the generation budget; the display counts completed batch events. `leaderboard_size` supplies the row capacity (default ten, display capped at ten); EliteSearch reports its configured elite count.

Optimizers declare optional leaderboard columns once using existing Rich columns:

```python
leaderboard_columns = {
    "operation": Column("Operation", no_wrap=True),
    "parents": Column("Parents", overflow="ellipsis"),
}
```

Their normal domain logs carry structured fields, for example:

```python
logger.info(
    "Starting search",
    extra={
        "progress": {
            "kind": "search_started",
            "optimizer": "MyOptimizer",
            "total_candidates": 50,
            "total_generations": 5,
            "columns": leaderboard_columns,
            "resumed": False,
        }
    },
)
```

Candidate events use stable attempt IDs and revisions; leaderboard events include already-ranked rows with standard fields and custom `extras` values. The dashboard preserves optimizer ranking and interprets model-provided strings literally. See the [event contract](superpowers/specs/2026-09-28-automatic-progress-dashboard.md#structured-logging-contract) for fields and lifecycle events. No registration, adapter, or renderer change is needed for another optimizer.

Known-size batches finish when all their candidate slots settle, including standalone generate/update loops. Native paper searches resumed from only a population checkpoint recover known accepted completions; unknown historical outcomes stay unresolved. Application searches use their detailed Run history where available.

AlphaEvolve and ShinkaEvolve's persistence adapters live in `research.alphaevolve.search` and `research.shinkaevolve.search`. They delegate to `rsikit.search`. Their example entry points remain available. Test/application console injection belongs on `Run.create(..., console=...)` or `Run.open(..., console=...)`, not on the search loop.

New unified search manifests record `optimization_schedule: round-v1`. Optimizer
checkpoints retain complete-round boundaries, pending feedback, and queued repairs
where recovery is supported. `Run.open` alone does not restore an optimizer.
Unified Elite and the paper AlphaEvolve example support recovery; historical
baseline AlphaEvolve, ShinkaEvolve, and LineageSearch do not gain resume support.
Legacy provenance remains unchanged; schedule transitions are separate appended
evidence. See [CLI recovery](CLI.md#resume-an-interrupted-search).
