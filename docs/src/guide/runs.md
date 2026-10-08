# Runs and recovery

Use a `Run` to retain policy source, per-seed evidence, and scores. `Run.create`
requires a new directory; omit `path` to create a timestamped directory under
`runs/`. `Run.open` reopens an existing `run.sqlite`. Both acquire an exclusive
lock, so only one open owner may use a run directory at a time. Use a synchronous
or asynchronous context manager to release it.

```python
from rsikit import Run

with Run.open("runs/my-search") as run:
    for policy in run.policies():
        print(policy.name, policy.id, run.scores(policy))
        episode = run.load_episode(policy, 0)
        if episode is not None:
            print(episode.error, episode.total_reward)
```

Only load trusted runs: episode `.pkl` files use pickle and can execute Python
during loading. Legacy tagged JSON episodes remain readable.

## What is saved

| Path | Contents |
| --- | --- |
| `run.sqlite` | Run settings, definitions, per-seed scores, and application-specific SQLModel records. |
| `exports/<id>_<name>.py` | Policy source with identity metadata, when exports are enabled. |
| `episodes/<id>/<seed>.pkl` | Validated episode data, including errors and artifacts. |
| `artifacts/<id>/<seed>/` | Files produced during the episode, such as logs or video. |

`save_policy` stores source and optional measurements without execution.
`save_episode` stores raw evidence; `Run.evaluate` separately writes successful
fitness scores. `scores(policy)` returns seed-to-score entries, with `None` for
requested evaluations that have no saved successful measurement. Use the intended
seed panel when comparing policies, rather than averaging every historical seed.
`Run.save` and `Run.database` allow research implementations to persist their own
typed records.

The unified CLI additionally writes `config.yaml`, `manifest.json`,
`measurements.jsonl`, and `summary.json`. Searches also record model calls in
`model_calls.jsonl`, selection in `selection.json`, and a successful winner in
`winner.py`. Algorithm-specific records and outputs vary. A summary distinguishes
completion, budget exhaustion, evaluation failure, and no valid candidate; do not
infer success just from a directory or exported source existing.

## Episode reuse

`Run.collect` streams saved successes first and schedules missing or failed
policy/seed pairs. `Run.evaluate` collects this stream into complete feedback and
persists successful fitness. Repeating a request can therefore reuse work after
an interruption, and failed episodes are retried.

Reuse keys are **policy identity and seed only**. A run assumes a fixed environment
and configuration. Changing the task implementation, data, maximum episode length,
or other evaluation settings requires a new run, even if the policy ID is unchanged.
The run does not infer or verify that equivalence for you.

## What resume means

| State | Supported recovery |
| --- | --- |
| Stored policies and episodes | `Run.open` and evaluate the same panel/configuration to reuse successes. |
| Persistent executor queue | `Executor(database=...)` retains queue tasks/results; it does not restore optimizer state or reconstruct caller `Job` handles. |
| Unified CLI EliteSearch | `rsikit resume DIRECTORY` restores saved Elite records, configuration, and model-call budget ledger. |
| Python EliteSearch | `EliteSearch.restore(organisms, generations)` restores its typed records; the caller supplies matching configuration and evaluation resources. |
| Paper AlphaEvolve | Reopening the same `database_path` restores its archive/checkpoint. Supply compatible configuration and the same task/context, checkpoint round boundaries, and close the agent. |
| Checkout `examples.alphaevolve` paper runs | `python -m examples.alphaevolve --resume DIRECTORY` restores its own saved experiment and population files; `--generations` supplies additional batches. |
| Original/improved AlphaEvolve, ShinkaEvolve, LineageSearch | Saved evidence/history is available; no corresponding unified CLI optimizer-resume command is implemented. |
| Sequential experimental algorithms | Their suspended proposal generators are not checkpoint restorations. Do not assume they resume after interruption. |

```sh
rsikit resume runs/my-elite-search
```

The separate checkout AlphaEvolve command resumes runs created by that example,
not arbitrary unified CLI runs. It requires OpenRouter credentials and preserves
the saved task, model, seed panel, and evaluation settings. Its additional-batch
budget differs from the unified CLI's cumulative model-call ledger.

CLI resume currently accepts only searches started with `--optimizer elite`.
It reuses the saved budget; it does not reset previously consumed calls or expand
an exhausted limit. Incomplete legacy streaming checkpoints are rejected where
the implementation cannot safely convert them to proposal rounds. Reopening a
queue, reopening a run, and restoring an optimizer are separate operations.

See [storage API](../api/runs.md) and [CLI reference](../api/cli.md).
