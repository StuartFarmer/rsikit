# Local runs

`Policy` is the agent: it implements `reset`, `act`, and `close`.
`generate` returns a named `Policy` subclass. Pass it directly to a run;
the runner supplies the environment spaces when it instantiates the policy.
Generated code runs in Docker, never on the host.

```python
policy = await generate(task, provider=provider)
with Run.create(name="cartpole", environment="CartPole-v1") as run:
    scores = await run.evaluate(policy, seeds=[0, 1, 2])
    print(policy.name, scores)  # {0: score, 1: score, 2: score}
    print(run.path)
```

Configure Slick's template root once as shown in the README. The LLM supplies the
policy's name and implementation. Pydantic validates that response; it does not
wrap the acting policy in a separate data object. The generated class and its
instances share the same Policy interface. Internally the host class delegates
its lifecycle to the isolated worker.

## Storage

Each run has one `run.sqlite` database with two SQLModel tables:

- `settings`: the run name and evaluation settings.
- `policy`: ID, model-generated name, implementation, and a seed-to-score mapping.

There are no episode records or additional metrics. A score is the sum of Gymnasium
rewards. `None` means the requested evaluation has not completed. `run.scores(policy)`
returns the stored mapping; `run.policies()` reloads executable Policy classes.
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
with Run.open("runs/YOUR_RUN") as run:
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
`evaluate` are serialized. Close the run before moving or copying its directory.
Use the same environment code and dependencies when reopening it.

Resume only evaluates already-stored policies. It does not generate more policies
or restore optimizer state. Databases from the earlier episode-record design are
not supported; create a new run.

## Video

`Run.create(record_video=True)` uses Gymnasium's `RecordVideo`. Install `.[video]`
for CartPole rendering. Videos live in the directory shown above and are not
tracked as database records. Retrying an unfinished seed replaces its video.
Rebuild the Docker image when upgrading to the restored `Policy` interface.
