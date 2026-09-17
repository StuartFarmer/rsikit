# Local runs

The environment owns task configuration. The executor owns worker concurrency,
timeouts, and its sandbox. Run stores policies, scores, and returned artifacts.

```python
import gymnasium as gym
from rsikit import DockerSandbox, Executor, Run

executor = Executor(sandbox=DockerSandbox(), concurrency=4, call_timeout=10)
with gym.make("CartPole-v1", max_episode_steps=500) as environment:
    with Run.create(name="comparison", environment=environment, executor=executor) as run:
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

`Executor(concurrency=4, call_timeout=10, sandbox=DockerSandbox())` starts **one Docker
container for a batch**, then launches up to four evaluation processes inside it.
Each process owns its environment and launches the generated agent in a separate
process. Observation/action communication stays inside the container. Completed
scores and artifacts stream back to Run. The container is removed when the batch
finishes or is cancelled. Fully cached evaluations start no container.

`call_timeout` bounds each agent lifecycle call. Native environment code remains
responsible for its own step behavior. Cancellation waits for worker and sandbox
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

## Storage and resume

Each run directory contains `run.sqlite`, exported policies under `exports/`, and
returned files under `artifacts/`. SQLite has two tables:

- `settings`: name and Python-export preference.
- `policy`: ID, generated name, implementation, and seed-to-score mapping.

`Run.create(export=True)` remains the default. Missing Python exports are recreated
when opening a run. `run.policies()` reloads definitions; `run.scores(policy)` returns
stored scores. `None` means unfinished. Identical policies and seeds reuse scores.
There are no episode metrics, optimizer checkpoints, or stored environment objects.

```python
with Run.open("runs/YOUR_RUN", environment=environment, executor=executor) as run:
    await run.resume()
```

Supply the same configured environment when reopening; Run does not reconstruct it.
The entire group's requested seeds are persisted before dispatch. Each result and
its artifacts are saved as they arrive. Resume retries unfinished evaluations,
including failed ones. It does not regenerate policies or restore mid-episode state.
One process may own a run directory at a time (macOS/Linux file lock). Close it before
moving or copying the directory.
