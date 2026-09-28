# Docker sandbox

```python
from rsikit import DockerSandbox, Executor

async with Executor(
    sandbox=DockerSandbox(episode_timeout=60),
    concurrency=4,
) as executor:
    async for policy_id, seed, episode in executor.evaluate(jobs, environment):
        print(policy_id, seed, episode.total_reward)
```

`Executor()` uses `DockerSandbox` by default. Keep its async context open across
batches to reuse one container and a clean Python forkserver with common imports
preloaded. Every evaluation gets a fresh process and working directory. The
policy, environment and scoring run together; complete Episodes and artifacts
return through bounded JSON. No per-action IPC, policy-worker pool, or state-reset
protocol is needed. `run_program` uses this same path for one standalone episode.

The host owns LLM calls, credentials, storage and Rich/TUI rendering. Requests and
results use a newline-delimited JSON pipe. The worker redirects Python prints and
native stdout to stderr, separate from the result pipe. The Docker client forwards
stderr in bounded chunks to the `rsikit.sandbox.docker` logger at INFO, with
`event="sandbox_log"`. Terminal control characters are replaced. After 1 MiB per
container, additional diagnostic output is drained and discarded. Concurrent
worker logs can interleave; they are diagnostic text, not structured progress.
Existing host progress events update as evaluations finish. No new UI event bus
or per-step traffic is required.

The container has no host mounts or forwarded API keys, no network, a read-only
root filesystem, an unprivileged user, dropped capabilities, no privilege
escalation, and CPU/memory/PID limits. An external supervisor enforces
`episode_timeout`, including source loading, resets, actions, policy close and
result encoding. It reaps the episode process group; descendants cannot create
new groups or sessions to escape cleanup. Container startup, queueing and final
process reaping are outside that deadline. Per-action deadlines are not supported.

The policy can inspect or alter its environment and scoring state. Docker protects
the host, not score integrity or mutually hostile jobs inside the same container.
Policy errors, abrupt episode exits and timeouts preserve the service and successful
siblings. Infrastructure failures or cancellation invalidate and remove the
container; a later explicit retry can start a new one.

## Migration

Use `DockerSandbox` in place of `InProcessDockerSandbox`. `SandboxPolicy`, the
separate evaluator/policy backend, and the native sandbox-runtime prototype have
been removed. Replace `Executor(call_timeout=...)` and
`run_program(call_timeout=...)` with whole-episode limits:
`DockerSandbox(episode_timeout=...)` or `run_program(episode_timeout=...)`.
The paper runner uses `--episode-timeout`; `--policy-timeout` is removed.
Poker retains its own private seat transport under its example package because
players must not read each other's hidden cards.

Rebuild the worker; incompatible protocol versions are rejected at startup:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
.venv/bin/python -m examples.benchmark_evaluator --samples 5 --output /tmp/evaluator.json
```

## Historical measurements

The following measurements predate consolidation. The old backend is no longer
available; these numbers describe the reason for choosing whole-episode execution.

Local ARM64 measurements on 2026-09-24, median of 10 alternating samples per
backend using the same image. These compare against the already optimized
separate-process backend. All scores matched.

| Environment | Steps | Former separate-process backend | Whole-episode backend | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Packing | 1 | 12.6 ms | 12.3 ms | 1.03× |
| CartPole | 500 | 57.7 ms | 14.5 ms | 3.97× |
| Blackjack | 2,485 | 231.8 ms | 26.7 ms | 8.67× |
| Bitcoin | 2,556 | 239.9 ms | 30.0 ms | 7.98× |

These are complete rollout timings in a warm container, excluding image build
and container startup. Cheap policies with many steps benefit most; one-step
packing is effectively unchanged. Raw samples, image IDs, source hashes and
instrumented timings are in [the benchmark report](in-process-benchmark-2026-09-24.json).
