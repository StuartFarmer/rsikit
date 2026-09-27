# InProcessDockerSandbox

```python
from rsikit import Executor, InProcessDockerSandbox

executor = Executor(
    sandbox=InProcessDockerSandbox(episode_timeout=60),
    concurrency=4,
)
```

Pass this Executor to `Run.create` or `Run.open`. Generated source is loaded only
inside Docker. Each episode gets a fresh process with both its environment and
policy; the complete rollout runs there and returns a score and artifacts. There
is no per-action serialization or socket exchange. This is the default backend
for `Executor()`. Pass `sandbox=DockerSandbox()` to use separate policy/evaluator
processes instead.

The container restrictions are shared with `DockerSandbox`: no host mounts, no
network, a read-only root filesystem, an unprivileged user, dropped capabilities,
no privilege escalation, and CPU/memory/PID limits. An external supervisor
enforces `episode_timeout`, including source loading, constructors, resets,
actions and cleanup. It terminates the episode's process group, including ordinary
subprocesses such as video encoders; descendants cannot start new process groups
or sessions to escape cleanup. `Executor.call_timeout` is intentionally not
enforced by this backend.

The policy can inspect or modify its environment and scoring state. This backend
is for runs that accept that tradeoff while retaining the Docker host boundary.
Candidate errors, abrupt exits and episode timeouts are reported as policy failures;
successful sibling results remain usable. Container cancellation still removes
the whole sandbox. Standard action validation, seeding, observation copies, score
persistence and artifact path validation use the existing runner.

Rebuild the image with the normal Dockerfile. Older images are rejected explicitly
by the new class rather than silently falling back to separate processes:

```sh
rtk proxy docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
rtk proxy .venv/bin/python -m examples.benchmark_evaluator \
  --in-process --image rsikit-sandbox:local --compare-image rsikit-sandbox:local \
  --samples 10 --output /tmp/in-process-benchmark.json
```

This compares `InProcessDockerSandbox` against `DockerSandbox` using the same
image, policies, environment and seeds, alternating order after warmups. Scores
must match. Instrumented environment/policy timings are recorded separately from
the uninstrumented comparison. No model calls or Cython/Numba compilation occur.

## Measured results

Local ARM64 measurements on 2026-09-24, median of 10 alternating samples per
backend using the same image. These compare against the already optimized
separate-process backend. All scores matched.

| Environment | Steps | DockerSandbox | InProcessDockerSandbox | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Packing | 1 | 12.6 ms | 12.3 ms | 1.03× |
| CartPole | 500 | 57.7 ms | 14.5 ms | 3.97× |
| Blackjack | 2,485 | 231.8 ms | 26.7 ms | 8.67× |
| Bitcoin | 2,556 | 239.9 ms | 30.0 ms | 7.98× |

These are complete rollout timings in a warm container, excluding image build
and container startup. Cheap policies with many steps benefit most; one-step
packing is effectively unchanged. Raw samples, image IDs, source hashes and
instrumented timings are in [the benchmark report](in-process-benchmark-2026-09-24.json).

Validation: all 199 tests passed against the updated image, plus an explicit
same-process video recording check and rejection of an older image. Tests cover
fresh episode state, policy crashes, timeout recovery, subprocess cleanup and
blocked attempts to escape the episode process group.
