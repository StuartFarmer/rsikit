# Application execution

```sh
export OPENROUTER_API_KEY='your-key'
./scripts/run examples.elitesearch --env CartPole-v1 --output runs/search
```

The host launcher builds `rsikit:local` using Docker's layer cache and runs the
entire module in one foreground container. Generation, model calls, evaluation,
SQLite persistence, logging and Rich all live there. A terminal gets normal Rich
rendering; redirected output uses ordinary stdout/stderr. Ctrl-C and exit codes
pass through Docker. There is no custom host communication protocol.

Only the output directory is mounted. Source, templates and datasets are baked
into the image. Use `runs/...` for application output and resume arguments; the
same files appear under the host output root after the container exits.

| Setting | Default |
| --- | --- |
| `RSIKIT_RUNS_DIR` | Repository `runs/` (host path) |
| `RSIKIT_ENV_FILE` | Optional repository `.env`, in Docker env-file format |
| `RSIKIT_CPUS` | `4` |
| `RSIKIT_MEMORY` | `8g` |

Set host `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` to override
values from the env file. Credentials are passed only at runtime, never baked
into the image. The launcher supplies `RSIKIT_IMAGE_ID` and, if Git is available,
`RSIKIT_GIT_REVISION` for manifests. Source snapshots capture the actual code.

The container uses the invoking UID/GID, a read-only root, a writable 1 GiB `/tmp`,
512 PID limit, dropped capabilities and no privilege escalation. It has outbound
network access for model APIs. Generated code shares application credentials,
network, mounted outputs and scoring state. This protects the rest of the host;
it does not isolate mutually hostile policies or protect scores from tampering.

```python
from rsikit import Executor

async with Executor(concurrency=4, episode_timeout=60) as executor:
    async for policy_id, seed, episode in executor.evaluate(jobs, environment):
        print(policy_id, seed, episode.total_reward)
```

`Executor` runs fresh episode processes in the current application. Calling it
outside Docker runs locally. Linux uses a clean Python forkserver to amortize
imports; macOS development uses spawn. The policy is an ordinary `Solution`
instance sharing its child's environment. No policy wrapper or Docker service
is involved. Each child has its own temporary working directory.

The parent enforces whole-episode deadlines and kills ordinary descendants on
completion, timeout or cancellation. Deliberately detached descendants are only
guaranteed to disappear when the whole container exits. There is no per-action
deadline. Candidate errors and timeouts preserve successful sibling episodes;
runtime failures propagate. A later evaluation starts a fresh process.

Results use a multiprocessing connection, capped at 64 MiB. Saved episode JSON
keeps its existing representation. Episode stdout/stderr becomes an `episode.log`
artifact; failures include at most 4 KiB of its tail. Rich stays in the main process.

The old `rsikit.sandbox` package and `DockerSandbox` are removed. Use
`Executor(episode_timeout=...)` or `run_program(episode_timeout=...)`.
Poker retains its separate Docker service and private player processes; use its
[own launcher in the paper repository](../../elitelist_papers/poker/README.md).

Measure evaluation within the application:

```sh
./scripts/run examples.benchmark_evaluator --samples 5 --output runs/evaluator.json
```

## Application migration measurements — 2026-09-28

Same reviewed policies and seed 1 on local ARM64 Docker Desktop; baseline medians
use 3 samples and application medians use 7. All scores match. Controller location
changed from the macOS host to the Linux application container; these describe
this machine and configuration, not a throughput guarantee.

| Workload | Former warm service | Application, warm forkserver | Startup and preparation | Policy + env.step |
| --- | ---: | ---: | ---: | ---: |
| Packing | 12.5 ms | 38.3 ms | 35.9 ms | 0.02 ms |
| Cartpole | 19.8 ms | 48.9 ms | 37.6 ms | 2.72 ms |
| Blackjack | 77.0 ms | 80.0 ms | 37.7 ms | 9.99 ms |
| Bitcoin | 122.9 ms | 101.6 ms | 35.5 ms | 6.36 ms |

The first Packing episode took 551 ms including clean
forkserver initialization. Startup/preparation measures submission through the
start of environment reset. The remainder includes reset, trajectory copying,
validation, artifacts, transfer and cleanup. Cheap episodes regress because fresh
application children do more preparation; long episodes amortize that work.
No persistent policy workers or custom host service were restored.

A source rebuild took 34.5 s with base/scientific dependency layers
already cached. Median cached build was 2.11 s; container start plus importing
RSIKit was 0.86 s; the complete cached launcher plus fixed Packing episode took
3.52 s. These are separate from warm episode timings. A later Docker metadata
session expired and needed the base image pulled again; cached builds can still
consult the registry. A clean dependency download/build was not measured.
A real controlling-terminal launch returned 130 on Ctrl-C and removed its container.

Validation: 210 standard tests, 12 paper-runner tests, and 30 separate poker tests
passed. Real launcher runs covered scripted generation, saved policies/episodes,
cached-seed reuse across containers, piped output, cancellation and preserved
outputs. Synthetic secret/output probes were absent from the image.

[Raw samples, image IDs and source hashes](application-benchmark-2026-09-28.json)
include the before/after evidence. Five pre-existing Ruff import-order findings
remain outside this migration. A deferred cosmetic poker diagnostic still says
64 MiB for a limit enforced at 8 MiB.

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
