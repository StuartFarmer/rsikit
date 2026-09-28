# Containerized RSIKit application

## Agreed outcome

Docker runs the complete standard research application: LLM calls, generation,
evaluation, persistence, logging, and Rich rendering. The host builds and launches
the image and retains output files. It does not orchestrate individual evaluations.

The first migration covers the standard research runners. The user explicitly
chose to preserve poker as a separate application with its existing player
isolation. Moving shared helpers that poker still imports is part of this change;
redesigning poker is not.

This replaces the current, uncommitted whole-episode sandbox consolidation. Start
from the working tree; do not reset that work or resurrect removed backends.

## User interface

```sh
./scripts/run examples.elitesearch --env CartPole-v1 --output runs/cartpole
./scripts/run examples.elitelist_papers.run --env CartPole-v1 --output runs/paper
./scripts/run examples.elitelist_papers.run --resume runs/paper
```

`scripts/run MODULE [ARGS...]` builds `rsikit:local` using Docker's existing layer
cache, then executes `python -u -m MODULE ARGS...` in one foreground container.
It locates the repository relative to itself, so invocation works from another
directory. It forwards arguments as an array, without `eval` or shell interpolation.
Docker is the only required host runtime; a host Python environment is unnecessary.

The launcher uses `--rm --init`, adds a TTY only for an attached terminal, forwards
signals and exit status, and does not mount the Docker socket or install a Docker
client inside the application image. The application requires outbound network
access for model APIs. There is no host/controller RPC protocol.

The repository's `runs/` is mounted at `/app/runs`. `RSIKIT_RUNS_DIR` can select
another host output root. Container commands use `runs/...` for outputs and
resumes; arbitrary absolute host paths are not rewritten. Application source,
templates, and bundled datasets are copied into the image, not bind-mounted.

Use the invoking user's UID/GID, a writable temporary HOME, a read-only image
filesystem and writable `/tmp`. Default container limits are 4 CPUs, 8 GiB RAM,
512 PIDs and a 1 GiB `/tmp`; `RSIKIT_CPUS` and `RSIKIT_MEMORY` override CPU/RAM.
These are whole-application limits, independent of `Executor.concurrency`.

Load the repository `.env` with Docker's env-file option when present;
`RSIKIT_ENV_FILE` selects a different file. Explicitly set host
`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` override env-file
values. Never source the file as shell code or put secrets in image layers.
The launcher supplies `RSIKIT_IMAGE_ID` and, when host Git is available,
`RSIKIT_GIT_REVISION` for run provenance. The source snapshot remains authoritative
for uncommitted code; a revision alone does not describe the working tree.

## Application execution

Keep the ordinary `Policy`, `Evaluator`, `Episode`, `Run`, and `Executor` concepts.
`Executor(concurrency=1, episode_timeout=60.0)` schedules episodes inside the
current runtime. It has no sandbox/backend parameter and never launches Docker.
Its existing async result iterator and error aggregation remain compatible with
`research.rollouts.Rollouts` and the search algorithms.

Use standard-library multiprocessing with a fresh child for each episode, retaining
a clean forkserver on Linux to amortize interpreter imports. Use spawn where
forkserver is unavailable for development tests. Source loading happens only in
the child: compile the source, obtain `Solution`, check that it subclasses `Policy`,
then instantiate and call it normally. Remove `DirectPolicy`; translate candidate
exceptions at the episode execution boundary instead of wrapping every method.

Retain environment-template serialization only where multiprocessing requires it
(cloudpickle is already installed). Internal result transfer uses a standard
multiprocessing connection, not JSON envelopes, base64, request IDs, or a service.
Receive while the child runs so results larger than a pipe buffer cannot deadlock.
Keep the existing 64 MiB episode-size ceiling and saved data format.

Preserve distinct environment/policy seeds, instructions, step limits, complete
trajectories, video artifacts, successful sibling results and repairable failures.
`run_program(..., episode_timeout=60.0)` uses the same local execution path.
Parent-owned deadlines must interrupt infinite synchronous loops. Cancellation
must reap children and ordinary descendants and release temporary directories.
A pool future's timeout alone is not an implementation of termination.

Generation, search logs, progress bars, and persistence run in the main application
process. Capture evaluation stdout/stderr in its temporary directory; return the
bounded log as an artifact, or include a bounded tail in failure diagnostics.
This avoids competing writes to Rich's live display without another event service.

## Trust model

The container is the outer boundary. Generated code is trusted with the application's
network, credentials and mounted outputs. Evaluation processes provide lifecycle
control and fresh state, not a security boundary against other application processes.
Internal object transfer does not carry the old hostile-host deserialization claim.
No promise is made to contain a deliberately detached descendant until container
exit. Standard Docker restrictions remain; no inner seccomp policy is needed for
standard research episodes. Poker retains its separate isolation requirements.

## Storage and removal

`Run` currently imports the episode codec from `rsikit.sandbox.codec`. Preserve
the JSON schema and old run readability, moving episode encoding/decoding and
their numeric helpers into `rsikit/episode.py`. Serialization needed by storage
stays; sandbox transport serialization goes away.

Delete the core `rsikit/sandbox/` package after moving its remaining responsibilities
to execution, episode storage, the application image, or poker. Delete
`DockerSandbox` exports, readiness/version handshakes, Docker lifecycle code,
remote error envelopes and service-only tests. Update launch instructions,
benchmarks, notebook commands and all runner configuration together.

Poker currently imports `_spawn`, frame/reap helpers, policy loading, seccomp and
codec helpers from the core sandbox. Make those private to the poker example
before deletion. Its specialized image may extend `rsikit:local`, but its evaluator
still launches separately from the host; the generic launcher must reject the
poker module with an instruction to use poker's documented command.

## Acceptance

One launcher command builds and starts a complete standard search. A scripted
provider test proves generation, evaluation, persistence and Rich output all run
inside the application container, without real model calls or nested Docker.
Saved outputs reopen after container removal. Ctrl-C cancels active work and leaves
a resumable run. Timeout, crash, artifact and concurrency checks pass. Poker's
existing isolation tests continue passing through its separate launch path.

Measure cold startup and repeated episodes separately against the present working
tree. Preserve fresh children; do not add persistent candidate workers to recover
performance without evidence. No Compose service, remote API, registry publishing,
cloud deployment, job scheduler, or new Python dependency is required.
