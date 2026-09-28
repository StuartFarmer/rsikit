# Containerized RSIKit Application Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan
> task-by-task in the current session. Track the checkboxes below. Implemented on the user-provided branch; see the execution report below.

**Goal:** Launch a complete research application with one Docker command and
remove the standard research sandbox subsystem.

**Architecture:** A root Dockerfile packages the full application. A short host
launcher builds and runs it; ordinary Python inside owns generation, local
evaluation processes, persistence and Rich. Poker remains a separate application.

**Tech Stack:** Docker CLI, Bash, Python, asyncio/multiprocessing, existing
cloudpickle, Gymnasium, Slick, SQLModel and Rich. No new Python dependency.

**Spec:** [Agreed design](../specs/2026-09-28-containerized-application.md).

## Global constraints

- Docker runs the complete standard research application: generation through UI.
- The first migration covers the standard research runners; preserve poker separately.
- Keep Python >=3.10 library support and Python 3.14 as the image's default.
- No Docker client/socket inside the standard application container.
- Keep existing saved episode JSON and run databases readable.
- `Executor(concurrency=1, episode_timeout=60.0)` replaces backend selection.
- Keep fresh evaluation processes and the existing 64 MiB episode-size ceiling.
- Generated code shares the application's credentials, network and mounted outputs.
- Preserve the current working tree; this supersedes, rather than rolls back, its refactor.

## Review focus

- Spaces in output paths and arguments: preserve exact arguments and the mounted output root (task 2).
- Piped execution and Ctrl-C: no forced TTY, correct exit status, no surviving container (task 2).
- Large results and infinite loops: receive concurrently and enforce real process deadlines (task 1).
- Old run records and changed execution provenance: reopen/resume without rewriting evidence (tasks 1 and 3).
- Poker's imports and hidden-card boundary: core deletion must not weaken or break its separate evaluator (task 4).

## File map

| Files | Responsibility after migration |
| --- | --- |
| `Dockerfile`, `.dockerignore`, `scripts/run` | Build the whole application; launch with terminal, credentials and output mount |
| `rsikit/execution.py` | Existing batch scheduling plus local child lifecycle and `run_program` |
| `rsikit/evaluation.py` | Ordinary policy/environment episode loop and lifecycle preparation |
| `rsikit/policy.py` | Policy contract and source loading; no remote-policy wrapper |
| `rsikit/episode.py`, `rsikit/run.py` | Episode values, unchanged disk serialization, persistence |
| `examples/elitelist_papers/poker/*` | All remaining specialized sandbox helpers, private to poker |
| `rsikit/sandbox/` | Deleted after its remaining consumers have migrated |

Prefer moving and reducing existing functions to adding new manager classes.
Do not introduce a replacement sandbox package under a different name.

### Task 1: Replace remote evaluation with local application execution

**Files:** Modify `rsikit/execution.py`, `rsikit/evaluation.py`, `rsikit/policy.py`,
`rsikit/episode.py`, `rsikit/run.py`, `rsikit/__init__.py`.
Create `tests/test_execution.py`; adapt `tests/test_evaluator.py`,
`tests/test_episode_storage.py`, `tests/helpers.py` and existing executor tests.

**Interfaces:** Keep `Executor.evaluate(jobs, environment)` yielding
`(policy_id, seed, Episode)`, async context ownership and `aclose()`.
Use `Executor(*, concurrency=1, episode_timeout=60.0)`.
Move `run_program(program, make_env, *, env_seed=None, policy_seed=None,
max_steps=None, instructions=None, episode_timeout=60.0)` to execution and retain
its top-level `rsikit` export. Move the current `load_policy` function to policy;
it returns an actual `Solution` instance, not another `Policy` wrapper.
Keep `encode_episode` / `decode_episode` signatures in `rsikit.episode` for storage.

- [x] Write failing behavioral checks: candidate/environment share a child PID;
  that PID differs from the controller; four episodes have fresh state; distinct
  seeds and instruction overrides survive; every transition and video artifact returns.
- [x] Add timeout/cancellation tests using an infinite synchronous loop and an
  ordinary subprocess. Assert the failed process tree is gone, successful siblings
  remain available, and a later episode works. Test a result larger than the OS pipe
  buffer under a deadline, abrupt exit, invalid action, and failed initialization.
- [x] Pin old JSON compatibility with a hand-written saved episode fixture and
  reopen a pre-migration Run. Preserve source identity and cached score semantics.
- [x] Run the new checks against the current implementation and inspect failures.
- [x] Reuse the existing semaphore/result aggregation. Replace sandbox lifecycle
  and requests with fresh `multiprocessing.Process` children and standard connection
  transfer. Retain forkserver on Linux; use spawn where unavailable. Keep child
  process-group cleanup in the parent and consume results before joining children.
- [x] Instantiate `Solution` directly; keep generation/validation source-only until
  the child loads it. Classify policy load/call failures and invalid actions as
  `PolicyError`, deadlines as `PolicyTimeout`, and runtime failures as
  `InfrastructureError`, preserving current repair-loop behavior. Keep exception
  conversion in the execution boundary, not in a replacement policy subclass.
- [x] Move environment preparation/cleanup into evaluation, preserving the ordinary
  `Evaluator` API. Redirect child prints into a log artifact and include at most
  4 KiB of its tail in failure diagnostics. Enforce the existing artifact/result cap.
- [x] Move disk codec functions into `episode.py` without changing their representation.
  Retain cloudpickle only for environment definitions required by multiprocessing.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_execution
  tests.test_evaluator tests.test_episode_storage -v`; require all checks to pass.

### Task 2: Package and launch the complete application

**Files:** Create root `Dockerfile`, `.dockerignore`, `scripts/run`,
`tests/test_container_launch.py`. Move numerical dependency checks out of the
old sandbox package into `tests/test_scientific_libraries.py`; update
`pyproject.toml` source-distribution includes for the launcher and Dockerfile.

**Interfaces:** `./scripts/run MODULE [ARGS...]` always performs a cached build of
`rsikit:local`, then runs `python -u -m MODULE ARGS...` with working directory `/app`.
Environment overrides: `RSIKIT_RUNS_DIR`, `RSIKIT_ENV_FILE`, `RSIKIT_CPUS`,
`RSIKIT_MEMORY`; defaults are repository `runs/`, optional repository `.env`,
4 CPUs and 8 GiB RAM. Application arguments are passed unchanged.

- [x] Write launcher tests with an executable Docker stub recording argument
  arrays: another working directory, spaces in paths, multiword arguments, missing
  Docker/build failure, unset credentials, and propagation of a nonzero exit code.
- [x] Run the failing tests before implementing the launcher.
- [x] Adapt the existing scientific-library image: install the full local project
  with model-provider, Box2D and video extras; copy `research/`, `examples/`, tests,
  Jinja templates, documentation needed for snapshots, and bundled datasets.
  Install dependencies before copying frequently edited source to use layer caching.
  Keep CPU-only Torch and single-thread numerical-library environment defaults.
- [x] Exclude `.git`, `.venv`, secrets, run databases, outputs, logs and caches from
  the build context. Do not copy the old minimal replacement `rsikit.__init__`.
- [x] Implement a Bash launcher with quoted arrays, repository-relative paths,
  `--rm --init`, conditional terminal attachment, invoking UID/GID, temporary HOME,
  read-only image, writable `/tmp` (1 GiB), 512 PID limit, and the output bind mount.
  Forward configured credentials at runtime only, without printing their values.
  Use `exec docker run ...` for signal and exit-status propagation.
- [x] Inspect the built image on the host and pass its ID as `RSIKIT_IMAGE_ID`;
  pass `RSIKIT_GIT_REVISION` when Git is available. Neither requires a Docker
  client or Git checkout inside the container. Keep the full source snapshot to
  identify uncommitted changes.
- [x] Reject the specialized poker module with its separate launch instructions;
  no implicit nested-Docker fallback. Keep normal outbound networking enabled.
- [x] Build the image; check imports, templates, datasets and scientific operations
  inside it. Verify a synthetic secret file and existing run data are absent from
  image layers, while a runtime sentinel variable is visible to the app.
- [x] Run a real foreground/PTY smoke and a redirected-output smoke. Send SIGINT
  during evaluation; verify cleanup, exit status and preserved mounted output.

### Task 3: Migrate runners, progress and resume

**Files:** Modify `examples/inner_loop/__init__.py`, `examples/elitesearch.py`,
`examples/alphaevolve.py`, `examples/shinkaevolve.py`, `examples/lineagesearch.py`,
`examples/blackjack_videos.py`, `examples/elitelist_papers/run.py`,
`examples/elitelist_papers/paper1.ipynb`, `examples/benchmark_evaluator.py`,
`README.md`, `docs/RUNS.md`, `docs/INNER_LOOP.md`, `docs/IN_PROCESS_SANDBOX.md`
and relevant example guides/tests. Remove `examples/benchmark_docker.py` after
moving its reusable fixtures and useful cold/warm measurements.

**Interfaces:** Existing module CLI options and `runs/...` output/resume paths.
Replace `sandbox=DockerSandbox(episode_timeout=x)` with `episode_timeout=x` on
`Executor`; default callers need no execution-backend configuration.

- [x] Add an end-to-end container test with the existing scripted LLM provider:
  generation, evaluation and `Run` persistence occur in that one container; the
  expected score, exported policy, episode JSON and readable Rich output exist.
  Use no paid API calls. Reopen the mounted run in a second container and verify
  successful seeds are reused and unfinished work resumes.
- [x] Migrate real callers and test doubles to the local executor contract.
  Keep search algorithms and `Rollouts` behavior unchanged. Do not add a generic
  backend interface solely to accommodate existing fake-sandbox tests.
- [x] Leave Rich/logging in the main Python process. Remove host log forwarding,
  remote progress parsing and Docker readiness messages for standard runners.
- [x] Update snapshots to include the root Dockerfile and launcher. Replace
  in-container `docker image inspect` / host-Git lookups with provenance supplied
  by the launcher: `RSIKIT_IMAGE_ID` and optional `RSIKIT_GIT_REVISION`, plus the
  existing source snapshot.
  Preserve original run manifests and append execution changes on resume.
- [x] Change notebook-generated commands to use `scripts/run` and mounted
  `runs/...` paths. Keep notebook analysis on the host and historical-data loading
  compatible. Compile notebook cells and exercise generated CLI arguments.
- [x] Run all standard runner, repair, storage, progress and resume tests in the
  application image; confirm no test depends on a nested Docker daemon.

### Task 4: Detach poker dependencies and delete the obsolete subsystem

**Files:** Modify `examples/elitelist_papers/poker/{pool,service,worker,candidate,codec}.py`,
its `Dockerfile` and README. Delete `rsikit/sandbox/` after migrating all consumers.
Replace `tests/test_docker_sandbox.py`, `tests/test_persistent_sandbox.py` and
sandbox transport tests with the applicable execution/launcher checks; retain
still-relevant poker transport tests and saved-episode codec checks.

**Interfaces:** Poker keeps its current host launcher, `TablePool`, private UIDs,
hidden-card protection and per-table cancellation. Its Dockerfile extends the new
application image; poker does not become part of the generic launcher.

- [x] Move `_spawn` into poker's pool module; move frame/read/reap helpers into its
  service; move policy loading, source bounds and seccomp dependencies into its
  private candidate module. Share only genuine application/storage utilities.
- [x] Adjust poker's image and imports, preserving its current UID changes and
  capabilities. No new security model or poker algorithm changes.
- [x] Rebuild poker and run its existing real-Docker, cancellation, private-memory,
  Numba, scoring and display tests. Require unchanged outcomes.
- [x] Remove the standard sandbox package, public `DockerSandbox`, service protocols,
  backend selector code, obsolete benchmarks and tests that only exercise deleted
  transports. Keep historical benchmark reports labelled as historical.
- [x] Search all runtime imports, docs and notebook sources for stale sandbox
  dependencies. Confirm the standard application has no Docker subprocess calls.

### Task 5: Verify the complete workflow and measure the result

- [x] Run the full repository suite inside the new application container, plus
  paper-runner tests and poker's separate suite. Run Ruff on all changed code.
  Report pre-existing unrelated check failures explicitly.
- [x] Smoke a reviewed fixed policy, a scripted multi-generation search, timeout,
  cancellation and resume through the real launcher. Outputs must survive `--rm`.
- [x] Compare cold build/start separately from warm Packing, CartPole and Blackjack
  evaluations using the same reviewed sources/seeds as the existing benchmark.
  Record process startup and useful evaluation time; explain regressions rather
  than rebuilding a persistent worker service speculatively.
- [x] Verify the delivered user instructions reduce to `scripts/run MODULE ARGS`,
  environment setup and the output directory. Review the final diff for leftover
  wrappers and duplicate lifecycle code. Keep implementation changes reviewable;
  do not publish an image or push changes as part of this plan.

## Execution recommendation

Implement inline in this order, followed by one focused independent review.
These tasks share execution and persistence contracts, so parallel implementation
would add coordination without much benefit. The plan is complete when the
standard workflow uses one application container and the old core sandbox package
is gone; preserving poker separately is an explicit scope choice.

## References

- [Docker run](https://docs.docker.com/engine/containers/run/): terminal attachment,
  bind mounts, runtime environment and process exit status.
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html):
  process contexts, connections and process lifetime management. Terminating a
  process does not itself terminate descendants, so lifecycle tests remain necessary.

## Implementation evidence

Completed on 2026-09-28. See [execution behavior and timings](../../IN_PROCESS_SANDBOX.md#application-migration-measurements--2026-09-28)
and [raw measurements](../../application-benchmark-2026-09-28.json).
The image, launcher, local executor, runner migration and core sandbox removal are implemented.
Independent review found no blocking runtime defects; corrected three documented workflows.
Cold source rebuild reused dependency layers; a clean dependency download/build was not measured.
