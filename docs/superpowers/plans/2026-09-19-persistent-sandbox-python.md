# Persistent Python Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Native execution is recommended because the process and transport changes share closely coupled lifecycle contracts.

**Goal:** Keep the Docker container and preloaded Python runtime alive across evaluations, removing per-episode interpreter launches while preserving fresh evaluator and candidate processes.

**Architecture:** One attached Docker process carries bounded JSON requests and responses between the host and a persistent Python supervisor. The supervisor starts a clean, preloaded stdlib forkserver and creates a separate evaluator/candidate pair per episode. The host retains its existing concurrency and lifecycle API.

**Tech Stack:** Existing Python, asyncio, multiprocessing forkserver, Unix sockets, Docker, cloudpickle, Gymnasium, NumPy, unittest, and Ruff; no new dependencies.

**Spec:** [Approved design](../specs/2026-09-19-persistent-sandbox-python.md).

## Execution record

Implementation and acceptance checks completed on 2026-09-19: 160 tests passed
in 41.537 seconds; scoped lint, formatting, and whitespace checks passed. Related service,
transport, and lifecycle assertions were consolidated in
`tests/test_persistent_sandbox.py`; test class names differ from the illustrative
snippets below. The baseline and persistent benchmark JSON files retain all raw
samples. Warm packing improved 79.22×; all four batch workloads improved.

Commit steps are explicitly deferred: this checkout already contained broad
uncommitted changes, including the executor implementation this work builds on.
The changes remain on `codex/runs`, preserving the user's existing work. A saved
working-tree baseline and scoped diff were used for review; no merge or push was
performed.

Independent review delivered two actionable findings: inherited forkserver
control access and buffered candidate socket teardown. Both were reproduced,
fixed, and checked. The reviewer did not deliver a final report after tool safety
filtering; the completed delta was reviewed locally. Candidate-only Linux seccomp
now blocks new control connections (ARM64/x86-64), and teardown aborts buffered
candidate writes. A further reproduced Docker stdout backpressure hang was fixed
by draining the client during close. These changes preserve the process limit
and cleanup contracts without new dependencies.

## Global Constraints

- “The container and supervisor live for the existing async Run/Executor context.”
- “Each episode still receives fresh evaluator and candidate processes.”
- “Use the standard-library multiprocessing forkserver context explicitly inside Linux Docker.”
- “Keep the public Sandbox protocol and Run lifecycle unchanged.”
- “There are no per-episode docker exec calls or Python executable launches after startup.”
- “There is no new implicit whole-episode deadline.”
- “Preserve the 64 MiB evaluation request/result bounds and the stricter existing policy/action message limits.”
- “Keep run_program's existing single-episode behavior working.”
- Retain Python >=3.10 source compatibility and the host/image Python minor-version check. Current benchmark image uses Python 3.14.
- Generated source executes only inside Docker. Never use host-side source execution as a test fallback.
- Prefix shell commands with `rtk`. Preserve the many existing uncommitted changes, especially in sandbox, execution, and test files.

## Review Focus

1. A ready service whose forkserver silently missed a preload must fail startup; test inherited preload identity in Task 1.
2. A child that dies without writing a result must fail promptly and leave no sibling process or temporary directory; test abrupt exit in Task 1.
3. Interleaved completions, unknown IDs, duplicate IDs, or broken framing must never deliver another job's result; test routing and batch failure in Task 2.
4. Cancellation while a large request is being written must invalidate the connection, settle every waiter, and reap the Docker client; test cancellation in Tasks 2 and 3.
5. A timeout followed by repair must retain the supervisor while discarding candidate/environment state; test timeout recovery and module contamination in Task 3.

## File map and ownership

| File | Responsibility |
| --- | --- |
| New `rsikit/sandbox/service.py` | Preload marker, startup probe, bounded service protocol, episode process ownership, result forwarding |
| `rsikit/sandbox/worker.py` | Reusable resource-limited candidate entry; retain legacy supervisor |
| `rsikit/sandbox/evaluate.py` | Evaluation using an injected candidate socket and caller-owned directory; retain one-shot entry |
| `rsikit/sandbox/__init__.py` | Extract the small request exchange method so subprocess and socket transports reuse validation/timeouts |
| `rsikit/sandbox/docker.py` | Attached service connection, readiness, request routing, failure fan-out, close |
| `rsikit/sandbox/Dockerfile` | Include service and make worker package importable by isolated forkserver interpreters |
| New `tests/test_persistent_sandbox.py` | Focused service, protocol, and persistence checks |
| `tests/test_sandbox_episode.py` | Adapt existing lifecycle/cancellation coverage to the new connection |
| `rsikit/execution.py` | Update obsolete docker-exec cleanup comment only unless tests prove a necessary lifecycle correction |
| `examples/benchmark_docker.py`, `docs/DOCKER_PERFORMANCE.md` | Preserve benchmark comparability and report measured results |

No changes to generation, policy contracts, database persistence, search algorithms, or dependencies.

## Task 1: Container service and fresh preloaded episode children

**Files:** `service.py`, `worker.py`, `evaluate.py`, `__init__.py`, `Dockerfile`, `tests/test_persistent_sandbox.py`.

**Interfaces:**

- Preserve `worker.candidate(channel)` and `worker.main()`.
- Add `worker.run_candidate(channel: socket.socket, directory: str | None = None) -> None`: enter the supplied episode directory, set resource limits, redirect standard descriptors, run the existing candidate loop, close the socket. Without a directory retain the inherited working directory for legacy execution.
- Extract `SandboxPolicy._exchange(payload: bytes) -> bytes`; its default implementation uses the existing subprocess stdin/stdout. ProcessPolicy overrides it only when a candidate socket was supplied.
- Extend `evaluate.evaluate(request: dict, *, channel: socket.socket | None = None, directory: str | None = None) -> dict`. Supplied resources select the persistent-service path; omitted resources preserve the one-shot path.
- Add `evaluate.run_evaluation(request: dict, channel: socket.socket, output: socket.socket, directory: str) -> None`: emit exactly one bounded JSON result/error to the private output socket.
- Add `service.main(workers: int) -> None` and `service.probe(channel: socket.socket) -> None`.
- Ready frame: `{"ready": true, "protocol": 1, "supervisor_pid": int, "forkserver_pid": int}`.
- Request frame: `{"id": str, "request": {"implementation": str, "environment": str, "python": [int, int], "seed": int, "call_timeout": float}}`.
- Result frame: `{"id": str, "result": {"score": float, "artifacts": object}}` or the same envelope containing existing `kind`/`error` fields.
- Limit the entire serialized envelope to 64 MiB excluding its trailing newline; read at most the limit plus newline and one overflow byte. Keep action/source limits unchanged.

- [x] **1. Add a failing Docker service smoke check.** Use unittest and asyncio subprocesses as the existing sandbox tests do. Add this invocation helper to `tests/test_persistent_sandbox.py`; generated code remains remote:

```python
async def start_service(workers=2):
    return await asyncio.create_subprocess_exec(
        "docker", "run", "--rm", "--init", "-i", "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", str(64 * workers), "--memory", f"{workers}g",
        "--memory-swap", f"{workers}g", "--cpus", str(workers),
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--entrypoint", "python",
        "rsikit-sandbox:local", "-I", "-u", "-c",
        "from rsikit.sandbox.service import main; main(" + str(workers) + ")",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=64 * 1024 * 1024 + 1,
    )
```

Add `ServiceTests.test_ready_and_repeated_requests`. Serialize `CirclePackingEnv(count=1)` and use `PACKING` from `examples.benchmark_docker`; send two sequential request envelopes through this helper and assert:

```python
ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 60))
self.assertEqual(ready["protocol"], 1)
self.assertTrue(ready["ready"])
self.assertNotEqual(ready["supervisor_pid"], ready["forkserver_pid"])
for job_id in ("first", "second"):
    process.stdin.write(json.dumps({"id": job_id, "request": request}).encode() + b"\n")
    await process.stdin.drain()
    response = json.loads(await asyncio.wait_for(process.stdout.readline(), 10))
    self.assertEqual(response, {"id": job_id, "result": {"score": 0.5, "artifacts": {}}})
self.assertIsNone(process.returncode)
```

Construct `request` using the existing five fields plus `python`, with host Python minor version and call_timeout 3. In `finally`, close stdin, await process exit with a bounded timeout, drain stderr, and forcibly remove the test container if graceful shutdown fails. Give each test container a unique explicit name so cleanup is independent of Docker client state.

- [x] **2. Run the failing check.** `rtk proxy .venv/bin/python -m unittest tests.test_persistent_sandbox.ServiceTests.test_ready_and_repeated_requests -v`. Before the image change, the service import/readiness must fail. An unavailable Docker daemon is an environment blocker, not the intended test failure.

- [x] **3. Factor candidate entry and injectable evaluation.** Move the existing resource limits, standard-descriptor redirection, and candidate-loop invocation into `run_candidate`; call it from the existing worker fork branch as well as the new service child. Do not change policy source parsing or the action codec. For the supplied socket path, adapt `ProcessPolicy` to own StreamReader/StreamWriter transport rather than a subprocess; its `_request` behavior and call deadlines remain the same. Close the socket on timeout, leaving child termination to its owner. Keep the legacy subprocess path intact.

The concrete async socket adapter is:

```python
reader, writer = await asyncio.open_connection(sock=channel, limit=MAX_MESSAGE + 1)
```

Use the current request-building and response-validation logic. Do not imitate a subprocess with a dynamically fabricated object. Move the inner exchange closure from SandboxPolicy._request into `_exchange`, retaining the existing wait_for around the call. In the direct candidate socket branch of ProcessPolicy.reset, send the start request without waiting for the legacy supervisor's ready message; the candidate loop does not emit one. Its socket override writes/drains `writer` and reads one line from `reader`. The supplied-directory branch must not delete the directory; the supervisor owns deletion after child termination. Both children enter that same directory so candidate-written files remain discoverable as artifacts. Keep the temporary-directory context for one-shot calls.

- [x] **4. Implement preload startup and verify inheritance.** Register `rsikit.sandbox.service` as the forkserver preload module; its module import loads `numpy`, `gymnasium`, `cloudpickle`, `evaluate`, and `worker` and records the importing PID and Linux native thread count. The forkserver module itself must not create processes, threads, or run an event loop at import time.

```python
PRELOAD_PID = os.getpid()
PRELOAD_THREADS = len(list(Path("/proc/self/task").iterdir())) if sys.platform == "linux" else None

def probe(channel):
    inherited = PRELOAD_PID == os.getppid() and PRELOAD_THREADS == 1
    with channel:
        channel.sendall(json.dumps({
            "inherited": inherited,
            "forkserver_pid": os.getppid(),
        }).encode() + b"\n")
```

Define these markers after the preload imports. In `main`, use `multiprocessing.get_context("forkserver")` and `multiprocessing.set_forkserver_preload(["rsikit.sandbox.service"])`, start a probe child, validate its message/exit, and only then emit ready. Probe failure is an infrastructure startup failure, never a silent fallback to fresh interpreters. Maintain the 60-second host startup bound.

Include `service.py` in Dockerfile COPY. Make `/opt/worker` importable in all isolated interpreters, including versions whose forkserver does not restore the supervisor's sys.path during preload:

```dockerfile
RUN python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'rsikit-worker.pth').write_text('/opt/worker\\n')"
```

Inspect the resulting file during the image check; it must contain a real newline and the existing worker directory. Keep the legacy image ENTRYPOINT.

- [x] **5. Implement episode ownership and the bounded service loop.** The supervisor receives/validates an envelope, creates a temporary directory and two socketpairs, and starts sibling processes from the clean forkserver:

```python
policy_side, environment_side = socket.socketpair()
result_reader, result_writer = socket.socketpair()
candidate = context.Process(target=run_candidate, args=(policy_side, directory.name))
evaluator = context.Process(
    target=run_evaluation,
    args=(request, environment_side, result_writer, directory.name),
)
candidate.start()
policy_side.close()
evaluator.start()
environment_side.close()
result_writer.close()
```

Wrap partial startup in cleanup so a failed second start still kills/joins the first. The forkserver only transfers each child's explicit sockets; candidate never receives the request containing the environment or the result socket. Before generated source runs, redirect inherited stdout/stderr so it cannot write service frames. Duplicate/reserve the supervisor output descriptor and redirect its ordinary stdout before starting the forkserver too.

Use asyncio I/O for job sockets and a write lock for service stdout. Track no more than `workers` active jobs. Reject duplicate IDs and over-capacity direct calls without launching children. Monitor evaluator exit as well as result arrival; EOF or exit without a valid result yields InfrastructureError. In every job's `finally`, kill still-running direct children, join both, close sockets, and remove its temporary directory before emitting the completion. On stdin EOF or protocol failure cancel/drain all jobs and exit. Container removal remains the final cleanup for descendants created by trusted environments.

- [x] **6. Add focused service failure assertions.** Extend `ServiceTests` with a candidate that calls `os._exit(7)` and then a successful packing request on the same service. Assert that the first returns a policy/transport diagnostic within the existing call deadline and the second scores 0.5. Add an environment whose `reset` calls `os._exit(8)`; assert an infrastructure error without waiting indefinitely. Record episode directory paths via a trusted test environment and inspect from a subsequent trusted environment that the paths no longer exist. Add one malformed request and one request over the frame limit; assert bounded failure and cleanup, with no generated source executed.

- [x] **7. Build and run the service checks.** Save the current image ID for the benchmark record, then build:

```sh
rtk proxy docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
rtk proxy .venv/bin/python -m unittest tests.test_persistent_sandbox.ServiceTests -v
```

Run the legacy state/instructions and restricted-library tests after the worker refactor:

```sh
rtk proxy .venv/bin/python -m unittest tests.test_sandbox_episode.SandboxEpisodeSmoke.test_state_instructions_and_examples tests.test_sandbox_episode.SandboxEpisodeSmoke.test_scientific_libraries_in_restricted_policy -v
```

Do not relabel skips as passes. If passing, commit only this task's changes; preserve all pre-existing user edits in the same files by staging only the intended hunks.

## Task 2: Persistent Docker connection and host request routing

**Files:** `rsikit/sandbox/docker.py`, cleanup comment in `rsikit/execution.py`, `tests/test_persistent_sandbox.py`.

**Interfaces:** Preserve `DockerSandbox.start(workers)`, `evaluate(implementation, environment, seed, call_timeout)`, and `close()`. Add private `_read_results()` and `_fail_pending(error)` methods. Keep the attached subprocess as `self.process`; a pending-future dictionary maps job IDs to responses. Retain `self.name` behavior and cleanup retryability.

- [x] **1. Add failing transport checks using the real service.** Add `TransportTests.test_connection_reused_and_responses_routed`:

```python
sandbox = DockerSandbox()
try:
    await sandbox.start(2)
    process = sandbox.process
    first, second = await asyncio.gather(
        sandbox.evaluate(PACKING, cloudpickle.dumps(CirclePackingEnv(count=1)), 1, 3),
        sandbox.evaluate(PACKING, cloudpickle.dumps(CirclePackingEnv(count=1)), 2, 3),
    )
    self.assertEqual(first, (0.5, {}))
    self.assertEqual(second, (0.5, {}))
    self.assertIs(sandbox.process, process)
    self.assertIsNone(process.returncode)
finally:
    await sandbox.close()
self.assertIsNotNone(process.returncode)
```

Also send two distinguishable seed-dependent scores, delay the first environment, and assert responses arrive in reversed completion order while each caller gets its own score. Use a trusted cloudpickled test environment to introduce the delay; never execute candidate source locally.

- [x] **2. Run before changing DockerSandbox.** `rtk proxy .venv/bin/python -m unittest tests.test_persistent_sandbox.TransportTests -v`. Expect failure because the current backend has no retained service process and launches docker exec per evaluation.

- [x] **3. Replace detached sleep with attached service startup.** Preserve `_spawn` and its cancellation-safe acquisition. Use the existing container restrictions, `--init`, `-i`, and an explicit Python entrypoint invoking `service.main(workers)`. Pipe stdin/stdout, retain the process, and wait for a validated protocol-1 ready frame before returning. Do not leave stderr as an undrained PIPE; send it to DEVNULL or drain it into a bounded diagnostic buffer. Reject repeated start and nonpositive workers. Preserve cleanup if startup partially succeeds.

- [x] **4. Route evaluate calls through the retained process.** Allocate a monotonically unique string ID, encode and size-check the full envelope before registering work, register its future, and lock only the write/drain operation:

```python
future = asyncio.get_running_loop().create_future()
self._pending[job_id] = future
try:
    async with self._write_lock:
        self.process.stdin.write(payload + b"\n")
        await self.process.stdin.drain()
    result = await future
finally:
    self._pending.pop(job_id, None)
```

Run one `_read_results` task per service. Validate bounded newline framing, envelope keys, ID type/membership, error/result shapes, finite numeric score, string artifact names, and base64 payloads before resolving a future. Reject bool as a score. Preserve PolicyError/PolicyTimeout/InfrastructureError mapping. A completed ID is removed immediately, so a duplicate response invalidates the session. Do not infer a per-episode timeout from call_timeout: it is a per-policy-call limit enforced inside the evaluator.

- [x] **5. Implement failure fan-out and deterministic close.** On reader death, malformed result, write failure, or cancellation, mark the session unusable and fail every pending future. Remove the container once using a close lock; keep the name until successful removal so Executor can retry cleanup. Closing must not await/cancel the reader from inside itself. Reap the Docker client after removal and drain/cancel the reader. Avoid unobserved future exceptions when the cancelled caller's future is no longer awaited. Do not replace an original cancellation or policy error with a cleanup failure.

- [x] **6. Pin framing and cancellation failures without Docker timing races.** Add `ProtocolTests` using an `asyncio.StreamReader` and pending futures to exercise `_read_results`. Feed a valid envelope for a nonexistent ID, a valid response twice, malformed JSON, a missing newline at EOF, and a frame exceeding 64 MiB. Each case must leave all pending futures resolved with InfrastructureError and the session unusable. Use one table-driven test, with a smaller patched frame limit for the overflow case. Use valid finite-score/base64 and invalid NaN/bool/base64 payloads as parallel cases. Record each consumed future exception so the test also catches event-loop warnings.

Add a mocked writer whose `drain` blocks, cancel the evaluating task, and assert the original CancelledError propagates, siblings fail, and close runs. This targets the actual write boundary rather than relying on payload size to force backpressure.

- [x] **7. Run focused checks and commit the task.**

```sh
rtk proxy .venv/bin/python -m unittest tests.test_persistent_sandbox tests.test_run -v
rtk proxy .venv/bin/ruff check rsikit/sandbox rsikit/execution.py tests/test_persistent_sandbox.py
```

Require no leaked container, Docker client, or pending asyncio task. Stage only intended hunks; do not commit unrelated current sandbox edits as collateral.

## Task 3: Prove persistence, isolation, cleanup, and measured improvement

**Files:** `tests/test_persistent_sandbox.py`, `tests/test_sandbox_episode.py`, `examples/benchmark_docker.py` only if compatibility needs it, `docs/DOCKER_PERFORMANCE.md`, new benchmark JSON artifact.

**Interfaces:** Exercise the unchanged Executor/Run API and the transport established in Tasks 1–2. The original benchmark JSON remains the baseline.

- [x] **1. Extend the real-Docker persistence check.** Reuse `test_batches_share_one_container_with_independent_environment_processes`. Keep its four distinct evaluator PID assertions, one hostname assertion, unchanged host environment assertion, and cleanup assertion. Also record candidate PIDs in an episode-local artifact and verify four distinct candidate PIDs. Capture the service's ready PIDs and attached Docker process before the second batch and assert they remain unchanged. Assert evaluator/candidate parents identify the same forkserver. Do not add production diagnostic RPCs solely for tests; use existing process state and trusted test artifacts.

- [x] **2. Add fresh-state and timeout recovery checks.** A candidate mutates a module global after checking it is absent:

```python
import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        assert not hasattr(np, "_rsikit_episode_marker")
        np._rsikit_episode_marker = True
        return np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
```

Evaluate this source twice sequentially using one persistent sandbox and assert both scores are 0.5. Separately use a trusted environment whose reset makes the same absent-then-set check on a distinct module-global marker; both episodes must pass. This checks actual interpreter contamination, not just fresh Policy instances.

Within one async Executor context run a candidate whose `act` loops forever using call_timeout 0.1, assert PolicyTimeout, then run PACKING successfully. Assert the same container and supervisor remain alive, and the timed-out job's children are gone. Assert the next episode has no leftover temporary files. Retain the existing standalone behavior where Executor closes after each call.

- [x] **3. Adapt cancellation and service-death coverage.** Change the existing test that delays `docker exec` creation to delay the attached `docker run` process acquisition instead. Preserve assertions that acquisition completes before cleanup, that cancellation is not replaced, and that no loop errors or container remain. Add cancellation after the service ready message while two episodes are active. Kill the supervisor using a trusted test-only Docker command while two callers are waiting; both must raise InfrastructureError within a test-level watchdog, after which the sandbox closes and the next evaluation starts a fresh service.

Use explicit events/readiness acknowledgements rather than sleep-based ordering. Test-level `asyncio.wait_for` watchdogs are test bounds, not changes to production episode semantics.

- [x] **4. Run the regression checks using the rebuilt image.**

```sh
rtk proxy .venv/bin/python -m unittest tests.test_persistent_sandbox tests.test_sandbox_episode tests.test_run tests.test_inner_loop -v
rtk proxy .venv/bin/ruff check rsikit/sandbox rsikit/execution.py tests/test_persistent_sandbox.py tests/test_sandbox_episode.py examples/benchmark_docker.py
rtk proxy .venv/bin/ruff format --check rsikit/sandbox/service.py tests/test_persistent_sandbox.py
rtk git diff --check
```

The sandbox suite includes scientific-library imports, Box2D, video artifacts, policy repair, and legacy execution. Resolve failures caused by this change. Report unrelated pre-existing failures precisely; do not expand into unrelated fixes. Inspect Docker after the run and compare with the initial container list, removing only containers created by these checks.

- [x] **5. Benchmark and compare.** Run the existing harness serially with respect to other test work:

```sh
rtk proxy .venv/bin/python -m examples.benchmark_docker --output docs/docker-benchmark-persistent-python-2026-09-19.json
```

Retain all original workloads, seeds, sample counts, and concurrency values. The existing `container_start` metric now means service-ready startup including preload; document that distinction. The isolated docker-exec diagnostic probes in the harness remain useful controls even though the production path no longer uses them. Update no benchmark thresholds to hide regressions.

Compare warm packing latency and the four batch medians:

```python
import json
from pathlib import Path
before = json.loads(Path("docs/docker-benchmark-2026-09-19.json").read_text())
after = json.loads(Path("docs/docker-benchmark-persistent-python-2026-09-19.json").read_text())
for name in ("warm_packing_episode", "packing_batch16_c1", "packing_batch16_c4",
             "cartpole500_batch16_c1", "cartpole500_batch16_c4"):
    old = before["measurements"][name]["median_ms"]
    new = after["measurements"][name]["median_ms"]
    print(f"{name}: {old:.1f} -> {new:.1f} ms; {old / new:.2f}x")
```

Require every episode correctness assertion to pass and a repeatable improvement in warm execution. If the observed ranges overlap enough to obscure improvement, perform one additional comparable benchmark and report uncertainty. If repeated interpreter launches remain or warm execution regresses, investigate before claiming completion. Record startup cost, process count, and idle memory alongside the warm results; do not call the original exec-import diagnostic a remaining production bottleneck after it has been removed.

- [x] **6. Update the report and finish review.** Add the new image ID, timings, methodology change, measured speedup, and any remaining limitations to `docs/DOCKER_PERFORMANCE.md`; retain the original measurements as historical evidence. Commit only the scoped implementation/test/report changes after checks pass. Review the final diff against the approved spec and obtain the execution-method-appropriate review before reporting completion.

## Plan self-review

- Spec coverage: Task 1 covers forkserver/preload, separate children, resources, artifacts, service framing, and legacy compatibility; Task 2 covers the persistent host connection and failure lifecycle; Task 3 covers end-to-end isolation, recovery, cleanup, and performance acceptance.
- Review Focus: preload and child-exit cases are in Task 1; response routing/framing and blocked-write cancellation are in Task 2; startup cancellation and timeout/state recovery are in Task 3.
- All new cross-task names and wire frames are defined above. Existing `_request`, `_spawn`, Executor, and Run interfaces come from the current checkout.
- Scope remains one execution backend, with no dependency additions, detached daemon, or persistent generated-policy objects.
