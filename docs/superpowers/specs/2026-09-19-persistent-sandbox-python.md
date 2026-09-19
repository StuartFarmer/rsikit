Persistent Python execution inside Docker — proposed design

User intent: keep the Docker container and initialized Python runtime alive after
startup, avoiding repeated interpreter startup and imports during evaluation.
The selected approach is a persistent supervisor with preloaded child processes.
This document covers the existing Executor/DockerSandbox path.

The container and supervisor live for the existing async Run/Executor context.
Each episode still receives fresh evaluator and candidate processes. Closing the
run stops the service and removes the container. Persistence across application
restarts, a detached daemon, and reuse of candidate state are outside this change.

Measured baseline: a reused-container packing episode takes 722 ms; launching
Python and importing the evaluator takes 399 ms. Four-worker packing throughput
is 5.09 episodes/second. These measurements motivate the design but do not predict
its speedup. See [benchmark report](../../DOCKER_PERFORMANCE.md).

Process ownership

```text
Host Executor
  └─ one attached Docker connection for the run
      └─ persistent Python supervisor
          └─ persistent, preloaded Python forkserver
              ├─ fresh evaluator for episode A
              ├─ fresh candidate for episode A
              ├─ fresh evaluator for episode B
              └─ fresh candidate for episode B
```

Use the standard-library multiprocessing forkserver context explicitly inside
Linux Docker. Preload NumPy, Gymnasium, cloudpickle, and the trusted evaluation
and candidate entry modules before accepting work. Confirm the preload actually
succeeded and that it did not start background threads; fail startup with a clear
infrastructure error if these assumptions do not hold. Keep the existing BLAS
thread limits. Other scientific packages remain available for candidate imports
but are not added to the common preload set in this change.

Python documents forkserver preloading as a way for children to inherit imported
module state, and cautions that imported libraries can create threads. Validate
this on the actual image, rather than using raw fork from an active asynchronous
evaluator. [Python multiprocessing reference](https://docs.python.org/3.14/library/multiprocessing.html#multiprocessing.set_forkserver_preload).

The supervisor creates an evaluator/candidate pair from the same clean forkserver
for each job. The candidate is not forked from an evaluator that has deserialized
or stepped an environment. The candidate receives only its policy request and
observation/action channel; it does not receive the serialized environment or the
host-facing result channel. Close unused descriptors in every process. Preserve
the current process separation without claiming that processes sharing one
container provide separate VM security boundaries.

Communication and concurrency

Start the service with one attached `docker run -i` process, retaining the current
network, filesystem, privilege, CPU, memory, and PID restrictions. Use Docker's
init support to reap orphaned descendants. DockerSandbox.start waits for a
versioned ready message after preload validation and retains the process handle.
There are no per-episode docker exec calls or Python executable launches after
startup.

Frame host requests and results as bounded newline-delimited JSON with unique
job IDs. Retain the existing base64 environment/artifact encoding, Python minor
version check, result schema, and error kinds. A single host reader routes
responses to waiting evaluations; serialize writes so concurrent requests cannot
interleave. Unknown or duplicate response IDs, invalid framing, EOF, or service
death fail all pending jobs with InfrastructureError.

Executor remains the public concurrency owner. The service also rejects work
beyond its configured in-flight limit, protecting direct DockerSandbox use from
unbounded process creation. No additional scheduling policy or worker-pool library
is introduced. Completed jobs may return out of order.

Use a fresh bounded socket channel per evaluator/candidate pair and retain the
existing action codec and start/act/close protocol. The evaluator performs trusted
environment deserialization, episode stepping, reward calculation, and artifact
collection in its own temporary directory. Candidate code executes only in its
fresh process after the existing resource limits and output redirection are set.
The supervisor receives JSON result bytes, not deserialized candidate objects.

Lifecycle and failures

- Candidate start, action, and close retain their existing call_timeout behavior.
  There is no new implicit whole-episode deadline.
- The supervisor owns both child handles. On normal completion or policy failure,
  it terminates any remaining child, joins both, and cleans the episode directory
  before reporting completion. A subsequent episode receives fresh processes.
- PolicyError and PolicyTimeout remain candidate failures. Within an async run,
  they leave the persistent service available for repaired or subsequent policies.
- Caller cancellation or infrastructure failure invalidates the sandbox and fails
  pending requests. Remove the container and reap the attached Docker client;
  the next evaluation can start a fresh sandbox. This preserves the existing
  conservative cancellation contract without a new per-job cancellation protocol.
- Explicit close removes the container, resolves pending waiters, and reaps the
  reader and Docker subprocess. Preserve the original exception if cleanup fails.
- Preserve the 64 MiB evaluation request/result bounds and the stricter existing
  policy/action message limits. Validate before queueing or decoding payloads.

Scope of code changes

Keep the public Sandbox protocol and Run lifecycle unchanged. DockerSandbox owns
the long-lived connection and response routing. A container-only service module
owns forkserver startup, child ownership, and result forwarding. Adapt evaluate.py
to accept a supplied candidate channel, and factor the resource-limited candidate
entry from worker.py so both paths reuse it. Include the service in the Docker
image. Keep run_program's existing single-episode behavior working.

Verification and acceptance

Extend the existing Docker integration checks to verify stable container,
supervisor, and forkserver identities across batches, while evaluator and
candidate identities change each episode. Verify real overlap up to the requested
concurrency, unchanged scores/artifacts, and no module-global state leaking from
one candidate or evaluator to the next.

Exercise a policy timeout followed by a successful evaluation in the same async
run, supervisor death with multiple pending requests, cancellation during startup
and active work, malformed/oversized results, and complete process/container
cleanup. Update the existing cancellation test that currently intercepts a
per-episode docker exec. Run the relevant Run, inner-loop, sandbox, scientific
library, and video checks using the rebuilt image.

Rerun examples.benchmark_docker with the same workload sizes and concurrency,
saving a new result file rather than overwriting the original baseline. Record
startup separately: persistent service startup now includes imports. Acceptance
requires removal of per-episode interpreter launches, preserved functional
behavior, and a repeatable improvement in warm episode latency and batch
throughput. Report measured gains and any startup/memory tradeoff; do not promise
a numerical speedup before measuring it.
