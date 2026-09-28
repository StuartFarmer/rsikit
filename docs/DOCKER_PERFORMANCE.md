Docker execution and Vercel Sandbox investigation — 2026-09-19

> Historical benchmark: the separate policy/evaluator and legacy host-environment
> paths described here have been removed. See [the current sandbox](IN_PROCESS_SANDBOX.md).


The baseline measurements favored keeping Docker for local execution. Container
startup was about 112 ms and was amortized across a run. Repeated Python process
and import work was the larger optimization target. The implemented persistent
Python service is measured in the final section below. Vercel is a viable remote
backend, but there is no measured evidence here that it executes these episodes
faster.

Reproduce from the repository root with the existing sandbox image:

```sh
rtk proxy .venv/bin/python -m examples.benchmark_docker --output /tmp/docker-benchmark.json
```

The [benchmark](../examples/benchmark_docker.py) uses the actual DockerSandbox,
Executor, and run_program paths, with assertions on every episode's score and
the Executor/DockerSandbox artifacts. All 228 episodes passed. Ruff lint and
format checks also passed. [Raw measurements](docker-benchmark-2026-09-19.json) include
all samples, the image ID, host versions, Docker resources, and source hashes.
No runtime implementation or dependencies were changed for the original baseline.

Measured on ARM64 macOS 26.4, Docker 28.5.1, with 12 CPUs and 7.65 GiB assigned
to Docker. Three existing containers remained running. Host Python was 3.14.2;
image Python was 3.14.7. Both used Gymnasium 1.3.0, NumPy 2.5.3, and cloudpickle
3.1.2. The image's episode, policy, environment, and worker source hashes matched
the checkout. The image was already built and cached: “cold” below means a fresh
container, not a cold Docker daemon, uncached filesystem, or image download.

| Operation | Median | Observed min–max | Samples |
| --- | ---: | ---: | ---: |
| Start restricted container, 1 CPU | 112.4 ms | 105.5–116.6 ms | 10 |
| First trivial packing episode | 731.8 ms | 714.7–758.4 ms | 10 |
| Remove container | 55.9 ms | 52.2–81.4 ms | 10 |
| Fresh container + episode + removal | 904.4 ms | 877.0–921.2 ms | 10 |
| `docker exec … true` | 60.7 ms | 57.4–65.3 ms | 10 |
| `docker exec … python -I -c pass` | 71.8 ms | 64.9–79.1 ms | 10 |
| Exec Python and import evaluator | 398.5 ms | 391.1–408.3 ms | 10 |
| Trivial packing episode, reused container | 722.3 ms | 712.1–757.7 ms | 10 |
| Legacy `run_program`, full CartPole episode including lifecycle | 739.8 ms | 730.7–792.9 ms | 10 |

| Warm batch, 16 episodes | Median batch time | Observed min–max | Episodes/second |
| --- | ---: | ---: | ---: |
| Packing, 1 worker | 11.663 s | 11.609–11.854 s | 1.37 |
| Packing, 4 workers | 3.143 s | 3.098–3.147 s | 5.09 |
| CartPole, 1 worker | 13.522 s | 13.323–14.560 s | 1.18 |
| CartPole, 4 workers | 3.813 s | 3.626–3.892 s | 4.20 |

Warm measurements exclude one warmup. Packing uses one circle with a known
optimal score of 0.5, intentionally minimizing computation to expose execution
overhead. CartPole uses the repository's hand-written policy, seed 1, and checks
that all episodes earn 500. Batch measurements use 16 jobs, three measured batches
after a warmup episode, and retain the container across batches. Concurrency 1
allocates 1 CPU/1 GiB; concurrency 4 allocates 4 CPUs/4 GiB. Throughput is jobs
divided by median batch wall time, not individual job latency.
Four workers improve packing throughput 3.71× and CartPole throughput 3.55×.

The baseline production flow was `Run → Executor → DockerSandbox.evaluate → evaluator
process → policy supervisor → forked candidate`. An async Run/Executor context
already retains its container. Each baseline evaluation launched `docker exec`, a
new evaluator interpreter, and a second interpreter for the policy supervisor.
Both imported the Python/Gymnasium/NumPy stack. Importing the evaluator alone adds
roughly 327 ms over the empty Python command in this diagnostic comparison.
These are separate measurements, not an additive profiler breakdown, but they
identify a much larger cost than container creation. The host exchanges one
request/result per episode; action-by-action communication stays inside Docker.

The older `run_program` path took 740 ms for a complete CartPole episode,
including container lifecycle, versus approximately 845 ms per episode in the
serial Executor batch. The older path keeps the environment on the host and
runs only the policy in Docker, avoiding the extra evaluator interpreter while
sending actions/observations through Docker's attached stream. Its container
also has a different memory limit (512 MiB versus 1 GiB). This is a useful
architectural comparison, not equivalent isolation or a reason to move the
current environment evaluation back onto the host.

These results motivated the persistent preloaded supervisor measured below,
retaining fresh isolated evaluator and candidate processes per episode.
Do not pool candidate instances or remove process isolation merely to reduce
latency. AlphaEvolve, ShinkaEvolve, and LineageSearch already default to four
workers; EliteSearch defaults to eight. Standalone Executor defaults to one.

Vercel's current offering fits the broad requirements:

| Requirement | Finding and consequence |
| --- | --- |
| Python integration | An official async Python SDK exists in the `vercel` package. It can implement the existing `start/evaluate/close` protocol. |
| Dependencies | Custom OCI images are supported through Vercel Container Registry. Rebuild this ARM64 image for `linux/amd64`, pin its digest, and preinstall dependencies. Python 3.14 managed images also exist. |
| Request transport | The Python SDK explicitly does not support process stdin. Upload a uniquely named JSON request file, run the existing evaluator with stdin redirected from that file, capture its JSON stdout, and remove the request afterward. Keep source out of interpolated shell commands. |
| Lifecycle | Reuse one sandbox across the run, with persistence disabled for disposable evaluations; destroy it at the end. Preserve cancellation cleanup, policy errors, call timeouts, and result-size validation. |
| Isolation | Firecracker supplies a VM boundary. Configure deny-all networking and verify an unprivileged worker, process limits, and filesystem protections. Docker runtime flags do not automatically transfer with an image. |
| Region | Select a nearby European region for this Madrid-based client. The default is `iad1`; available choices include Paris, Frankfurt, and London. |

The SDK supports process execution, filesystem operations, lifecycle management,
and deny-all network policies. Its lack of stdin requires the transport change
above. [Python SDK reference](https://vercel.com/docs/sandbox/python-sdk-reference).
VCR requires an optimized AMD64 image, and Sandbox does not execute Docker
ENTRYPOINT/CMD automatically. [Image documentation](https://vercel.com/docs/sandbox/concepts/images).
Region selection and the default are documented in
[Sandbox regions](https://vercel.com/docs/sandbox/concepts/regions).

Vercel advertises millisecond startup, but publishes no comparable end-to-end
measurement on that page for this workload. That claim excludes no specific
network, transfer, import, or result-retrieval costs. Treat it as a vendor startup
claim, not a benchmark against our 722 ms warm episode. A direct port retains
the repeated Python work and adds remote transport. Faster remote CPUs or more
parallel capacity could still win; that remains unmeasured.
[Vercel Sandbox overview](https://vercel.com/docs/sandbox).

At the default `iad1` rates, paid usage costs $0.128 per active CPU-hour and
$0.0212 per provisioned GB-hour, with a one-minute minimum memory charge. A
fully utilized 4-vCPU/8-GB sandbox therefore costs approximately $0.6816/hour
for CPU and memory; idle provisioned memory alone costs $0.1696/hour. These
exclude transfer, storage, and creation charges; regional rates vary. Hobby
allows 10 concurrent sandboxes and 45-minute sessions; Pro allows 10,000 and
24-hour sessions. Reusing a sandbox avoids repeated provisioning and minimum
memory charges. [Pricing and quotas](https://vercel.com/docs/sandbox/pricing).

No live Vercel benchmark was run. This process had no Vercel environment
variables, and the repository had no linked `.vercel/project.json`. No image or
project code was uploaded. A useful next comparison would use the same pinned
Python dependencies, workloads, scores, and 1/4-worker batches in a prepared
AMD64 image, recording creation, upload, execution, download, and cleanup
separately. Compare warm batch throughput and latency before choosing a backend.

These measurements cover execution overhead and a small Gymnasium workload.
They do not characterize solver-heavy policies, Torch imports, videos, large
artifacts, maximum concurrency, or unloaded-host performance. Ten samples and
three batches support this local comparison, not reliable tail-latency claims.

Persistent Python implementation — measured 2026-09-19

Implemented the approved persistent supervisor and preloaded forkserver. The
container, supervisor, forkserver, and attached Docker client remain alive for
the existing async Run/Executor context. Each episode gets a fresh evaluator
and candidate process, inheriting NumPy, Gymnasium, and cloudpickle imports.
Closing the context removes the container. Standalone execution still closes
after its call; legacy `run_program` retains its existing behavior.

| Operation | Baseline median | Persistent median | Speedup |
| --- | ---: | ---: | ---: |
| Warm packing episode | 722.3 ms | 9.1 ms | 79.22× |
| 16 packing episodes, 1 worker | 11663.4 ms | 252.8 ms | 46.13× |
| 16 packing episodes, 4 workers | 3142.7 ms | 61.5 ms | 51.09× |
| 16 CartPole episodes, 1 worker | 13521.7 ms | 948.7 ms | 14.25× |
| 16 CartPole episodes, 4 workers | 3812.9 ms | 346.0 ms | 11.02× |

Packing throughput increased from 1.37 to 63.29 episodes/s at one worker and
from 5.09 to 260.12 episodes/s at four workers. CartPole throughput increased
from 1.18 to 16.86 episodes/s and from 4.20 to 46.24 episodes/s respectively.
Warm packing's observed range was 8.5–44.0 ms, versus 712.1–757.7 ms before.
All four batch ranges were also entirely below their baseline ranges.

Startup now waits for imported Python runtimes and a successful inheritance
probe: 905.0 ms median, versus 112.4 ms for starting the original sleeping
container. The first packing evaluation then takes 11.0 ms and removal 61.8 ms.
A fresh container plus one episode plus removal takes 981.6 ms, versus 904.4 ms
before (8.5% slower). This change benefits repeated evaluations within one
context; it does not improve one-off lifecycle latency.

The unchanged benchmark ran 228 episodes with all score/artifact assertions
passing, using the same workloads, seeds, sample counts, CPU/memory limits,
and three existing background containers. The standalone exec/import control
remained approximately 400 ms, but production evaluations no longer call
`docker exec` or start a Python executable per episode. Legacy CartPole remained
approximately unchanged at 736.0 ms. Warm costs now include child creation,
request serialization, episode work, and per-action IPC; these measurements do
not separately attribute those costs.

Reproduce and inspect the samples:

```sh
rtk proxy .venv/bin/python -m examples.benchmark_docker --output docs/docker-benchmark-persistent-python-2026-09-19.json
```

[Raw persistent measurements](docker-benchmark-persistent-python-2026-09-19.json)
include the exact image ID and implementation hashes. The rebuilt image ID is
`sha256:ba239640a37c2c7691cfaf8fcb81c627ae85c8a0efe66dd1c5881b21d6e27b52`. Python
remained 3.14.7, Gymnasium 1.3.0, NumPy 2.5.3, and cloudpickle 3.1.2. The rebuilt
Python base has a newer build date, so this is not a byte-identical base-image
comparison; the stable exec/import controls support the process-reuse finding.

After successful work, Docker reported 74.07 MiB idle memory for one configured
worker and 74.50 MiB for four. Both had four processes: init, supervisor,
resource tracker, and forkserver. Docker's PIDs counter was five because it
also counts the supervisor's output thread. No episode children remained.
These are post-run snapshots, not peak memory or a measured memory delta against
the baseline. [Idle snapshots](docker-persistent-idle-2026-09-19.json) preserve
Docker stats, process listings, and image runtime versions. Shared memory means
summing process RSS would overcount physical memory.

The rebuilt image passed all 160 tests in the full regression suite, including persistent
identities across batches, fresh evaluator/candidate module state, independent
artifacts, timeout recovery, process death, cancellation, framing, scientific
libraries, Box2D, video recording, and legacy execution. Lint and formatting
checks passed. Additional regression checks cover missing preload, oversized
requests, candidate control connections, and shutdown under socket/pipe
backpressure.

Review found that a forkserver child could otherwise reconnect to its parent's
control endpoint, bypassing the intended candidate process limit. A candidate-only
Linux seccomp filter now denies new connections and io_uring setup while retaining
the already-connected episode socket. It validates the syscall architecture,
rejects alternate ABIs, and supports ARM64 and x86-64; only ARM64 was exercised
here. Other architectures fail closed. Candidates cannot open new Unix/network
connections in this path; subprocess, network, and filesystem restrictions remain
in force. This preserves the current container boundary, not separate VM isolation
between jobs. Buffered candidate writes are discarded during timeout cleanup;
Docker output is drained during shutdown to avoid backpressure deadlocks.

No dependencies were added. Heavy scientific packages beyond the common preload
still import in each candidate. Vercel remains unbenchmarked; this implementation
requires Linux forkserver/socket/seccomp support and does not create a remote
backend or a daemon that survives application exit.
