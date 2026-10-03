Single-evaluation sandbox-runtime experiment — 2026-09-28

Historical report: the prototype was removed after choosing the consolidated
Docker backend. Measurements and validation below describe the experiment at that time.

The experimental prototype implemented the former `Sandbox`
interface using one `srt` invocation and one fresh Python interpreter per
evaluation. It reuses the direct policy runner and data-only episode codec.
There is no persistent worker, forkserver, job-ID protocol, or shared result
reader. `Executor` continues to control concurrency.

**This is an experiment for reviewed policies, not a production sandbox for
untrusted generated code.** Docker remains the default. The prototype does not
provide aggregate CPU, memory, process-count or disk limits. A detached child
can survive process-group cleanup; the benchmark demonstrated this and explicitly
terminated the test child afterward. Restricting files/network does not solve
resource exhaustion or process lifetime.

The prototype prepares a private read-only copy of the worker package once per
execution context, matching Docker's minimal `rsikit` initialization. Each job
gets its own writable working directory. The sandbox can read the configured
system runtime directories, the host virtual environment, worker code and job
directory; it cannot generally read the repository or home directory. The child
environment is built from a small allowlist, without inherited API credentials.
Requests/results are bounded JSON, with the existing trusted environment pickle
sent inward only. Policy prints go to a separate, bounded diagnostic stream.

Read permissions are deliberately specific. Different Python distributions or
scientific packages may require additional native libraries. The experiment uses
the current Python environment; it does not provision the full Docker scientific
stack or promise arbitrary host-defined environments can be imported.

Measured on ARM64 macOS 26.4, Python 3.14.2, npm package
`@anthropic-ai/sandbox-runtime` 0.0.77. The package's CLI reports version `1.0.0`.
Five alternating samples per backend after the first invocation, using identical
reviewed policies and seed 1. All scores matched. Docker retained its warm
forkserver; `srt` launched fresh interpreters for every sample.

| Workload | One-shot srt median | Warm Docker median |
| --- | ---: | ---: |
| Packing, 1 step | 270.8 ms | 19.8 ms |
| CartPole, 500 steps | 263.2 ms | 26.8 ms |
| Blackjack, 2,485 steps | 306.2 ms | 79.2 ms |

Eight packing evaluations at concurrency four took 548 ms with `srt` and 31 ms
with Docker (one batch each, after warmup). One-time backend preparation took
17 ms for the native code snapshot and 939 ms for Docker service startup. The
native preparation figure excludes starting `srt` and Python: those costs are
included in every native evaluation. This distinguishes a single cold evaluation
from a repeated research workload: the observed setup plus first packing episode
was 291 ms with `srt` and 952 ms with Docker, excluding shutdown. These are not equivalent resource budgets or
dependency environments, and they do not establish general tail latency.

The reviewed probes verified correct episodes, fresh state, concurrent jobs,
artifacts, protected synthetic secrets, blocked outbound connections, timeout
recovery, bounded diagnostic output and cancellation. Normal child processes were
cleaned up; a child using `start_new_session=True` survived. A separate macOS
probe rejected the attempted `RLIMIT_AS` address-space cap; no memory ceiling is
claimed. Linux execution has not been validated.

Validation: all 236 repository tests passed with native checks enabled, including
seven native/output checks. The three added Python files pass Ruff lint and
format checks. Repository-wide lint still reports pre-existing import ordering
in `examples/elitelist_papers/paper1.ipynb` and the four
`research/{alphaevolve,elitesearch,lineagesearch,shinkaevolve}/generation.py`
modules. Repository-wide formatting also reports existing issues in that
notebook, `rsikit/common/__init__.py`, and
`docs/superpowers/plans/2026-09-19-persistent-sandbox-python.md`. These files were
not changed by this experiment.

The prototype is 275 lines including its worker entry point, compared with 528
lines in the existing Docker launcher plus service. Both reuse evaluation and
codec code. This is a smaller execution path, not a net repository deletion:
the alternate Docker backend, public `run_program`, and poker evaluator still
depend on the existing infrastructure.

The native prototype, its tests and its benchmark driver were removed when Docker
became the sole backend. The raw measurements remain as a historical record;
these source hashes describe the experiment before consolidation, not current code.

Sources: [raw measurements](srt-benchmark-2026-09-28.json),
[upstream sandbox-runtime](https://github.com/anthropics/sandbox-runtime).
