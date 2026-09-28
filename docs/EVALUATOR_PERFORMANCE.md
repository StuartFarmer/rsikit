# Evaluation throughput — 2026-09-24

This report describes the API and scalar-result protocol at measurement time.
Current execution returns full Episodes; use the [current run API](RUNS.md).
The current benchmark requires both images to support the Episode protocol;
use the historical checkout to reproduce the archived image comparison.

The target is enough evaluation capacity to keep up with generated candidates,
including all requested seeds. Environment computation is only one part of that
cost. This change addresses worker feeding and per-action communication before
considering Cython or Numba.

## What changed

- EliteSearch submits each candidate for evaluation as soon as generation finishes.
  It still breeds from a fixed prior-generation leaderboard and waits for every
  candidate before ranking the next generation.
- Concurrent `Run.evaluate()` calls share one Executor worker limit. Different
  policies overlap; overlapping requests for the same policy retain score reuse.
  `resume()` evaluates the policy snapshot whose locks it acquired.
- Each isolated evaluator process handles one episode, so its policy socket now
  uses synchronous I/O with one absolute deadline covering the complete request
  and response. The host still schedules episodes asynchronously. Cancellation
  and process cleanup remain under the sandbox supervisor.
- The internal evaluator/candidate channel uses bounded length-prefixed frames.
  Numeric arrays carry raw bytes, dtype and shape; other values use the existing
  JSON codec. Array/message bounds, action validation and process separation remain.
  The legacy `run_program` channel retains its existing JSON transport.
- Overlapping progress totals accumulate rather than resetting on each submission.

Injected EliteSearch evaluation callbacks must support concurrent calls. Use a
shared `Run`/`Executor` to bound actual episode workers. Other search algorithms
keep their existing proposal/batch scheduling; they use the faster sandbox
transport when run with the rebuilt image.

## Reproduce

Keep a baseline image before rebuilding, then run alternating warm episodes:

```sh
rtk proxy docker tag rsikit-sandbox:local rsikit-sandbox:evaluator-baseline
rtk proxy docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
rtk proxy .venv/bin/python -m examples.benchmark_evaluator \
  --image rsikit-sandbox:local --compare-image rsikit-sandbox:evaluator-baseline \
  --samples 10 --output /tmp/evaluator-benchmark.json
```

The benchmark records image IDs, source hashes, samples, scores, step counts,
environment time and policy time. Comparisons alternate baseline/optimized order
and exclude per-step timing instrumentation. Each image receives a warmup for
each workload. Score equality and artifact expectations are asserted.

Instrumented residual time includes child creation, reset/close, copying,
validation, communication and artifact handling; it is **not** a direct IPC
measurement. The simple policies deliberately expose evaluator overhead rather
than characterize expensive generated solvers.

The separate scheduling experiment uses real Run persistence and Executor with
a synthetic backend: 12 candidates arrive approximately 5 ms apart, each needing
an 80 ms and an 8 ms episode, with four worker slots. Its baseline drains queued
candidates into batches exactly as the former EliteSearch consumer did. This
measures scheduling behavior, not Docker or model performance.

## Inspect a real search

Set the `rsikit.execution` logger to `DEBUG` to capture `episode_timing` records.
They include `queue_seconds`, `evaluation_seconds`, `active`, and `queued`, as
well as policy ID and seed. Queue timing starts inside Executor; it excludes
waiting for a duplicate policy's Run lock and initial sandbox startup.

Existing timestamped `policy_generated` events provide candidate arrival rate.
Compare arrival rate multiplied by uncached seeds per candidate with sustained
episode completion rate. Growing queue depth indicates insufficient evaluation
capacity. Approximate worker utilization is summed episode service time divided
by worker count and the observation window. These are occupied worker slots,
not measured CPU utilization.

No model calls, new dependencies, environment arithmetic changes, Cython or
Numba were used in this work.

## Results

[Raw measurements](evaluator-benchmark-2026-09-24.json) contain ten samples per
condition. These are warm, single-worker, uninstrumented episode medians with
alternating image order, using the same base image and policies:

| Workload | Steps | Baseline | Optimized | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| One-circle packing | 1 | 24.5 ms | 26.5 ms | 0.92× |
| CartPole | 500 | 128.0 ms | 84.2 ms | 1.52× |
| Blackjack, 24 shoes | 2,485 | 459.6 ms | 309.2 ms | 1.49× |
| Bitcoin | 2,556 | 516.8 ms | 323.4 ms | 1.60× |

Scores matched in every comparison. The one-step workload gained nothing and
its median was about 2 ms slower; this change primarily benefits repeated policy
calls. Samples overlap, and these results are not a tail-latency guarantee.

The synthetic scheduling comparison improved candidate throughput from 32.0 to
40.1 per second (25%), worker occupancy from 72% to 90%, and mean queue wait from
141 ms to 80 ms. This gain is separate from transport; do not multiply the two
ratios into a claimed end-to-end search speedup.

The instrumented optimized Bitcoin episode spent about 14.3 ms in environment
steps and 4.7 ms in policy actions out of 350.4 ms total. Blackjack spent 26.9 ms
in environment steps and 3.0 ms in policy actions out of 311.4 ms. Substantial
runner/process/communication overhead remains, so compiling environment math
alone still has limited headroom for these simple policies.

Measurements ran on ARM64 macOS with an existing Docker workload observed near
800% CPU. No existing workload was stopped. Alternating comparisons reduce,
but do not eliminate, interference from changing background load.

The tested optimized image is tagged `rsikit-sandbox:evaluator-fast`; the original
image is retained as `rsikit-sandbox:evaluator-baseline-20260924`. The report pins
both image IDs. To repeat this comparison after activation, use the optimized
image with `--compare-image rsikit-sandbox:evaluator-baseline-20260924`.

## Validation

All 195 tests in `tests/` passed against the optimized image, including Docker
isolation, fresh episode state, action validation, per-call and episode timeouts,
cancellation, artifact handling, scientific libraries, Box2D and video. New
checks cover overlapping submissions, duplicate-score reuse, resume locking,
queued cancellation, bounded/fragmented numeric frames, one absolute transport
deadline, socket backpressure and overlapping progress counts.

Changed Python files pass Ruff lint and formatting checks. Repository-wide Ruff
lint still flags import ordering in the independently edited
`examples/elitelist_papers/paper1.ipynb`. Repository-wide formatting reports nine
unrelated targets: `README.md`, the earlier persistent-sandbox plan,
`examples/elitelist_papers/{paper1.ipynb,run.py,test_discrete_actions.py,test_paper1.py,test_tie_break.py}`,
`rsikit/sandbox/docker.py`, and `tests/test_paper_agent.py`. These files were not
reformatted as part of this work.
