# Ocean benchmark: make LLM generation the bottleneck

Status: implemented, 2026-10-02. The native 2048 adapter, evaluator,
capacity/replay driver and budgeted search example are available. The sizing
table below remains hypothetical; short implementation checks do not establish
that evaluation keeps up with a particular live model.

## Run it

From a repository checkout with project dependencies installed and a C compiler
(`cc`, or `CC`), run this small correctness check:

```sh
python -m examples.benchmark_ocean correctness \
  --seeds 2 --max-steps 32 --output runs/ocean-check
```

No PufferLib pip install, GPU, raylib, download or model key is needed. The pinned
C source and MIT license are included; a content-hashed headless shared library
is compiled into the temporary directory on first use. Output directories must
be new. For the specified four-CPU/eight-GiB container allocation, replace
`python -m` with `./scripts/run` in these commands. Local runs record the requested
budget but do not enforce CPU affinity or a memory limit.

Run a short check of complete 32-seed panels, then synthetic arrivals:

```sh
python -m examples.benchmark_ocean capacity --mode batch --workers 4 \
  --duration 1 --panels 8 --repeats 1 --output runs/ocean-capacity-smoke
python -m examples.benchmark_ocean replay --rate 1 --duration 10 --repeats 1 \
  --workers 4 --output runs/ocean-replay-smoke
```

These deliberately short runs leave `pilot_pass=false`. For the full offline
protocol, run default correctness, then `capacity --matrix`; the matrix executes
all 15 mode/worker/batch configurations three times and can take substantially
longer than the smoke check. For a single selected configuration, omit `--matrix`:

```sh
python -m examples.benchmark_ocean correctness --output runs/ocean-correctness
python -m examples.benchmark_ocean capacity --mode batch --workers 4 \
  --correctness-report runs/ocean-correctness/report.json \
  --output runs/ocean-capacity
python -m examples.benchmark_ocean replay --rate 1 --workers 4 \
  --correctness-report runs/ocean-correctness/report.json \
  --capacity-report runs/ocean-capacity/report.json --output runs/ocean-replay
```

Synthetic replay reports capacity/queue evidence but cannot pass the real-LLM
gate. Run `replay --trace PATH` for recorded arrivals; paths in that trace are
resolved relative to its file. The same `--trace` can supply the exact policy mix
to correctness and capacity. Evidence reports must match source, workload,
platform and mix. A real trace must span at least 300 seconds after any `--speed`
acceleration; `--duration` creates synthetic traces and does not tile real traces.
Use `--speed 1.25` for the separate headroom replay.

Use repeated `--policy path.py` arguments instead of the eight reviewed baseline
programs when testing a saved corpus. Default policies are hand-authored examples,
including bounded lookahead; they are explicitly not labeled LLM-generated.
`--diagnostics` adds per-phase timing to a separate run, which cannot count as
uninstrumented pilot evidence. `--help` lists all parameters.

Results include `manifest.json`, `report.json`, policy sources, per-job JSON and
`evaluations.jsonl`. Replay adds `admissions.json`, `queue.json` and `queue.svg`
per repetition. Worker RSS is a high-water measurement, not a simultaneous
process-tree memory total.

### Live search

For the standalone 50-population × 10-generation × 10-seed EliteTable run,
see the [ready-to-run command](OCEAN_CONTROLLER_OPTIONS.md#standalone-elitetable-run-with-no-outer-optimization).
Only `g2048` is implemented. `--population`, `--generations`, `--elites`,
`--max-repairs` and `--seeds` configure the existing search; call/token reservation
defaults scale to the requested workload. No outer controller optimization runs.

The live example requires an explicit model and budget. Set `MODEL`,
`INPUT_PRICE` and `OUTPUT_PRICE` to the chosen model and conservative upper USD
prices per million tokens, and provide `OPENROUTER_API_KEY`:

```sh
./scripts/run examples.ocean_search --output runs/ocean-elite-0 \
  --model "$MODEL" --arm elite --search-seed 0 --spend-cap 5 \
  --input-price "$INPUT_PRICE" --output-price "$OUTPUT_PRICE"
```

The default is 20 proposals × 5 generations, at most two repairs each, four model
calls in flight and four evaluation workers. It reserves worst-case per-call
input/output token cost **before** sending, counts failed calls and repairs, and
stops at its call/token/spend reservation limit. Its UTF-8 input bound assumes a
byte-tokenized text model; provider billing is not independently enforced by this
local ledger. Responses retain actual usage/cost when supplied. No calls were
made to a paid provider during implementation checks.

Repeat with `--arm independent` and paired `--search-seed` values 0 through 4
for the comparison. The example performs 128-seed finalist selection and a
512-seed frozen test, and writes `winner.py`, `selection.json`, `summary.json`,
`generation_tails.json`, `arrival_trace.json`, and the existing Run database.
`arrival_trace.json` is accepted directly by the replay command. A full recursive
controller experiment remains the follow-up described below.

Run the focused native/process/CLI tests with:

```sh
python -m unittest tests.test_ocean_native tests.test_ocean_evaluator \
  tests.test_ocean_benchmark tests.test_ocean_search
```

### Implementation measurements

One local ARM64 macOS pass, one candidate worker, eight reviewed policies,
32 seeds each, up to 2,000 decisions per episode, with matching native binaries:

| Path | Eight complete panels | Median panel service | Valid panels/s |
| --- | ---: | ---: | ---: |
| Reference, fresh process per episode | 132.99 s | 12.34 s | 0.060 |
| Batched, fresh process per panel | 6.26 s | 0.47 s | 1.277 |

All 256 per-seed results matched. The aggregate difference is **21.2×** including
worker startup and result persistence. Policy complexity matters: the slowest
batched panel took 2.45 seconds. These are one-pass implementation checks, not the
repeated capacity pilot or proof of generation-bound live search. Resource limits
were not enforced locally, and some verification processes overlapped the reference
run. Linux forkserver results may differ from macOS spawn.
[Raw results and per-seed evidence](ocean-benchmark-2026-10-02.json).

A subsequent worker-scaling check repeated each configuration three times, with
16 full candidate panels per repetition and the same eight-policy mix:

| Candidate workers | Median valid panels/s | Throughput relative to one batched worker |
| --- | ---: | ---: |
| 1 | 1.267 | 1.00× |
| 2 | 2.207 | 1.74× |
| 4 | 3.366 | 2.66× |

Each panel still contains 32 games. Configurations ran sequentially, and all
per-seed results matched the original evidence. These short runs include startup,
persistence and final drain; they do not establish the full steady-state or
real-LLM pilot. [Worker-scaling measurements](ocean-worker-scaling-2026-10-02.json).

The CLI was also exercised across all 15 configurations with a short workload,
and live-search wiring completed the full 32/128/512 seed split using a scripted
provider. No paid LLM calls were made and no real-model pilot pass is claimed.

Validation: all 20 Ocean tests and Ruff checks pass. The full local suite ran
257 tests with four errors in existing tests outside this implementation:

- `ApplicationEpisodeTests.test_replay_records_best_completed_policy_without_changing_original_scores`
  and `test_run_records_real_video_artifact`: video episode workers exited without a result.
- `ApplicationEpisodeTests.test_scientific_libraries_in_episode_process` and
  `ScientificLibrariesTests.test_numerical_operations`: optional `control` package missing.

No shared evaluator, optimizer, dependency configuration or existing test was
changed to suppress those errors.

## Goal

Evaluate generated policies faster than the LLM produces them, at useful seed
coverage. Evaluation should keep up with arrivals without building a backlog or
holding up the next search generation. Once that is true, stop optimizing speed
and use spare capacity for more reliable policy comparisons.

Start with Ocean **2048** on CPU and reuse RSIKit's **EliteSearch**. First measure
the evaluator without paid model calls; then confirm the result in a live search.
Repeat on Breakout after the first experiment passes.

## The example we are testing

A candidate plays 32 independently seeded games, each capped at 2,000 decisions:
at most 64,000 individual environment transitions.

| Assumed policy + environment throughput | Rollout time for 64,000 steps | Serial rollout capacity |
| --- | ---: | ---: |
| 10,000 steps/s | 6.40 s | 0.156 candidates/s |
| 100,000 steps/s | 0.64 s | 1.563 candidates/s |
| 1,000,000 steps/s | 0.064 s | 15.625 candidates/s |

These figures exclude startup, loading, scoring, serialization and persistence.
Real episodes can end early. Measure full candidate latency and actual step
counts; do not present this table as an Ocean performance result.

Let `lambda` be the actual generation rate in evaluation jobs/second, including
repaired candidates submitted for another evaluation. Let `mu` be measured
aggregate evaluator capacity on the allocated hardware. The target is:

```text
mu >= 1.25 * lambda
```

The 25% headroom is a pilot choice for variable arrivals and evaluation lengths.
For initial sizing only, `workers ≈ ceil(1.25 * lambda * mean_service_seconds)`.
This assumes service time holds under concurrency; the benchmark must verify it.

For example, at 10 arrivals/s and 0.64 s **complete** evaluation service time,
seven workers barely cover average demand; eight provide 25% nominal headroom.
At 0.064 s, one worker would suffice. Rollout-only times cannot be substituted
for complete service times without measuring the missing overhead.

## Freeze the workload

| Setting | Initial value |
| --- | --- |
| Upstream | PufferLib `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2` |
| Environment | `g2048` |
| Curriculum | `[env] scaffolding_ratio=0` |
| Search seeds | Integers 0–31, explicit per-environment RNG initialization |
| Episode cap | 2,000 decisions, or earlier native ending |
| Primary fitness | Mean conventional tile merge score across all 32 seeds |
| Additional metrics | Maximum tile, shaped return, decisions, ending reason |
| Policy | Stateless Python/NumPy program; LLM writes code, never acts per step |
| Hardware budget | One machine/container, 4 allocated CPU cores, 8 GiB RAM |
| Threading | One numerical-library thread per worker; no nested oversubscription |
| Candidate deadline | 60 s for the entire panel, including initialization and result preparation |
| Rendering | Off during timed runs |
| Evaluation cache | Disabled for throughput measurements |

Pin the platform, C library, compiler flags, Python/NumPy versions and native
library hash. `rand_r` streams need not match across platforms. Record actual
hardware and CPU allocation; local results do not automatically transfer to a
different host. Build once; report build time separately from candidate timing.

2048 has its own adaptive episode cap, initially 1,000 ticks. Our external cap
does not replace that rule. Preserve the native rule in every comparison and
clear lifetime state before each seed. This is a benchmark of this exact Ocean
configuration, not unrestricted conventional 2048.

Use eight frozen policy programs for the evaluator benchmark: a cheap legal-move
heuristic, a one-step board heuristic, a bounded-lookahead policy, and five
representative generated candidates. Prefer saved candidates when available;
otherwise author reviewed examples before starting the timed comparison. Record
their hashes and preserve the same mix in every mode. Report results per policy
as well as for the mixture, so a cheap policy cannot hide slow realistic ones.

## Smallest implementation

Reuse source storage, generation, edits/remixes, elite ranking and checkpoints.
The experiment-owned evaluation callback returns
`{policy_id: Measurement(scores={seed: merge_score})}`. Do not rewrite EliteSearch
or the shared evaluator to conduct this experiment.

The evaluator uses a small native CPU binding to Ocean's
`puf_init/reset/step/close` functions, compact results and one fresh candidate
worker per seed panel. Initialize native environment objects inside the worker;
do not attempt to pickle live C pointers. Current upstream trainer batching is
CUDA-coupled and is not a drop-in CPU Python vector evaluator.

The research-only batch convention for `Solution.act` is observation shape
`(B, 16)`, action shape `(B,)`, with matching declared spaces. Board entries are
tile exponents; actions are 0 up, 1 down, 2 left, 3 right. This convention is new
adapter behavior, not a change to the shared scalar evaluator. Require independent rows;
no cross-board statistics or shared evolving state. Start with deterministic
policies. Any stochastic policy needs independent per-seed random streams.

Use the same policy source for scalar and batch timing: scalar evaluation calls
the batch function with one row. Verify row-permutation and batch-size invariance.
Maintain a fixed mapping from active rows to seed IDs. Remove completed cases;
do not continue their auto-reset games or count inactive padding as useful steps.

Return score, length, ending reason and timing per seed. Save full trajectories
only for the correctness subset and chosen finalists. Retain source, failures,
scores and checkpoint persistence in the timed application path.

The initial timing pilot uses reviewed policies under RSIKit's documented
cooperative execution model. Generated code currently shares a process with
scoring state; this is not a protected benchmark against adversarial programs.
Before recursive self-modification, put scoring, hidden tests and budget
enforcement outside candidate control and remeasure that configuration.

## Correctness gate

Before comparing speed, run the same eight policies on the same 32 seeds through
scalar and batched modes. Require identical per-seed actions, scores, lengths
and ending reasons for deterministic policies on the pinned platform. Verify
repetition and reordered batches produce the same seed results.

Handle the upstream details explicitly:

- Disable scaffolding and initialize RNG from the requested seed, not the slot
  index. Freshen the entire episode state, including `lifetime_max_tile`.
- `merge_score` is conventional game score. Upstream's log field `score` means
  maximum tile; `episode_return` is shaped reward.
- At native termination, read the completed log delta exactly once: the live
  board may already be the next episode's reset board.
- At external truncation, capture the live merge score explicitly. Upstream
  does not emit a completed-game log for our external cap.
- Distinguish an in-range but ineffective move from an out-of-range action.
  Invalid move penalties and native timeouts remain part of the pinned task.
- Reject malformed/nonfinite actions; crashes and deadlines produce failures,
  never a partially averaged score. Every accepted candidate has all 32 seeds.

Keep one small runnable check covering these invariants with the adapter.
Run failure probes separately from successful-policy throughput so rapid
rejection cannot inflate the claimed evaluation capacity.

## Bench A: evaluator capacity, no live LLM

Compare the same workload in three modes:

| Mode | Execution | Purpose |
| --- | --- | --- |
| A: reference | Scalar Gym adapter, current full-history evaluator and fresh episode processes | Establish the existing execution pattern on Ocean |
| B: summaries | Scalar steps, one candidate-panel worker, compact results | Measure amortized startup and reduced recording costs |
| C: batches | Batched actions/steps, one candidate-panel worker, compact results | Measure the extra benefit of batching |

The Gym adapter in A is also new work. All modes use the same pinned simulator,
policies, seed panel and ending rules. B changes both worker lifetime and result
retention, so A→B is their combined effect, not a pure logging ablation.

1. Record cold initialization and first-panel latency for each mode.
2. Run one untimed warmup panel. Each subsequent panel still starts a fresh
   candidate worker in B/C; warmup does not imply persistent policy workers.
3. Measure candidate concurrency 1, 2 and 4. In C measure environment batch widths
   1, 8 and 32 within the fixed 32-seed panel. Keep the 4-core budget fixed.
4. Keep the evaluator supplied with jobs for at least 30 seconds and 100
   completed panels per configuration, whichever takes longer. Stop arrivals,
   drain admitted jobs, and include drain time in aggregate throughput.
5. Repeat each configuration three times, alternating mode order to reduce
   background-load bias. Reevaluate identical programs; do not use cached scores.

Report the bottleneck directly: environment time, policy time, startup/reset,
validation/copying, result transfer and persistence. Use lightly instrumented
separate diagnostic runs for per-step attribution; exclude profiler overhead
from headline timing. Do not sum overlapping worker wall times as elapsed time.

Choose the smallest worker/batch configuration that meets the capacity target.
If representative policy computation dominates, measure a cheaper/batched policy
implementation before considering generated C or CUDA. Compilation would then
be charged to every new candidate, not hidden in warmup.

## Bench B: can evaluation keep up with generation?

First obtain an arrival trace from a representative existing run: evaluation
submission timestamps, candidate identities/sources, repair attempts, and model
call start/end timestamps. Estimate generation capacity from periods when
generation is active and not waiting for evaluations. Whole-run throughput from
an evaluation-bound run would underestimate the target and make passing trivial.

If no usable trace exists, report a synthetic rate sweep first (0.1, 1 and 10
jobs/s), and measure real arrivals in Bench C. Synthetic rates are test inputs,
not claims about the configured model. For the real target, freeze model,
generation concurrency and provider limits in the run manifest.

Replay the same candidate workload and arrival timestamps into A and the selected
C configuration, with generation replaced by the saved trace. This avoids paying
for repeated LLM calls and preserves burstiness. Preserve source/latency pairing
where available. Repeated trace cycles must execute again with caches disabled;
use fresh request IDs. Record producer backpressure rather than silently dropping
jobs if an overload run reaches its queue limit.

Run three traces of at least five minutes each. Record queue depth once per
second, every submission/start/finish event, and the drain after the final arrival.
Also test 1.25× the measured arrival rate to check headroom. Finite replay must
finish with every job accounted for, including failures.

The declared pilot passes when all of these hold:

- Score parity and correctness checks pass.
- Saturated aggregate capacity is at least `1.25 * lambda` on the same workload
  mix and CPU budget. Report successful panels/s separately from failure handling.
- At the real arrival rate, mean queued jobs in the last minute is no more than
  one above the first post-warmup minute in each replay. Plot the full queue trace
  to expose bursts and growth hidden by that simple criterion.
- p95 queue wait is no greater than `max(0.1 s, 10% of median LLM-call latency)`.
  For synthetic-only runs, report wait without claiming this real-LLM gate passed.
- No drops, increasing backpressure or worsening failure rate explain the result.

These are predeclared engineering thresholds, not a statistical proof of queue
stability. A longer soak is warranted if arrivals or expensive policy tails vary
substantially. Report failure honestly; do not reduce seed coverage to make the
same benchmark appear to pass.

## Bench C: live program search

Use the chosen evaluator behind EliteSearch's callback. Run an initial
100-proposal search: 20 candidates × 5 generations, elite size 5, 32 search seeds,
and at most two repairs per proposal. Freeze model, generation concurrency,
token ceilings and an explicit spend cap before launch. Count every repair and
failed proposal. Stop at the first budget limit and retain the best valid policy.

Generation and evaluation should overlap as candidates arrive. EliteSearch
still ranks at a generation boundary: a stable average queue alone does not
prove evaluation stopped delaying the loop. For each generation, record the
time from its last generation/repair LLM response to its last persisted evaluation.
Target a median residual evaluation tail below 10% of the generation's elapsed
LLM-call window; also report the maximum. Model calls triggered by repairs belong
in that window. Include ranking/checkpoint time separately.

If the queue and tail gates pass, the operational objective is met: LLM generation
sets the pace for this task, policy mix and resource allocation. Record how much
seed coverage fits before evaluation becomes limiting again; test 128 seeds as
a separate sensitivity run, never as the same 32-seed benchmark.

To test program improvement, compare independent best-of-100 generation with
fixed EliteSearch across five paired runs each. Use identical model/resource
ceilings; report quality against both proposal count and actual spend. Independent
proposals receive no elite descriptions, source or scores—`new_fraction=1` alone
does not remove that context from the existing prompt.

Select among each run's top five policies on seeds 1000–1127 (128 validation
cases), then evaluate the frozen winner once on seeds 2000–2511 (512 test cases).
Never feed test results into search or repairs. Report per-run mean merge score
and paired differences with uncertainty across search runs. Use the same reviewed
heuristic as a predeclared fallback if a run yields no valid policy. Five paired
runs are a pilot; they may not resolve small effects.

## Results to save

Save one manifest, an event log and a result table alongside policy sources and
per-seed scores under the run output directory. Required timing events are:

```text
candidate/job ID, policy hash, generation, attempt/repair
LLM start, LLM finish, evaluation submitted, worker acquired
worker/reset ready, rollout finished, result received, score persisted
actual decisions, successful seeds, native endings, external truncations, failure
```

Use a common parent monotonic clock for queue and completion timestamps. Worker
subphase durations use their own monotonic clock; do not subtract clocks from
different machines. Service time spans worker acquisition through score persistence;
response time spans submission through persistence and includes queue wait.

| Mode / batch / workers | Valid panels/s | Actual steps/s | Median / p95 service | p95 queue wait | Final queue / drain | Peak RAM | Score parity |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| A: reference | measured | measured | measured | measured | measured | measured | pass/fail |
| B: summaries | measured | measured | measured | measured | measured | measured | pass/fail |
| C: batches | measured | measured | measured | measured | measured | measured | pass/fail |

The conclusion should state: **“At X generated jobs/s, Y seed games per candidate,
and Z CPU cores, evaluation sustained M valid panels/s, p95 queue wait W, and
residual generation tail T.”** Report the measured reference speedup as secondary.
This is stronger evidence for our goal than simulator steps/s alone.

## Follow-up: improve the searcher itself

The [Ocean meta-experiment](OCEAN_META_EXPERIMENT.md) specifies the fixed evaluator,
three objectives (performance, output-token efficiency and evaluation efficiency),
outer controller search, held-out protocol and proposed experiment budgets.

After Bench C, permit edits to parent selection, mutation prompts and operator
allocation while freezing the evaluator and budgets. Compare a frozen searcher,
a fixed editor proposing controller revisions, and a recursive arm where promoted
controller code helps propose successors. Evaluate each revision using fresh inner
searches, not inherited winning policies, and count all nested generation costs.
Keep final transfer cases hidden and repeat outer searches independently before
claiming recursive improvement. This is a follow-up, not a prerequisite for making
evaluation fast enough.

## References

- [Current evaluator](../rsikit/evaluation.py), [executor](../rsikit/execution.py),
  [EliteSearch](../research/elitesearch/agent.py), [Measurement](../research/rewards.py).
- [Current execution boundaries](IN_PROCESS_SANDBOX.md).
- [Pinned Ocean API](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferenv.h).
- [Pinned 2048 implementation](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/g2048/g2048.h)
  and [configuration](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/config/g2048.ini).

The preceding research notes in `outputs/puffer-ocean-search.md` contain broader
version comparisons and the source audit. This specification is self-contained;
those local research outputs are not required to interpret the benchmark.
