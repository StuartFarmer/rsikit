# Upstream Ocean evaluation

The runner uses PufferLib's **existing Python factories and C batch bindings**.
The installer applies `scripts/ocean-episodes.patch` to the pinned upstream source.
There is no duplicated simulator or runtime compiler.
Pinned upstream: `3b5c6046bb8b46685d62d151720025507e3418c2` (Python-facing 3.0).
The search and benchmark CLIs support `g2048` and `breakout`.

## Run

Docker installs and builds the upstream bindings automatically:

```bash
./scripts/run examples.benchmark_ocean correctness \
  --env g2048 --seeds 2 --batch-size 4 --max-steps 32 \
  --output runs/ocean-upstream-check
```

The upstream Linux build downloads x86-64 native libraries. The Ocean launcher
therefore selects `linux/amd64`; on Apple Silicon Docker emulates that architecture.
Native macOS builds are also supported. Do not compare emulated Docker timings
with native macOS timings.

For local development:

```bash
uv sync --extra ocean --extra dev
.venv/bin/python scripts/install_ocean.py
.venv/bin/python -m unittest tests.test_ocean_upstream tests.test_ocean_evaluator
```

The installer fetches the pinned source into the Python environment and invokes
upstream `setup.py build_g2048 --inplace` / `build_breakout --inplace`. It adds the
checkout to that environment's Python path. Build requirements include Git, a C
compiler, and internet access for upstream Raylib/Box2D downloads. Upstream
source is patched only for the episode fixes described below. The old PyPI
distribution pins older NumPy/Gymnasium; we build
the selected bindings against our installed libraries instead of downgrading the
project. This supports the selected Ocean environments, not the full upstream
trainer or every legacy Gym integration. Runtime evaluation needs no downloads.

## Evaluation protocol

Protocol ID: `ocean-upstream-episodic-v2` (compatibility patch `episodes-v1`).

- Each `--seeds` entry requests **one independently seeded episode**.
- `--batch-size B` is the maximum number of those episodes stepped together.
  It does not multiply the episode count. The final batch may be smaller.
- Each game stops at its native terminal signal or `--max-steps H`, whichever
  comes first. Finished lanes freeze; no replacement episodes are started.
- Total transitions are **at most `number_of_seeds × H`**. Actual counts are saved.
- A panel with S seeds has a wall-time allowance of `S × timeout` seconds,
  including process startup. A candidate runs in its own isolated process.
- `--env-kwargs` on the legacy example CLI forwards environment options.
  The unified CLI uses environment flags or YAML. Width, seeds, log interval,
  and buffers remain evaluator-owned.

**10 seeds, batch size 32, max steps 2,000 means 10 episodes and at most 20,000
transitions per policy.** A 50-population, 10-generation search requests 5,000
search episodes (at most 10 million transitions), before repairs or held-out
panels. The previous fixed-horizon protocol accidentally multiplied this budget
by 32 and continued through autoresets; those scores and timings are historical.

The patch keeps upstream physics, observations, reward functions and native
vector stepping. A seed list passed to `reset` selects episodic operation;
integer reset seeds retain the upstream training/autoreset mode. It fixes 2048's
cleared done flag, stops finished lanes, and exposes current per-lane metrics.
Breakout also stops its frameskip loop when the episode ends. Native 2048 ends
include its own dynamic tick limit, starting at 1,000; this limit is still reported
as termination by upstream. The evaluator's additional cap is reported as truncation.

Episodic mode uses per-game `rand_r` state instead of shared process-global
`rand()`, so a seed identifies the same game regardless of batch width, order or
other lanes finishing. This changes trajectories from the previous protocol.
Reproducibility requires the same native build/platform: libc RNG implementations
can differ. Manifests include the upstream revision, patch version and binary hash.

Generated policies must be deterministic and stateless, deciding independently
for each row. Each batch loads a fresh policy and resets it with policy seed 0;
environment seeds control the games. Batch width is fixed within each batch,
including frozen finished rows, and may differ between calls to the evaluator.

## Scores and observations

| Key | Meaning at episode end or evaluator cap |
| --- | --- |
| `merge_score` (2048 default) | Accumulated merge score |
| `score` | 2048 maximum tile; Breakout game score |
| `episode_return`, `perf` | Current upstream episode metric |
| `return` (Breakout default) | Sum of rewards from this episode only |

Capped games retain their earned scores. Missing score keys fail evaluation.
Fitness is the arithmetic mean of the requested per-seed episode scores. Each row
records `steps`, `ending` (`terminated` or `truncated`), `episodes=1`, and metrics.

2048 now receives upstream `uint8 (B,289)` observations, not `(B,16)`:
columns 0:16 are encoded magnitudes, 16:32 are empty flags, 32:288 are 16 one-hot
exponent channels per cell, and 288 is the snake-pattern flag. Actions remain
0=up, 1=down, 2=left, 3=right. The prompts and bundled baselines use this interface;
old saved 16-column policies need adapting.

Breakout receives upstream `float32 (B,118)` observations. Actions are 0=noop,
1=left, 2=right. Default upstream frameskip is 4; use
`--env-kwargs '{"frameskip": 1}'` to select a different benchmark explicitly.
Neither runner exports videos.

## Search and benchmarking

The existing EliteSearch command still works with explicit workload settings:

```bash
./scripts/run examples.ocean_search \
  --env g2048 --arm elite --model "${MODEL:?Set MODEL}" \
  --population 50 --generations 10 --elites 10 --max-repairs 2 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --batch-size 32 --max-steps 2000 --score-key merge_score \
  --workers 4 --generation-concurrency 25 \
  --spend-cap 100 --input-price 0.01 --output-price 0.01 \
  --output runs/ocean-upstream-search
```

The price flags are user-supplied accounting ceilings, not model price claims.
Finalist selection still uses episode seeds 1000–1127 and the winner's test uses
2000–2511. They use the same B/H/scoring settings as search and add substantial
work. Every manifest records the protocol, upstream commit, binary hash,
environment kwargs, scoring key, seeds, and workload.

`reference` mode starts one process per episode seed. `batch` mode groups the
requested episodes into batches of up to B within one candidate process. `summary`
is an alias for `batch`. All modes evaluate the same seeded episodes.

The correctness command checks repeated batches, seed-order independence, and
process-boundary parity and equality across batch widths. The capacity `--matrix`
compares modes
and 1/2/4 workers at that same width. Throughput includes startup and persistence.
Run fresh benchmarks before making speed claims. Historical measurements and
the old design remain in [OCEAN_BENCHMARK_LEGACY.md](OCEAN_BENCHMARK_LEGACY.md).

## Episodic fix verification (2026-10-03)

- Rebuilt the patched bindings on native macOS ARM64 and Linux x86-64 Docker.
- Six native regressions pass: terminal flags, frozen final state, step caps,
  retained capped scores, independent seeds across widths/order, and Breakout endings.
- All **280 tests pass** in Docker as UID 501, with a read-only workspace and
  networking disabled. Log: `/private/tmp/ocean-episode-docker-tests.log`.
- Changed Python files pass Ruff checks and formatting; `git diff --check` passes.
- No new throughput benchmark was run on the shared host.

## Historical fixed-horizon speed check (2026-10-03)

**Everything below used `ocean-upstream-fixed-horizon-v1`. These are preserved
historical measurements, not performance claims or worker recommendations for
the corrected episodic protocol. No replacement timing run has been made while
the host is busy with other work.**

The then-current upstream path retained a large batching advantage on `g2048`:
**25.73× in one complete eight-policy comparison on one worker**. This checks
serial versus vector evaluation under the new protocol; it is not a direct
timing comparison between the old custom binding and the upstream binding.

Both paths executed **64,000 game transitions per policy**, or **512,000 per
eight-policy pass**, with the same reviewed NumPy policy implementations:

| Path | Configuration per policy | Eight-policy time | Transitions/s |
| --- | --- | ---: | ---: |
| Serial | B=1, 32 seeds, 2,000 steps; fresh process per seed | 431.060 s | 1,188 |
| Vector | B=32, one seed, 2,000 steps; one process | 16.751 s | 30,565 |

Per-policy service times include startup, validation, evaluation, and persistence:

| Policy | Serial seconds | Vector seconds | Speedup |
| --- | ---: | ---: | ---: |
| legal-priority | 42.587 | 1.504 | 28.32× |
| merge-greedy | 42.654 | 1.539 | 27.71× |
| empty-cells | 42.432 | 1.573 | 26.97× |
| corner | 42.528 | 1.617 | 26.29× |
| smooth-board | 42.519 | 1.579 | 26.92× |
| corner-smooth | 42.520 | 1.584 | 26.84× |
| merge-two-ply | 87.845 | 3.700 | 23.74× |
| board-two-ply | 87.972 | 3.651 | 24.10× |

This was native macOS ARM64, Python 3.14.2, NumPy 2.5.3, with numerical-library
thread limits set to one. Hardware allocation was not OS-enforced. All 16 measured
panels succeeded and their transition counts were checked. The harness's first
and warm-up panels were excluded, but each measured candidate still paid for
fresh child-process startup. The vector pass ran first.

**Only one full comparison completed.** Three repetitions were planned, but the
second serial pass developed substantial timing drift: previously ~42.5-second
panels took 86–184 seconds. The host wall clock also advanced about 19.7 minutes
more than `perf_counter`, consistent with sleep or a clock discontinuity. The six
completed repeat panels reproduced their original full result rows exactly.
The run was stopped during its seventh panel; the third repetition was not
started. These partial timings are retained in the evidence, not included in a
claimed three-run median. The 25.73× result is a local observation, not a stable
repeated estimate or a sustained-capacity qualification.

A separate, smaller process-lifetime check kept **B=32, seeds=[0,1], H=2000**
identical in both modes, using `legal-priority` and `board-two-ply`. `reference`
took **17.162 s** and `batch` took **11.617 s**, a **1.48×** speedup in one pass.
All scores and full result rows matched exactly. This isolates process lifetime
at a fixed batch width; it does not by itself reproduce the serial-to-vector
speedup, and its timings were collected after the host slowdown.

Changing batch width changes upstream random streams, so the main comparison
matches transition counts, not trajectories or scores. The historical 21.23×
benchmark also stopped games at episode endings, whereas this benchmark always
runs the full horizon with autoresets. Its absolute times and search-duration
estimates cannot be carried forward. These results apply to the reviewed policy
mix; arbitrary generated policies and other Ocean games need their own timings.
No LLM calls or full optimizer searches were included.

Compact evidence, hashes, manifests, and partial-repeat timings:
[ocean-upstream-speedup-2026-10-03.json](ocean-upstream-speedup-2026-10-03.json).
Full local artifacts and driver scripts are under
`runs/ocean-upstream-speedup-2026-10-03/`.

The old comparison cannot be reproduced with the corrected evaluator: the same
flags now request a different workload. Preserve the artifact manifests when
comparing historical results; new benchmarks must use equal episode seed lists.

## Historical policy evaluation throughput by worker count (2026-10-03)

**Sixteen workers were fastest among the tested configurations**, reaching a
median **1.435 completed policy evaluations/second**. In the follow-up sweep,
this was **12.85% faster than eight workers**. Twelve workers offered essentially
the same median throughput as eight. This is the best tested setting for this
workload and finite queue, not a proven optimum for every search workload.

Every candidate received exactly **64,000 transitions**: `g2048`, batch size 32,
seed `[0]`, horizon 2,000, and `merge_score` scoring. Each trial evaluated the same
eight reviewed policies twice, with caching disabled: **16 jobs and 1,024,000
transitions**. All jobs were ready at the start, and the evaluator's worker limit
controlled concurrency. Batch width, seeds, horizon, and policy order stayed
fixed. This sweep changes scheduling, not the scoring protocol.

Three trials per worker count used different configuration orders. Timings
include candidate process startup, execution, validation, result persistence,
and draining all 16 jobs; two initialization panels per trial are excluded.

| Initial sweep: workers | Median evaluations/s | Observed range | Median time for 16 jobs |
| ---: | ---: | ---: | ---: |
| 1 | 0.344 | 0.170–0.376 | 46.466 s |
| 2 | 0.589 | 0.336–0.630 | 27.172 s |
| 4 | 0.873 | 0.569–0.889 | 18.322 s |
| 8 | 1.108 | 0.797–1.146 | 14.436 s |

Because eight workers led that sweep, a second sweep compared higher counts
against a fresh eight-worker control:

| Follow-up: workers | Median evaluations/s | Observed range | Median time for 16 jobs |
| ---: | ---: | ---: | ---: |
| 8 | 1.272 | 1.028–1.319 | 12.583 s |
| 12 | 1.275 | 1.270–1.375 | 12.545 s |
| 16 | **1.435** | **1.406–1.516** | **11.150 s** |

All **336 measured evaluations passed**, covering **21,504,000 transitions**.
Full result rows matched exactly across both sweeps, all worker counts, and all
repetitions. This used the same pinned upstream binary on native macOS ARM64,
with 12 reported logical CPUs and numerical-library thread limits of one.

The host was not reserved exclusively for benchmarking, and the initial sweep
showed substantial timing variation. All trials are retained; the reported
ranges are observed minima/maxima, not confidence intervals. The largest measured
wall-clock versus monotonic-clock gap was 0.053 seconds. Use the second sweep's
eight-worker control when comparing 8/12/16, rather than mixing their ratios with
the earlier, slower baseline. These finite 16-job trials do not establish
sustained capacity or the best concurrency for a larger queue; counts above 16
were not tested.

Sixteen workers led that historical workload; this is not a recommendation for
the corrected evaluator. The absolute rates above apply to **one batch seed per policy**.
The example search configuration uses ten seeds and therefore performs ten times
as many transitions per policy; its evaluation rate needs a separate measurement.
No LLM generation, repairs, held-out evaluation, or videos were included here.

Evidence: [ocean-upstream-workers-2026-10-03.json](ocean-upstream-workers-2026-10-03.json).
The [driver](ocean_worker_scaling.py) now follows the corrected episode budget,
checks every result, and saves raw events; it does not reproduce these old timings.
Run from the repository root, using fresh output directories:

```bash
PYTHONPATH=. .venv/bin/python docs/ocean_worker_scaling.py runs/worker-sweep 1 2 4 8
PYTHONPATH=. .venv/bin/python docs/ocean_worker_scaling.py runs/higher-workers 8 12 16
```

## Historical fixed-horizon verification (2026-10-03)

- Built the unmodified upstream bindings on native macOS ARM64 and in Linux
  x86-64 Docker, using Python 3.14, NumPy 2.5.3, and Gymnasium 1.3.0.
- **258 tests passed** in the read-only Docker container as UID 501, against the
  current workspace, with networking disabled. Log:
  `/private/tmp/ocean-upstream-docker-final.log`.
- **24 Ocean/launcher tests passed** locally. They cover fixed-width workload
  accounting, direct upstream reward/log agreement, invalid actions, repeatability,
  process parity, failure recovery, cancellation, search wiring, and arrival replay.
- Both environment correctness commands passed. A scripted-provider search
  completed real evaluation, 128 validation seeds, and 512 test seeds at B=2/H=4.
  This was a plumbing check, not a policy-performance result; no paid model calls.
- Changed Python files pass Ruff lint and formatting; `git diff --check` passes.

The host-wide run had four existing environment-dependent errors:
`ApplicationEpisodeTests.test_replay_records_best_completed_policy_without_changing_original_scores`,
`ApplicationEpisodeTests.test_run_records_real_video_artifact`,
`ApplicationEpisodeTests.test_scientific_libraries_in_episode_process`, and
`ScientificLibrariesTests.test_numerical_operations`. The first two concern video
workers; the latter two lack the host's `control` package. All four passed in Docker.
The initial Docker verification mistakenly used root and triggered three non-root
assertions; the corrected full run above passed.

Repository-wide Ruff still reports existing import-order errors in
`research/{alphaevolve,elitesearch,lineagesearch,shinkaevolve}/generation.py`, plus
formatting in `docs/RUNS.md` and
`docs/superpowers/plans/2026-09-19-persistent-sandbox-python.md`. Those unrelated
files were not changed.
