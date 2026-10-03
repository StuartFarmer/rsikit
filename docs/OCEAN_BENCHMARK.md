# Upstream Ocean evaluation

The runner uses PufferLib's **existing Python factories and C batch bindings**.
There is no RSIKit C adapter, vendored simulator, or runtime compiler.
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
compiler, and internet access for upstream Raylib/Box2D downloads. No upstream
source is patched. The old PyPI distribution pins older NumPy/Gymnasium; we build
the selected bindings against our installed libraries instead of downgrading the
project. This supports the selected Ocean environments, not the full upstream
trainer or every legacy Gym integration. Runtime evaluation needs no downloads.

## Evaluation protocol

Protocol ID: `ocean-upstream-fixed-horizon-v1`.

- Each `--seeds` entry is an independent **batch reset seed**.
- `--batch-size B` means B games per seed, with fixed batch shape throughout.
- `--max-steps H` means H vector steps per seed. Upstream handles autoresets.
- Every seed starts a newly constructed environment and a newly loaded policy.
- Total game transitions per candidate are `number_of_seeds × B × H`.
- Batch width is part of the benchmark definition: changing it changes the task's
  random streams and workload. Equality across batch widths is not required.
- `--timeout` is the panel's wall-time allowance per batch seed; a panel with S
  seeds has a total limit of `S × timeout` seconds, including process startup.
- `--env-kwargs` accepts a JSON object forwarded to the upstream constructor.
  Batch width, seed, log interval, and buffers are evaluator-owned.

For example, 10 seeds with B=32 and H=2000 are **640,000 game transitions per
candidate**, across 320 initial game instances. Autoresets mean this is not a
fixed count of completed episodes. A 50-population, 10-generation search has
320 million transitions before repairs, validation, or test evaluation.
The old 30,000-episode/10-minute estimate no longer applies.

The upstream 3.0 API uses a base seed and slot-derived initialization seeds,
and these environments can use process-global randomness. Reproducibility is
for a fixed environment/version/platform, batch width, seed, and policy. Separate
candidate processes prevent candidate RNG interference. Do not run simultaneous
in-process rollouts. A seed does not identify an isolated lane independent of its
batch. Generated policies should remain deterministic and stateless per row.

## Scores and observations

`--score-key` selects the benchmark objective:

| Key | Meaning |
| --- | --- |
| `merge_score` (2048 default) | Upstream mean merge score over completed episodes |
| `score`, `episode_return`, `perf` | Corresponding upstream completed-episode average |
| `return` (Breakout default) | Sum of all rollout rewards divided by B, including unfinished episodes |

Logged metrics score zero when no episodes complete. Otherwise a missing score
key fails evaluation. Completed-episode metrics exclude unfinished games at the
horizon; select a sufficiently long horizon and inspect `episodes`, or choose
`return` when unfinished trajectories should contribute. Candidate fitness is
the arithmetic mean of its per-seed scores, giving each batch seed equal weight.

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
Finalist selection still uses batch seeds 1000–1127 and the winner's test uses
2000–2511. They use the same B/H/scoring settings as search and add substantial
work. Every manifest records the protocol, upstream commit, binary hash,
environment kwargs, scoring key, seeds, and workload.

`reference` mode starts one process per batch seed. `batch` mode runs all seeds
of a candidate in one process. `summary` is retained as an alias for `batch`.
All modes use upstream native batching at the same B; these are process-lifetime
comparisons, not scalar-versus-vector simulator comparisons.

The correctness command checks repeated batches, seed-order independence, and
process-boundary parity at a fixed width. The capacity `--matrix` compares modes
and 1/2/4 workers at that same width. Throughput includes startup and persistence.
Run fresh benchmarks before making speed claims. Historical measurements and
the old design remain in [OCEAN_BENCHMARK_LEGACY.md](OCEAN_BENCHMARK_LEGACY.md).

## Verification (2026-10-03)

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
