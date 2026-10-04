# GEPA controller-search experiment

For the current **EliteTable outer optimizer with generated five-generation sub-evolvers**,
use [the EliteTable experiment](ELITETABLE_META_EXPERIMENT.md) and
`experiments/ocean-elitetable-meta.yaml`. This page describes the retained GEPA setup.

This implements a **mixed-environment pilot** based on [the meta-experiment design](OCEAN_META_EXPERIMENT.md).
GEPA 0.1.4 edits the complete Python search controller in
[`research/meta_ocean/controller.py`](../research/meta_ocean/controller.py).
The controller generates, evaluates and revises game policies. GEPA receives the
results of complete fresh controller trials; it does not directly edit game policies.
The same controller source runs fresh on **2048, Breakout and Maze**, with a separate
full search budget for every environment/replicate. Scores are normalized per task,
then averaged with equal environment weights. All three tasks participate in GEPA
selection; held-out evaluation tests new seeds/maps, not a held-out environment.

Three independent campaigns optimize normalized final performance, performance per
actual output token, and performance per evaluated proposal. A GEPA example is the
entire replicated development benchmark, so ratios use aggregate resource totals.
GEPA parent reevaluations and candidate validation each consume another full benchmark
credit. The runner enforces its own hard admission cap as well as GEPA's budget.

## Run

Run the coordinator on the host with the existing patched Ocean installation and
a running Docker daemon. Do not use `scripts/run`: the coordinator needs to start
separate containers, and the ordinary launcher does not expose Docker.

```bash
uv sync --extra meta --extra ocean
uv run python scripts/install_ocean.py
uv run python scripts/install_maze.py
docker build -f research/meta_ocean/Dockerfile -t rsikit-meta-worker:local .

# Read-only configuration/workload inspection; no workers, credentials or model calls.
uv run python -m research.meta_ocean \
  --config experiments/ocean-gepa-meta.yaml --print-config

# Requires OPENROUTER_API_KEY in the environment; makes paid calls.
uv run python -m research.meta_ocean \
  --config experiments/ocean-gepa-meta.yaml \
  --objective performance --output runs/gepa-meta-performance-1
```

Omit `--objective performance` to run all three campaigns independently, as the
checked-in config requests. The model is fixed across inner generation and outer
editing. Set your price ceilings and spend caps in the YAML before launching;
the supplied prices are illustrative user-set bounds, not verified provider quotes.
The command never overwrites or resumes an existing output directory.

The shared Rich dashboard starts before Ocean/Docker setup and shows calibration,
policy evaluations, private audits, GEPA selection, and model-call usage. Completed
game counts update after each native batch (at most 32 games). The same events are
saved to `run.log`; redirected output uses plain logs. Calibration runs 384 games
before the first model call. PufferLib's legacy Gym import may print Gym's maintenance
notice at startup; this notice alone does not indicate an evaluation failure.

The default **per objective** limits are two controller edits, nine complete
development benchmarks, two development replicates, and at most 50 policy evaluations
per environment-specific trial. Each frozen controller is subsequently run on three
validation replicates and five final-test replicates **in each environment**, alongside
the unchanged starting controller. All
trials have the same 50-evaluation/250,000-output-token pilot ceiling. The separate
500-evaluation final tier in the larger design is not used by this pilot.

The default all-objectives workload ceiling is 306 controller trials, 15,300 policy
evaluations, 153,000 search games, 59,904 private audit games, and 384 calibration
games. Aggregate model reservation ceilings total $1,545 (306 × $5 trial caps plus
three $5 editor caps); actual admission can stop earlier. `--print-config` recalculates
these upper bounds. Creating this implementation and running its tests does not
launch that paid workload. The smaller panels reduce overhead per task, but keeping
the full search budget for three tasks triples the number of possible inner searches.
The final 512-case audits dominate the remaining audit workload.

## Game counts and scoring

**One policy evaluation = ten seeds × one episode per seed × up to 2,000 steps, in one environment.**
Batch size 32 only limits concurrency. The maximum is **20,000 transitions**, not
640,000. Early termination reduces actual work. Invalid programs, duplicates,
timeouts and repaired submissions each consume another evaluation credit; reusing
an already received result inside a controller costs nothing.

For each trial, the controller commits one successfully evaluated policy ID. After
the controller container exits, the trusted runner audits that policy on 64
disjoint private cases during development and validation (`audit_cases`), and 512
cases during final testing (`final_audit_cases`). If the controller fails, its last commitment survives;
without a commitment, the common starter survives. Invalid audit behavior receives
raw score zero. Private audits are logged separately and never count as inner
evaluation credits. They provide no feedback to that trial's controller.

Calibration evaluates two frozen policies on the same **64 additional seeds per task**
(`calibration_cases`), outside every search/audit panel. Every task must produce
`reference > baseline`, otherwise execution stops before paid generation. All three
objective campaigns reuse the same frozen anchors. There is no automatic panel growth;
increase `calibration_cases` if a task's scale proves unstable.

| Task | Starter | Reference | Raw score and episode limit |
| --- | --- | --- | --- |
| 2048 | Legal-move priority | Board two-ply | Merge score; at most 2,000 decisions |
| Breakout | No-op | Ball-tracking paddle | Episode reward; frameskip 4; at most 2,000 decisions |
| Maze | Initially stationary | Depth-first explorer with per-row memory | Solved = 1; 15×15 map; native timeout 450 decisions |

For environment `e`, average over its independent search replicates first:

```text
S_e = 100 × (mean private score_e − baseline_e) / (reference_e − baseline_e)
S = mean_e(S_e)
T = mean_e(mean actual output tokens per trial in e)
E = mean_e(mean evaluated proposals per trial in e)
performance = S
token efficiency = 1,000,000 × S / T
evaluation efficiency = 100 × S / E
```

Efficiency entries qualify only at `S >= 100`. GEPA uses an ordering-preserving
selection score: unqualified entries rank by S below all qualified entries;
qualified entries rank by their chosen ratio. Unknown token usage and zero resource
denominators have no defined ratio and cannot win the corresponding efficiency
leaderboard. Raw per-task scores and aggregate metrics remain visible even if nobody qualifies.

Model admission conservatively reserves the full input/output bound before every
call, never refunds reservations, and caps input at ten times the trial output
budget. These reservations can stop generation before its nominal output budget
is used. Actual provider completion usage, including provider-reported reasoning
already included in completion totals, defines T. Missing usage stays unknown.
Outer editor calls and costs are recorded separately from deployment efficiency.

The selected controller is frozen before external validation and test. Their results
cannot trigger edits or reselection. GEPA's own development “valset” is distinct
from these held-out phases. Replicate IDs define disjoint search and audit panels;
repeated development benchmarks reuse the same panels but start fresh controllers,
archives, policies and model budgets. Independent search replicates, not individual
games, are the replication unit.

2048 and Breakout retain the pinned 3.0 episodic bindings. Maze uses the newer
[pinned Ocean source](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/maze/maze.h)
through a headless C adapter built by `scripts/install_maze.py`. The installer verifies
the source SHA-256, retains upstream generation/observations/movement, removes rendering
and autoresets, freezes terminal rows, and counts goal/timeout endings once. It generates
one fixed-size map directly from each seed at difficulty 0.5, instead of sampling the
upstream shared map bank. This is an adapted task configuration, not the upstream training
benchmark. Map hashes across all configured calibration/search/audit splits are checked
before model calls and saved to `maze_splits.json`; duplicates abort the run.
Maze permits bounded per-row memory, reset for each batch. Tests verify batch-width and
seed-order independence. Compiler/platform differences may change generated maps; source,
adapter, binary hashes and map identities are recorded.

## Isolation and artifacts

Controller and policy containers have no network, host mounts, credentials, run
directory or Ocean installation. They run as an unprivileged user with a read-only
filesystem, bounded temporary storage, dropped capabilities, PID limits, two CPU
limits and 1 GiB memory limits. At most one controller and one policy worker run
together. JSON is the only return channel; generated code is never imported or
unpickled in the trusted coordinator. The coordinator owns native game state and
accepts only validated actions from the policy worker. Timeouts remove containers
by unique name. The selected image is pinned by its inspected digest for a run.

The pilot executes requests serially. This has a throughput ceiling: four-way
generation/candidate scheduling and a separate timing qualification can be added
when pilot correctness is established. Existing cooperative-evaluator timing figures
do not apply to this isolated path.

Each run stores resolved configuration, source/native/image provenance, frozen
calibration results, GEPA logs and checkpoints, raw editor requests/responses,
controller proposals, model usage events, oracle admissions/responses, policy
sources, per-seed game results, commitment records, scorecards and phase summaries.
Cancelled evaluations remain charged and are marked incomplete; their lost
transitions are unknown. `completed_transitions` counts only completed panels.

## Verification and scope

```bash
uv run python -m unittest tests.test_meta_ocean tests.test_ocean_maze -v
RSIKIT_META_DOCKER_TESTS=1 uv run python -m unittest tests.test_meta_ocean tests.test_ocean_maze -v
```

The second command exercises actual GEPA, Docker isolation, Ocean rollouts,
controller failure recovery and held-out scoring with **scripted model responses**.
It makes no API calls. Docker and the worker image must already be available.

This pilot uses frozen heuristic calibration anchors and a frozen-controller control.
The broader design's learned best-of-N anchors, independent-generation control,
held-out environment transfer, concurrent scheduling and recursive proposer comparison
remain separate work. Its results cannot establish general search improvement or a benefit
from recursive self-improvement. The integration uses the released
[GEPA adapter API](https://github.com/gepa-ai/gepa/blob/v0.1.4/src/gepa/core/adapter.py).
