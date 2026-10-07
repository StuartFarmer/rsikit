# EliteTable search over sub-evolvers

The outer optimizer is the existing elite search engine used by the EliteTable
experiments. It **writes Python sub-evolver programs**: new approaches, mutations,
and remixes of measured elites. The supplied experiment's outer budget is **20 proposals per
generation × 5 generations**, retaining 10 elites. Each proposal is a search
algorithm, not a game-playing policy. Invalid sub-evolvers can receive five repair
attempts using their exact source and actual validation/execution error.

Each sub-evolver is evaluated from scratch on 2048, Breakout and Maze, using one
development run per environment. Each trial runs **up to five inner
generations**, stopping early only on a hard resource limit, with **1–25 game policies per generation** in the supplied config
(the configurable hard maximum remains 50). The host enforces the
generation order and population limits, evaluates every submitted policy, and
returns the complete previous population with its source, scores and errors.
The generated program chooses population sizes, parents, prompts, mutations,
remixes and diversity mechanisms. The host enforces repairs. It can retain episode-local
search history between its five calls. It cannot run its own evaluations or
change the host's limits.

The outer writer receives the full specifications for all configured tasks, using
the same descriptions as the fixed EliteTable baseline. Each inner generation call
automatically includes the current task's specification before the sub-evolver's
request. Generated code controls the search prompt, but cannot accidentally omit
the game's observation format, action meanings, reward or episode rules.

The winning game policy is the highest search-scoring valid proposal across all
five generations. After the sub-evolver exits, that policy is privately audited.
Unexpected incomplete sub-evolver execution is reported as a failed outer candidate, which
EliteTable can repair. Failures of individual game policies are normal inner
feedback and consume population slots. There are no free retry evaluations.
The host unwraps a single unambiguous Python code block in model responses and
submitted policies before validation. Every invalid or crashing policy enters the
host's enforced loop using EliteTable's existing `SelfHealer`: send the exact failed
source and diagnostic to the model, validate and reevaluate the repair, then repeat
until success or five repair attempts. This works in the final generation too and
does not depend on the generated sub-evolver reading errors or requesting repairs.
The repaired source and score replace that slot's result. Five failed repairs leave
an explicit failure. The model/spend/evaluation limits remain hard limits; exhausting
one stops repair with a recorded budget error. A repair remains a revision of the
same population member, but every model call and reevaluation is charged.

## Run

The coordinator uses the host's Ocean installation and Docker daemon. The worker
image must be rebuilt for the new sub-evolver protocol:

```bash
.venv/bin/python scripts/install_ocean.py
.venv/bin/python scripts/install_maze.py
docker build -f research/meta_ocean/Dockerfile -t rsikit-meta-worker:local .

# Inspect resolved limits without model calls or workers.
.venv/bin/python -m research.meta_ocean \
  --config experiments/ocean-elitetable-meta.yaml --print-config

# Paid run; OPENROUTER_API_KEY must be set. Use a fresh directory.
.venv/bin/python -m research.meta_ocean \
  --config experiments/ocean-elitetable-meta.yaml --output runs/meta-elite-001
```

The default is **one performance-ranked search**, with token/evaluation efficiency
shown in its artifacts. `--objective all` launches three separate campaigns;
`tokens` and `evaluations` choose the respective efficiency ranking, with the
quality floor of S ≥ 0 (matching the fixed EliteTable baseline). The previous GEPA experiment remains
available under its original config. Existing processes do not adopt code or
configuration changes; no running experiment is stopped or restarted automatically.

## Resume

Stop the old coordinator with Ctrl-C and wait for it to exit, then use the same
configuration and output directory:

```bash
.venv/bin/python -m research.meta_ocean \
  --config runs/meta-elite-002/config.yaml \
  --output runs/meta-elite-002 --resume
```

The runner restores the outer population, lineage, scores and model-call ledger.
Completed environment/replicate trials are reused, including the fixed baseline.
A committed policy awaiting its private audit resumes at that audit. Partial
baseline searches restore the existing engine's checkpoint. Interrupted generated
sub-evolver trials restart because arbitrary container state is not serialized;
their prior model calls and admitted evaluations still count against their limits.
Unknown usage and in-flight calls keep their reservations. Cancellation never
creates a successful trial checkpoint. A restarted trial can exhaust its remaining
budget and stop; it does not receive a fresh 250-call allowance. Budget exhaustion
is recorded as `stop_reason: budget_exhausted`, then the best policy found is privately
audited. This rule applies to both baseline and candidate trials; budget exhaustion
does not invalidate a usable incumbent or abort the campaign.

Resume checks the saved configuration, simulator metadata, task definitions, image
digest and code hashes, and refuses concurrent writers. It uses the original pinned
worker image even if the local image tag has moved. No worker rebuild is needed
for the host-enforced repair update.
Only `trial_workers`, `evaluation_workers` and `model_workers` may change on resume;
all search, scoring and budget settings remain fixed. The new worker settings are
saved to `config.yaml` and recorded with their previous values in `resumes.jsonl`.

For a v2/v3 run such as `meta-elite-002`, the first resume performs a recorded upgrade
to the current protocol: it keeps the outer call ledger and first population's generated sub-evolver
sources. It archives **both the old baseline and candidate evaluations** as
`*.interrupted-N`, then measures them under enforced repairs. The old baseline
disabled repairs and cannot represent the new comparator. The evaluation ceiling
now includes repair revisions; outer repairs are raised to five. Token and spend
caps are unchanged. New-protocol trials have fresh trial budgets; archived work
remains discovery cost. The outer model budget is not reset.

The v6 task-context upgrade from v4/v5 **preserves the measured baseline and first
population's generated sub-evolver programs**, but archives candidate evaluations
and scores as `*.interrupted-N`. Candidate trials run again with task context and
fresh trial budgets; archived trials remain discovery cost. The outer model-call
ledger is retained. Subsequent outer proposals receive all task specifications.
`performance/upgrade-task-context.json` records this one-time migration. No Docker
rebuild is required. The v6-to-v7 worker-reporting update preserves all completed
work and proposals; it does not reset candidate evaluations or budgets.

If an old baseline failed only because it exhausted its model budget, resume audits
its saved incumbent instead of repeating its search. Its previous failed summary
and commit are retained as `*.interrupted-N`.

`resumes.jsonl` retains previous/current manifests, and `repair-upgrade.json` plus
`performance/upgrade-repairs.json` record the migration. `checkpoint.json` stores a consistent outer snapshot;
`baseline_checkpoint.json` does the same for each baseline search. Snapshots are
replaced atomically, while call and oracle ledgers remain append-only.

## Budgets and concurrency

| Setting | Default | Meaning |
| --- | ---: | --- |
| `population` | 20 | New sub-evolver proposals per outer generation |
| `generations` | 5 | Outer EliteTable generations |
| `inner_generations` | 5 | Enforced generations in each sub-evolver evaluation |
| `generation_size` | 25 | Maximum population members and sub-evolver model calls per inner generation |
| `inner_max_repairs` | 5 | Enforced repair attempts per failed policy, additional to its original attempt |
| `trial_workers` | 8 | Concurrent independent environment/replicate trials |
| `evaluation_workers` | 16 | Concurrent policy panels across all trials |
| `model_workers` | 8 | Concurrent inner and outer model calls, shared limit |

One policy evaluation still means **ten seeds × one game × at most 2,000 steps**.
Batch size 32 limits game concurrency and does not multiply that budget. Maze's
native 450-step timeout can end its games earlier. Population submissions execute
concurrently under the shared evaluation limit; their ordered results return at
the generation barrier. A sub-evolver can request parallel model generation by
passing multiple prompts to `await self.generate(prompts)`.

The per-trial population ceiling is 125 original policies. With up to five repair
revisions each, there can be at most 750 policy evaluation attempts. Original and
repair calls share the 1,024,000 output-token reservation ceiling and $25 spend cap.
At 4,096 reserved output tokens per call, that permits at most 250 total model
calls, so the full theoretical repair allowance may exhaust the token budget. These
are ceilings, not targets or billing predictions. Conservative input/output
reservations, per-call limits and timeouts can stop execution sooner. The outer
writer has a separate $25 cap. Failed outer repairs can consume additional full
trial budgets. `--print-config` includes those worst-case repair allowances.
This is a substantially larger search than the prior serial pilot; concurrency
improves throughput but does not reduce its total possible work.

There is **no heuristic calibration**. The reference is the existing EliteTable
policy-search engine, run for five generations of 25 proposals, retaining ten
elites and using its standard new/edit/remix operators with five enforced repair
attempts per failed policy. Revisions stay within the same population slot and
are charged against the same trial resource ceilings. It receives the same model, starter
policy, task instructions, resource ceilings, search seeds and private audit
seeds as the generated sub-evolvers.

The baseline runs once per environment/replicate/split and its results are reused
throughout this run. Outer proposals start concurrently; their fitness waits for
the matching baseline. Baselines are not automatically reused across separate runs.
Development and validation private audits use 64 games; final tests use 512.
The frozen winner and fixed baseline each receive one validation and one test
run per environment on separate seeds; those results do not affect selection. Held-out
baseline searches start only after the winner is frozen.
Maze map hashes are checked for split overlap before any paid generation.

For each environment, let C and B be the candidate and baseline mean private
scores across matched replicates. Its gain is `200 × (C − B) / (|C| + |B|)`, or
zero when both scores are zero. S is the equal-weight mean of those gains across
environments. **Zero matches EliteTable; positive beats it; negative loses.**
This symmetric percentage gain is bounded between −200 and 200; it is not ordinary
percentage change. It handles a zero baseline and prevents 2048's larger score
units from dominating Maze. Raw candidate/baseline scores and differences are also saved.

## Leaderboard and evidence

The Rich leaderboard ranks **sub-evolver programs**, showing their names,
descriptions, generations, mutation/remix parents, selection score, gain S,
output tokens T, policy evaluations E, and each environment's gain over EliteTable.
Inner policy events do not replace the outer leaderboard.

Each objective directory stores `run.log`, `leaderboard.json`, `organisms.json`,
`generations.json`, outer model calls, the frozen `winner.py` and `selection.json`.
Development results retain each proposed sub-evolver, generation populations,
per-policy results, model usage, final game policy, private audit and scorecard.
`repairs.jsonl` stores host repair attempts, exact failed sources/diagnostics,
raw repair responses and resulting evaluations. Generation results identify the
submitted source, final source and repair count.
Held-out validation/test results are saved separately. `baseline/{split}/{environment}/{replicate}/`
stores the fixed engine's configuration, organisms, generations, model usage,
policy panels, final policy and audit. The manifest hashes the baseline engine
and prompt templates so the reference is identifiable.

The container has no network, credentials, host mounts or simulator. All model
calls and game evaluations use trusted host oracles with independent accounting.
The [Ocean suite guide](GEPA_META_EXPERIMENT.md) describes the native environments
and Maze adapter; its heuristic calibration applies only to the legacy GEPA experiment.

```bash
.venv/bin/python -m unittest tests.test_meta_elitetable -v
RSIKIT_META_DOCKER_TESTS=1 .venv/bin/python -m unittest tests.test_meta_elitetable -v
```

The Docker integration uses scripted model responses. It exercises writing,
execution failure, source-aware repair, mutation, five-generation evaluation,
held-out audits, the model gateway and leaderboard without paid API calls. It also
runs the actual fixed EliteTable engine, checks matching baseline/candidate audit
seeds, and verifies invalid baseline proposals consume their slots.
