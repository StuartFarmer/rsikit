# Paper 1 protocol — 2026-09-25

**Frozen preparation:** use the [companion protocol](companion/examples/elitelist_papers/PROTOCOL.md)
and [launch handoff](companion/RUN_MAIN.md). The text below records the development-stage plan.

Status: study scope agreed; main-study configuration prepared. Complete the
reproducibility preflight below before freezing a release and launching main runs.
Existing `pilot-*` runs remain exploratory and are not main-study replicates.

## Question and scope

Can EliteTable evolve executable policies whose held-out reward improves over a
fixed search budget, and what happens after a policy reaches a score target?
The LLM weights remain fixed. Describe observed improvement, saturation,
regression on held-out episodes, and unsuccessful searches. No claim that the
method outperforms independent sampling or other algorithms belongs in Paper 1.
Independent sampling and comparative evidence belong in the follow-up paper;
systematic ablations are also deferred.

CarRacing is a central case study, selected because its exploratory trajectory
showed sustained improvement. Disclose that selection and retain all main-study
replicates, including disappointing ones. Raw reward differences across tasks
have different scales; a larger CarRacing difference does not establish a larger
normalized effect than LunarLander.

## Main experiment

| Setting | Value |
| --- | --- |
| Environments | CartPole-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3, BipedalWalker-v3, CarRacing-v3 |
| Independent searches | 10 per environment, search seeds 0–9; 60 searches total |
| Generations | Exactly 10 for every successful search; `--no-early-stop` |
| New candidates per generation | 50; 500 scheduled per complete search |
| Retained elites | 10 |
| Operators | Generation 1: fresh proposals. Later: 20% fresh, 40% remixes, remainder edits; three remix parents |
| Repairs | At most 5 per candidate; repair calls recorded separately |
| Model | `openai/gpt-oss-120b:nitro`; archive actual provider and response metadata |
| Search episode seeds | 0–9, fixed across candidates and searches |
| Held-out episode seeds | 1000–1099; separate from the inspected pilot panel 100–199 |
| Environment settings | Standard installed Gymnasium 1.3.0 profiles and registered episode limits; record all resolved settings |
| Policy observations | Native observations; CarRacing uses 96×96 RGB images |
| Reporting targets | CartPole 475; MountainCar −110; Acrobot −100; LunarLander 200; BipedalWalker 300; CarRacing 900 |
| Proposed execution limits | 10 s per policy call and 10 s per episode; 4 episode workers, 50 model requests concurrently |

Execution limits and the evaluator image must pass the preflight before the
protocol is declared frozen. In particular, the CarRacing pilot had episode
timeouts. Confirm that the chosen limit and concurrency measure policy behavior
consistently on the target hardware. Any adjustment happens before main runs,
is documented here, and applies to the entire affected environment's main group.
Do not silently mix backends or limits inside that group.

Targets are annotations, not stopping rules. New manifests preserve them in
`reporting_target`; `config.target_score` is null when early stopping is disabled.
Do not change an old early-stopped run into a main replicate: its original prompt,
stopping rule, and development history differ. Start new `main-v1-*` directories.

The budget is 30,000 scheduled candidates across 60 complete searches, before
repair calls and held-out evaluation. Ten search seeds imply up to 300,000
candidate-episode evaluations before failed attempts/repairs and cache reuse.
These are planning counts, not measured environment steps, runtime, or API cost.
Launch sequentially by run initially; record actual resource use.

## Analysis fixed before main results

- Evaluate each generation's search-selected winner on the held-out panel only
  after search finishes. No held-out feedback, repair, or winner reselection.
- Primary quantity: generation-10 minus generation-1 held-out mean reward,
  paired within each independent search and reported separately by environment.
- Plot all ten observed generations, individual searches, median and interquartile
  range, and valid-run counts. Report final reward, improvement, and search/held-out
  target attainment separately. The IQR is descriptive, not a confidence interval.
- Report the median paired improvement and a descriptive 95% percentile bootstrap
  interval from 10,000 resamples of whole search IDs, RNG seed 20260925. With ten
  searches, intervals have limited precision; do not infer population-wide reliability.
- Keep generation-one and random-action references. These do not establish a
  benefit over budget-matched independent generation.
- Count all 60 planned searches in the inventory. Invalid policies, timeouts,
  incomplete runs, and missing held-out evaluations stay visible. Never replace
  missing rewards with zero or discard an unfavorable replicate.
- Resume infrastructure-interrupted runs with the same protocol and cached work.
  Record any irrecoverable failure; a replacement is an additional labelled run,
  not an invisible replacement for the failed attempt.
- For each environment, illustrate the complete run nearest the median paired
  held-out improvement, with lower search seed breaking ties. Show CarRacing's
  generation-1/5/10 policies and lineage. Report missing checkpoints explicitly.
  Policy differences are descriptive; causal operator claims require ablation.
- Report calls, repairs, invalid candidates, returned token usage/cost, and elapsed
  time across attempts. Missing billing stays unknown. No exact step-efficiency claim
  until environment-step counts are recorded and checked.

## Reproducibility deliverable

Deliver one paper and a standalone companion repository with its own setup,
source, prompts, experiment configuration, archived evidence, and analysis.
It must run without the author's checkout or absolute local paths. Vendor the
needed current code with provenance rather than rewriting the search algorithm.
Pin dependencies and the worker environment; retain third-party notices and
resolve the project's license before public distribution.

Three separate acceptance checks:

1. **Rebuild the paper evidence offline:** a fresh checkout reproduces the tables
   and figures from the archived raw results, without Docker, API credentials,
   or paid calls. Numeric tables must match; incidental PDF metadata may differ.
2. **Reevaluate saved policies:** using the archived environment/image and seeds,
   rerun generated code inside the sandbox. Compare episode returns and failures;
   document hardware/numerical tolerances instead of silently accepting mismatches.
3. **Rerun the search:** documented paid commands launch the exact protocol and
   save the same evidence contract. Hosted LLM output/provider routing is not
   deterministic; new searches need not produce identical code or rewards.
   Report their distribution alongside the archived main-study evidence.

## Preflight and freeze checklist

- [x] Preserve target annotations with early stopping disabled.
- [x] Configure six environments, ten generations, ten replicates, and fresh test seeds.
- [x] Include the lockfile and Dockerfile in future run source snapshots.
- [x] Refresh the pilot evidence and preserve its provenance; do not pool it into main results.
- [x] Assemble the standalone companion checkout and verify imports/install in a clean directory.
- [ ] Freeze the host dependencies, worker dependencies/base image, prompts, source revision,
  tie-breaking rule, and environment configuration; archive hashes and a runnable image.
- [x] Validate the CarRacing timeout/concurrency budget with archived pilot policies
  on development seeds, then check all six environments through the intended backend.
- [x] Test ten-generation continuation after target attainment and interrupted resume
  through the complete workflow; verify per-generation held-out reports.
- [ ] Reconcile complete API usage/cost accounting and estimate the full batch budget
  from a representative end-to-end run, including repairs and held-out work.
- [x] Regenerate analysis from the clean checkout without credentials.
- [ ] Record the frozen release hash/date here, then run `main-v1-*` without protocol changes.

The current source snapshots are evidence archives, not yet an independently
validated release. The parent checkout has uncommitted code; its Git SHA alone
does not identify the code used. Snapshot hashes and the frozen companion release
must identify the actual main-study implementation.

Standalone execution preflight completed on 2026-09-25: see [measurements and checks](companion/PREFLIGHT.md).
The 60-search main experiment and manuscript are still outstanding.
