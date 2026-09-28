# Evaluation ownership implementation plan

Approved design: conversation, 2026-09-28. User authorized implementation.

Goal: Evaluator produces Episode; execution transports episodes; optimizers interpret them;
Run persists one optimizer run without owning execution.

Constraints: preserve existing uncommitted work, no new dependencies, caller-owned
Evaluator instances/reset/cleanup, safe data-only sandbox output, preserve successful
siblings and durable checkpoints on failures. Work inline in the current checkout.

- [x] 1. Transport complete Episode objects and artifacts through sandbox/Executor.
  Add bounded codec validation; test initial/final observations, flags, arrays, artifacts,
  malformed output and cleanup. Remove scalar Result. Keep worker setup outside Evaluator.
- [x] 2. Make Run storage-only: create/open without environment/executor; save_policy,
  save_episode/load_episode and generic SQLModel save/database; preserve existing data.
  Test reopen/move, identity, artifact containment, checkpoint roundtrip, no Docker startup.
- [x] 3. Move EvaluationResult and screening to AlphaEvolve; use minimal reward measurements
  for other optimizers. Add research rollout collection using Executor with explicit Run
  persistence and optional cached episodes. No core evaluate_gym or fitness aggregation.
- [x] 4. Migrate optimizers, CLIs, replay, benchmarks and tests to explicit execution ownership.
  Resume optimizer checkpoints rather than invoking Run.resume. Preserve partial success.
- [x] 5. Remove obsolete core APIs, update docs/examples, rebuild sandbox, run full suite,
  review complete changes and fix material findings.

Progress and decisions
- Existing Run score columns remain readable; saved optimizer-supplied scores are data,
  not computed by Run. Old scores are not fabricated into trajectories.
- Existing optimizer SQLModel records and experiment configuration are the checkpoint
  formats; avoid introducing a parallel checkpoint framework.

Validation
- New episode codec/storage tests first failed against scalar execution and Run's
  environment requirement, then passed after migration. Full suite: 221 tests pass,
  including real Docker execution and exact multi-step trajectories on both backends.
- Additional example suites: 38 tests pass, including interrupted resume, cached seeds,
  changed-contract rejection, held-out reporting, action coercion, and poker repair.
- Synthetic scheduling benchmark runs in both modes using full Episode results.
- Ruff lint/format pass for all Python source files; changed documentation is formatted.
  Repository-wide checks still flag the pre-existing paper1.ipynb import/format issues
  and formatting in the historical persistent-sandbox plan.
- Fresh independent review found no critical/important production issues. Its invalid
  output test gap was fixed: malformed Episodes now assert the finite-reward and path
  diagnostics, rather than failing merely because old score tuples were supplied.
- Migration regressions found by tests were fixed: mean_rewards re-raises the original
  timeout subtype, all fake backends return Episodes, and poker failures use explicit
  failure keywords. Successful siblings and completed seed reuse remain covered.
