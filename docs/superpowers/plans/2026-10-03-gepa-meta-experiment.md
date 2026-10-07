# GEPA Ocean meta-experiment implementation plan

**Goal:** Implement the 2048 plumbing pilot in `docs/OCEAN_META_EXPERIMENT.md`, with GEPA editing controller Python source and three separate objective campaigns.

**Architecture:** A trusted local runner owns Ocean, model credentials, accounting and private audits. Disposable Docker workers execute controller and policy source through bounded JSON messages. GEPA evaluates an entire replicated benchmark as one example so efficiency uses aggregate ratios. Existing episodic rollout logic remains authoritative.

**Scope:** 2048 pilot; no paid launch, Maze integration, recursive self-editing arm, or claims of general search improvement. The user's request authorizes implementation of the existing experiment design. Preserve unrelated working-tree changes.

## Constraints and review focus

- Ten independent search episodes, batch width 32, at most 2,000 steps each.
- Charge malformed, duplicate and failed policy submissions before validation. No cache across controller trials. Private audit after controller exit only.
- Meter outer editing separately from inner search. Unknown token usage means unavailable token efficiency, never zero usage. Enforce conservative admission bounds.
- Generated code never executes in the trusted runner. No host mounts, network, credentials or simulator in workers. Kill workers on timeout/cancellation.
- Freeze calibration and split panels; never use validation/test feedback for promotion. Preserve the last committed evaluated policy after controller failure.

## Tasks

- [x] Add failing tests for aggregate objectives, config/split validation, failed/duplicate submission accounting and committed-policy fallback.
- [x] Implement config, trial ledger and GEPA adapter with a generate/evaluate/revise seed controller. Pin GEPA as an optional dependency.
- [x] Implement bounded Docker JSON worker and protected policy factory for existing Ocean rollout; test isolation, timeouts and ten-seed accounting.
- [x] Add runnable YAML, command help/config inspection, durable artifacts and documentation; verify actual GEPA with scripted generation and actual Ocean/Docker without API calls.
- [x] Review changes and run relevant regressions; report verified scope and remaining larger-suite work.

## Verification record

- 13 meta-experiment tests pass with `RSIKIT_META_DOCKER_TESTS=1`, including the real GEPA/Docker/Ocean campaign with scripted generation and held-out phases.
- 29 Ocean, experiment and package-boundary regression tests pass. Ruff and diff whitespace checks pass.
- Independent review identified GEPA's swallowed editor-budget exceptions, promotion before full validation, interrupted-evaluation accounting and malformed JSON/action handling; fixes have regression coverage.
- Repository-wide discovery ran 293 tests: four errors and five intentionally disabled Docker tests. The errors are outside the changed flow:
  - `test_application.ApplicationEpisodeTests.test_replay_records_best_completed_policy_without_changing_original_scores`: video episode process exited without a result.
  - `test_application.ApplicationEpisodeTests.test_run_records_real_video_artifact`: video episode process exited without a result.
  - `test_application.ApplicationEpisodeTests.test_scientific_libraries_in_episode_process`: local environment lacks `control`.
  - `test_scientific_libraries.ScientificLibrariesTests.test_numerical_operations`: local environment lacks `control`.
- No paid model calls or paid campaign were launched. The worker image was built locally; GEPA 0.1.4 was installed in the project virtual environment and locked as an optional dependency.
