# Unified Optimizer Contract Implementation Plan

> The later episode-feedback revision replaces `Measurement` with seed-keyed
> `Episode` objects carrying candidate errors. See [the current contract](../../INNER_LOOP.md#one-optimization-loop).


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make every existing search algorithm a compliant example of the same externally managed propose → evaluate → update interface.

**Architecture:** Move neutral measurements into core and extend the existing optimizer protocol with explicit round/completion semantics. One core runner drives every algorithm; each optimizer retains its own selection state and interprets measurements, including construction of AlphaEvolve's private `EvaluationResult`. Environment adapters retain execution, screening, and resource ownership.

**Tech Stack:** Python 3.10+, asyncio, dataclasses/typing, existing Gymnasium/Slick/Pydantic/SQLModel/Rich integrations, unittest, Ruff.

**Spec:** [Unified optimizer contract](../specs/2026-10-04-unified-optimizer-contract.md). This records the conversation's accepted architecture and the concrete interfaces used below; read it before implementation.

## Global Constraints

- Python 3.10+; existing dependencies only; existing unittest and Ruff tooling.
- Core never imports research/examples. Algorithms remain independent clients of core.
- Paid calls are not needed for verification. Do not modify unrelated files.
- Preserve existing algorithm selection, ancestry, repair budgets, accepted seed evidence, persistence, and supported recovery paths.
- The new schedule is complete-round `round-v1`; generation/evaluation overlap is deliberately deferred. Do not claim identical trajectories or performance.
- No new `fit()` loop, universal AlphaEvolve result type, stateful optimizer base class, or algorithm-specific branches in the shared runner.
- Use `rtk` for shell commands as required by the repository's supplied instructions.
- Implemented on `unify-envs`; the full Docker suite passes. Repository formatting retains two pre-existing historical-plan code-fence failures; changed files pass.

## Review Focus

1. Same policy ID produced by multiple attempts: one evaluation updates all associated attempts without losing ancestry (Tasks 1–3, 7).
2. Failure/repair mixtures: successful siblings survive, new repair IDs replace old pending IDs, and repairs do not consume a second original proposal or generation (Tasks 2–5).
3. Interrupted runs: partial execution/model-call evidence survives; resume never resets spend or repeats completed promotion (Tasks 4, 6).
4. Information loss: per-seed variation, independently measured features, and cheap screening survive the common measurement boundary (Tasks 1–2).
5. Scheduling: parent snapshots and family culling barriers remain correct even though overlap is removed; empty proposals cannot busy-loop (Tasks 1, 4–5, 7).

## File responsibilities and sequence

| Files | Responsibility |
| --- | --- |
| `rsikit/evaluation.py`, `rsikit/__init__.py` | Canonical `Measurement` and public exports; existing episode evaluator stays intact. |
| `rsikit/optimization.py` | Structural optimizer protocol, stateless feedback validation if useful, and one `search()` loop. |
| `research/rewards.py`, `research/environments.py`, `research/ocean/environment.py` | Neutral scoring/evaluator adapters and compatibility import. |
| Existing algorithm `agent.py`, records/history/database files | Proposal/update transitions and algorithm-owned checkpoint state. |
| Existing AlphaEvolve evaluation/pipeline/search and Shinka search modules | Evaluation composition and thin migration wrappers; delete duplicated loops. |
| `research/alphaevolve/cli.py`, `research/shinkaevolve/cli.py`, `research/lineagesearch/cli.py` (new), existing Elite adapter | Small component adapters using the existing CLI file contract. |
| `research/cli.py`, `research/experiment.py`, existing examples/docs/tests | Wire all built-ins, preserve lifecycle/output behavior, demonstrate interchangeability. |

Implement Task 1 first, then AlphaEvolve, ShinkaEvolve, EliteSearch, and LineageSearch. Integrate the runner into CLI/examples only after their protocol behavior passes. Finish with shared conformance and migration checks.

## Task 1: Establish neutral measurements and the shared runner

**Files:** modify `rsikit/evaluation.py`, `rsikit/optimization.py`, `rsikit/__init__.py`, `research/rewards.py`, `tests/test_optimizer_contract.py`, `tests/test_evaluation.py`; create `tests/test_search.py`.

**Interfaces:** publish the exact `Measurement`, `Optimizer`, and `search` signatures in the spec. `research.rewards.Measurement is rsikit.Measurement` must be true. `search` takes a bound evaluator function, not an evaluator subclass.

- [x] Add `test_measurement_preserves_evidence_and_rejects_invalid_values`: `{0: 0, 1: 10}` remains intact; named metrics/features survive; positional `Measurement({0: 3}, "ok")` still works; failure forces rejection; reject NaN/Inf, boolean scores, noninteger seed keys, invalid field types, and empty accepted evidence.
- [x] Add runner tests with a tiny two-round fake: calls are exactly propose/evaluate/update/propose/evaluate/update; returned best is the optimizer's best; reject missing/extra IDs, wrong result types, and duplicate proposed IDs before update; `[]` with `done=False` raises instead of looping; already-done returns best without evaluator calls. Check exceptions/cancellation checkpoint and propagate, including preserving a primary error when checkpointing fails.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_search tests.test_optimizer_contract tests.test_evaluation -q`; confirm the new cases fail for the absent API, then implement the minimal definitions and sequential runner. Keep existing episode-contract tests until their migration in Task 2.
- [x] Run the same command plus `tests.test_package_boundaries`; require all applicable new tests to pass and identify any old API tests awaiting the explicit migration. No algorithm imports or type branches in core.
- [x] Commit only this task's files after its checks pass. Commit message: `feat: define round-based optimizer and measurement contract`.

## Task 2: Make all AlphaEvolve variants compliant

**Files:** `research/alphaevolve/original/agent.py`, `research/alphaevolve/improved/agent.py`, `research/alphaevolve/paper/agent.py`, `research/alphaevolve/paper/evaluation.py`, `research/alphaevolve/paper/pipeline.py`, `research/alphaevolve/search.py`, `research/alphaevolve/history.py`, `research/alphaevolve/paper/database.py`, relevant `__init__.py`; tests `test_alphaevolve.py`, `test_variants.py`, `test_paper_agent.py`, `test_paper_pipeline.py`, `test_optimizer_contract.py`, `test_paper_feedback.py`.

**Interfaces:** all three classes implement `propose()`, `update(Mapping[str, Measurement])`, `done`, `best`. Add `batch_size: int = 10`, `proposals: int = 250`, `generation_concurrency: int = 4` to baseline `Config` (inherited by paper); validate positive batch/concurrency and nonnegative proposal budget. Legacy wrappers explicitly translate their supplied budgets into these settings. Keep `generate(n, concurrency=...)` as a lower-level helper, not a second loop.

**Evaluation boundary:** keep `EvaluationResult` private to the algorithm package. Change `assess(...)` and `evaluate_cascade(...)` to return neutral `Measurement`; change `EvaluationStage.evaluate` and grader types accordingly. Preserve cheap-stage pruning and latest-measurement precedence. `AlphaEvolve.update()` constructs the internal results and passes them to existing archive operations.

- [x] Extend the common contract test to original/improved/paper. One round with seed scores `{0: 3, 1: 7}` yields mean 5; paper derives worst 3 and stability -2. `{0: 0, 1: 10}` and `{0: 5, 1: 5}` remain distinguishable. Explicit custom metrics/features override derived defaults and missing configured descriptors fail before archive changes. Duplicate attempts share one returned policy ID and both settle on one update.
- [x] Add repair/round cases: next proposals queue only failed candidates; a repair ID change preserves the original attempt; original proposal limits exclude repairs; repeated updates and propose-before-update raise; founders are measured before descendants; all-invalid generation eventually reaches its finite proposal limit. Preserve original versus improved founder tests.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_optimizer_contract tests.test_alphaevolve tests.test_variants tests.test_paper_agent tests.test_paper_pipeline tests.test_paper_feedback -q` to see the expected contract failures. Implement transitions, neutral screening/assessment, and private result construction. Persist original-attempt count, pending phase, and repair queues alongside existing paper checkpoint metadata, with absent-field handling for legacy completed checkpoints.
- [x] Replace paper pipeline orchestration with a thin call to core `search`; migrate overlap-specific tests to explicitly assert founder ordering, stage concurrency limits, and complete-round barriers. Keep screening tests asserting that expensive evaluation is never called after rejection. Repeat the command; all tests must pass. Do not describe the new schedule as the published asynchronous pipeline.
- [x] Commit after passing checks: `refactor: drive AlphaEvolve through the shared optimizer contract`.

## Task 3: Make ShinkaEvolve compliant

**Files:** `research/shinkaevolve/agent.py`, `research/shinkaevolve/search.py`, `research/shinkaevolve/records.py`, `tests/test_shinkaevolve.py`, `tests/test_optimizer_contract.py`.

**Interfaces:** add `batch_size: int = 25`, `generations: int = 10`, `generation_concurrency: int = 4` to `Config`; validate positive batch/concurrency and nonnegative generation count. Existing example/wrapper arguments explicitly override these defaults. `propose()` opens a generation or performs queued repairs. `update()` consumes neutral measurements and derives mean fitness before existing population/model-gain operations. Rename the old scalar updater to the explicit `update_scores` helper if still needed internally.

- [x] Add Shinka to the shared contract tests. Check model gains and migration counters update once per original attempt, repairs do not create another generation, duplicate feedback cannot double-count, and seed scores produce the same means used by current selection tests.
- [x] Add `test_reflection_runs_before_next_proposal_without_evaluation_in_optimizer`: preserve existing reflection/novelty provider calls in proposal preparation, and use a fail-on-call evaluator sentinel to prove the optimizer never invokes evaluation itself.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_shinkaevolve tests.test_optimizer_contract -q` for the expected new failures; implement the phase transitions and turn `run_search(...)` into configuration/persistence plus core `search` delegation.
- [x] Repeat the command; require existing patch, novelty, allocation, migration, provider-error, and generation-record tests to pass. Do not add unsupported resume promises.
- [x] Commit after passing checks: `refactor: expose ShinkaEvolve proposal and feedback rounds`.

## Task 4: Extract EliteSearch's proposal and feedback phases

**Files:** `research/elitesearch/agent.py`, `research/elitesearch/records.py`, `research/elitesearch/cli.py`, `research/meta_ocean/checkpoints.py`, `tests/test_elitesearch.py`, `tests/test_experiment.py`, `tests/test_meta_resume.py`, `tests/test_optimizer_contract.py`.

**Interfaces:** `EliteSearch(task, provider, evaluate=None, *, ...)` temporarily accepts its legacy evaluator only for a thin `run()` compatibility wrapper. Canonical `propose()`/`update()` never access it. The wrapper delegates to core `search` and returns `self.elites`, preserving old callers. New integrations omit `evaluate`. Existing `Config.population_size`, `elite_size`, and `generations` own round size/stopping.

- [x] Add Elite to common conformance. Pin two generations with population 5/elites 2: all second-generation parents come from the first-generation snapshot; promotion occurs once only after terminal repair outcomes; existing operator counts and tie rules remain unchanged.
- [x] Add failure/restore cases: one success and one failure cause only the failure's replacement to be proposed; a completed generation is not promoted again on restore; an incomplete round restores its outstanding candidates and repair allowance from records. Preserve the existing model-call-ledger assertions and moved-run-directory test.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_elitesearch tests.test_optimizer_contract tests.test_meta_resume -q` for the expected failures. Split `_experiment`/`_measure`/`run` into proposal generation, validated update, and phase finalization, reusing `_population`, `_rank`, `_promote`, and repair machinery. Keep candidate-level paid-call checkpoints.
- [x] Repeat the command plus `tests.test_experiment`; keep completed candidate records byte-equivalent where existing tests require it. Update schedule tests to assert within-stage concurrency and the deliberate round barrier. Verify budget exhaustion preserves already returned evidence and does not mark unresolved rounds complete.
- [x] Commit after passing checks: `refactor: run EliteSearch through shared proposal rounds`.

## Task 5: Extract LineageSearch's proposal and feedback phases

**Files:** `research/lineagesearch/agent.py`, `research/lineagesearch/records.py`, `tests/test_lineagesearch.py`, `tests/test_optimizer_contract.py`.

**Interfaces:** match Elite's optional legacy evaluator/delegating `run()` pattern, preserving `Study` as the wrapper's return value. `done` reflects terminal study state. Existing `Config` remains the source of attempt limits, family counts, expansion sizes, and patience. Use a small private phase value plus existing family/trial records; no general state-machine framework.

- [x] Add Lineage to conformance and pin the sequence: discovery → founder planning for all families → complete exploration sweep → global cull → individual bonus expansions. Check global culling waits for all sweep expansions and their repairs, and later parents are surviving frontier members only.
- [x] Retain paired-seed uncertainty tests and add mismatched-seed rejection before mutation. Pin a truncated final expansion's existing `full_batch=False` patience behavior; generation/planning failures must retain existing attempt and patience accounting. All families retired or the attempt cap reached terminates without empty-round spinning.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_lineagesearch tests.test_optimizer_contract -q` for expected failures. Separate `_expand` planning/generation from measured family updates; retain selected parents, expansion sizes, and phase boundaries until update. Queue repairs in proposal preparation; no evaluator invocation or hidden model calls in update.
- [x] Repeat the command; existing decomposition, refinement/pivot, culling, hypothesis, repair-budget, and best-policy assertions must pass. Keep `Study` and trial evidence available to checkpoint adapters without adding universal record methods to the core protocol.
- [x] Commit after passing checks: `refactor: separate LineageSearch planning from evaluation`.

## Task 6: Connect every built-in to the common runner and lifecycle

**Files:** create `research/alphaevolve/cli.py`, `research/shinkaevolve/cli.py`, `research/lineagesearch/cli.py`; modify `research/elitesearch/cli.py`, `research/cli.py`, `research/experiment.py`, `research/environments.py`, `research/ocean/environment.py`, `research/alphaevolve/search.py`, `examples/{alphaevolve,shinkaevolve,elitesearch,lineagesearch,ocean_search}.py`; tests `test_cli.py`, `test_experiment.py`, `test_ocean_search.py`, `test_meta_ocean.py`, `test_meta_elitetable.py`.

**Interfaces:** retain the existing component `Options`, `add_arguments`, and `async optimize(*, task, provider, evaluate, run, options, seed)` contract. Each built-in adapter constructs a compliant optimizer, calls core `search`, and returns unique successful finalists sorted by search fitness. Built-in selectors are `alphaevolve`, `shinka`, `elite`, `lineage`; AlphaEvolve adds `--variant paper|original|improved` with paper default. Use per-algorithm Config-backed options with `extra='forbid'`; expose numerical/boolean/string Config controls, not arbitrary callable hooks. Keep generation concurrency/timeout in the shared generation section, mapped into each optimizer's configuration.

- [x] Add table-driven CLI/config tests: each selector resolves; variant selection works; invalid settings fail before provider calls; unsupported videos/resume combinations fail explicitly. Search-phase measurements and independent validation/test panels remain the same across selectors.
- [x] Add integration tests spying on core `search`: every built-in delegates to it and returns correctly ordered policy finalists. Gymnasium/Ocean adapters return the canonical Measurement identity. Persist `optimization_schedule: round-v1` in new search manifests and preserve that value on resume.
- [x] Add recovery tests for both existing supported paths: unified Elite and paper AlphaEvolve. Preserve prior call reservations, source/ancestry, cached successful episodes, partial rounds, and winner selection. Accept tested legacy completed checkpoints; reject incomplete legacy streaming checkpoints before generation rather than guessing. Keep original provenance unchanged on legacy resume and record schedule changes separately in appended run evidence.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_cli tests.test_experiment tests.test_ocean_search tests.test_meta_resume tests.test_meta_ocean tests.test_meta_elitetable -q` to expose migration failures; implement small adapters and replace loop bodies in examples. Preserve Elite's generation-video callback and the custom-file optimizer contract. Update dependency-boundary allowlists narrowly for shared orchestration/provider utilities where an adapter actually requires them.
- [x] Repeat those checks in the existing Docker environment if optional Ocean/native dependencies are absent locally. Require preserved results/artifacts/usage accounting; keep new Shinka/Lineage resume unsupported. Commit after passing checks: `feat: expose compliant optimizers through unified experiments`.

## Task 7: Prove interchangeability and finish the migration

**Files:** `tests/test_optimizer_contract.py`, `tests/test_search.py`, `tests/test_package_boundaries.py`, affected existing tests; `README.md`, `research/README.md`, `docs/CLI.md`, `docs/INNER_LOOP.md`, `docs/RUNS.md`, `docs/{ALPHAEVOLVE,SHINKAEVOLVE,ELITESEARCH,LINEAGESEARCH}.md`.

**Interfaces:** demonstrate `from rsikit import Measurement, search` with the same evaluator function and loop for all six implementations. Document optimizer-specific construction separately. The search return value is always the best policy definition or `None`; algorithm history stays on its owning optimizer/Run.

- [x] Complete a six-implementation conformance matrix using existing scripted providers and real optimizer classes, not method-presence mocks. For each: success, a failed candidate followed by repair/exhaustion, screening rejection, duplicate/unknown feedback, pending-round misuse, and completion. Compare deterministic algorithm invariants, not old asynchronous arrival order.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_search tests.test_optimizer_contract tests.test_package_boundaries -q`; require one runner, no algorithm type branching, no cross-algorithm imports, and no core references to AlphaEvolve's EvaluationResult. Adjust stale episode-feedback tests to the explicit new contract rather than weakening assertions.
- [x] Rewrite API examples around the common loop. Explain seed-level evidence, algorithm-owned rounds/results, explicit failures, repair rounds, and schedule migration. Correct wheel packaging claims to match `pyproject.toml`. Remove duplicated orchestration after auditing callers with `rtk proxy rg -n 'propose\(|update\(|run_search\(|\.run\(|paper.*search' rsikit research examples tests`; retain only necessary thin wrappers.
- [x] Run repository checks: `rtk proxy ./scripts/run unittest discover -s tests -v`, `rtk proxy ./scripts/run ruff check .`, and `rtk proxy ./scripts/run ruff format --check .`. These run offline/scripted tests, not paid experiments. Report environment blockers explicitly; a missing Ocean installation is not a passing full suite.
- [x] Commit after passing checks: `docs: document and verify the unified optimization workflow`. Report that within-stage concurrency remains but cross-stage overlap is removed; no throughput claim without a separate measured comparison.

## Completion checklist

- [x] All six optimizers use the same core runner and neutral feedback shape.
- [x] AlphaEvolve alone constructs and owns its richer EvaluationResult.
- [x] Scalar means do not replace required seed evidence or measured descriptors.
- [x] No optimizer evaluates policies internally through its canonical proposal/update API.
- [x] Repair, culling, promotion, completion, supported recovery, and budget invariants pass.
- [x] All built-ins are reachable through the unified CLI; examples no longer implement divergent loops.
- [x] The complete-round scheduling change and API migrations are documented.
- [x] Full relevant checks pass, with no paid calls or unrelated edits.

## Deferred work

Streaming/partial updates, global prompt-root removal, new recovery support for previously non-resumable algorithms, and a `fit()` convenience method are outside this plan. None is required to make the algorithms compliant with the agreed complete-round interface.

Implemented sequentially in this workspace. Fresh review and its verified fixes are recorded in the branch history.
