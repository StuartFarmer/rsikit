# Automatic Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implement inline in the existing workspace; the user supplied this branch. Do not stage or commit the earlier, still-uncommitted Docker migration as part of this work.

**Goal:** Make the approved Rich dashboard appear automatically for every standard optimization loop, with optimizer-owned custom leaderboard columns.

**Architecture:** Run supplies scoped logging and lifecycle management. Optimizers emit structured domain records through ordinary Python logging; one shared handler maintains and renders the display. The optimizer supplies rankings and column declarations, with no per-example UI setup or snapshot adapters.

**Tech Stack:** Python >=3.10, stdlib logging/contextvars/collections/time, installed Rich >=13,<16, existing unittest and scripted providers; Docker application runtime.

**Spec:** [Automatic optimization progress dashboard](../specs/2026-09-28-automatic-progress-dashboard.md). Records the user's approved direction and concrete planning decisions; implementation is recorded in the execution ledger.

## Global Constraints

- Integration must be automatic in the optimization loop, like logging.
- Application scripts must not construct dashboards, supply snapshot adapters, or call dashboard updates.
- Each optimizer can declare its own leaderboard columns.
- Poker retains its separate application and display.
- No new dependencies, event transport, second persistence system or Docker protocol.
- No paid model calls for validation; preserve scoring and existing resume guarantees.
- All shell commands must be prefixed with `rtk`; use `rtk proxy` for raw commands.

## Review Focus

- Concurrent Runs and inherited async task contexts: no cross-run logs, leaked handlers or conflicting Live displays (Task 2).
- Repaired/duplicate policies and stale revisions: identity is an attempt, not a policy hash; no double counting (Tasks 1, 3, 4).
- Per-seed and screening results: do not mark a candidate complete before the optimizer settles its full assessment (Tasks 3, 4).
- Resume, unknown history and early termination: truthful counts, unknown durations, no invented 100% (Tasks 1, 3, 4).
- Model-supplied text and large error bursts: literal safe display, bounded UI memory, complete durable logs (Tasks 1, 2, 5).

## File responsibilities

- `rsikit/progress.py`: structured-record handling, bounded display state, Rich rendering, scoped logging lifecycle. Replace existing ProgressHandler behavior here; retain show_scores only for remaining explicit report consumers.
- `rsikit/run.py`: enter/exit display support and optional Console injection; no research imports or optimizer-table inspection.
- `research/rollouts.py`: automatic environment metadata event; retain episode caching behavior.
- Optimizer `agent.py` modules and `research/alphaevolve/paper/pipeline.py`: domain lifecycle events and custom columns alongside existing state transitions.
- New `research/alphaevolve/search.py` and `research/shinkaevolve/search.py`: relocate real loops currently in example modules; no new algorithm framework.
- Standard example modules: remove old UI/logging setup, import relocated loops, retain CLI and artifact work.
- `tests/test_progress.py`: contract, rendering, lifecycle and scripted integration regressions; reuse existing algorithm tests for semantics.

### Task 1: Implement the event consumer and approved layout

**Files:** Modify `rsikit/progress.py`, `tests/test_progress.py`.

**Interfaces:** Internal `RunDisplay(path: Path, console: Console)` with `consume(record: logging.LogRecord) -> None`, `render() -> RenderableType`, `close() -> None`. Public producer interface is only logging with the spec's `progress` payload. `leaderboard_columns: dict[str, rich.table.Column]` is an optimizer declaration, not a plugin registry. Keep implementation in the existing module initially.

- [x] Add a deterministic consumer test: submit batch size 50, mark 44 unique proposals done and 26 evaluations settled; assert combined completed/total equals `(70, 100)`. Replay records and assert counts unchanged. Repair a candidate to revision 1; submit a stale revision 0 completion and assert it cannot finish revision 1.
- [x] Add table-driven cases for permanent proposal rejection, terminal evaluation failure, cancellation, zero budget, unknown budget, overlapping batches and an early stop. Assert only resolved slots count, oldest unfinished batch remains selected, and stop/cancel does not force 100%.
- [x] Add a render test at 120x45 and 80x24 with custom columns, long descriptions, literal `[bold]`, ANSI escape text and a traceback. Assert all core labels, valid column values, width bounds, omission indicators, safe literal content and supplied leaderboard order. A malformed payload must leave prior state intact and produce a diagnostic.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_progress -v`; confirm new tests fail for missing consumer behavior.
- [x] Implement the specified payload validation, attempt/revision state, counters, bounded recent views and approved Rich layout. Use Rich Table/Panel/Progress/Live, Text for untrusted strings, deque for bounded history, and monotonic time for current-session durations. Bound retained detail for completed batches: keep aggregate totals and the latest leaderboard, not all historical descriptions.
- [x] Run the same focused tests; require PASS. Regenerate a sample preview from the actual renderer for visual review, avoiding a second mockup implementation.

### Task 2: Activate scoped logging automatically through Run

**Files:** Modify `rsikit/run.py`, `rsikit/progress.py`, `research/rollouts.py`, `tests/test_progress.py`, `tests/test_run.py`.

**Interfaces:** Preserve Run APIs, adding keyword-only `console: Console | None = None` to create/open and carrying it to initialization. Internal `bind_run(path: Path, console: Console | None) -> contextlib.AbstractContextManager` owns scope; Run enters it exactly once in sync/async context entry and exits it from close. `ProgressHandler(logging.Handler)` routes a record to the ContextVar-bound RunDisplay and file handler. Neither optimizers nor examples call bind_run.

- [x] Add lifecycle tests: entering a Run and executing an ordinary structured log activates output without manually adding a handler; simply opening/reading or entering an unused Run stays silent; sync/async exit, repeated close, exceptions and cancellation remove owned resources and restore logger settings.
- [x] Add an asyncio test with two Run contexts and different task messages; assert each run.log contains only its own messages. Assert at most one Live owns a Console, sibling runs retain logs, and closing one run does not disable the other's logging. Test logs emitted after a task's Run has closed do not write to its closed file.
- [x] Add redirected-output and error-burst cases: no cursor controls, complete traceback in run.log, bounded rendered tail. File write errors must remain visible. Use Console injection at Run construction; remove test dependence on per-example handler construction.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_progress tests.test_run -v`; confirm the new lifecycle assertions fail before implementation.
- [x] Implement scoped binding with ContextVar and a shared routing handler attached only to rsikit/research namespaces while runs are active. Track original settings once and restore on last release. Preserve pre-existing application handlers, prevent duplicate ancestor attachment, append UTF-8 run.log records with UTC timestamps, and use standard logging exception formatting. Add a single Live owner per Console; redraw at four Hz only in a TTY.
- [x] Emit the environment metadata record in Rollouts initialization using `environment.spec.id` when present and the environment type otherwise. Do not change collect, seed locks or caching.
- [x] Run the focused commands; require PASS, including existing Run persistence tests.

### Task 3: Integrate EliteSearch and LineageSearch inside their existing loops

**Files:** Modify `research/elitesearch/agent.py`, `research/lineagesearch/agent.py`, `examples/elitesearch.py`, `examples/lineagesearch.py`, `examples/elitelist_papers/run.py`, `tests/test_progress.py`. Existing regression suites: `tests/test_elitesearch.py`, `tests/test_lineagesearch.py`, `examples/elitelist_papers/test_resume.py`, `examples/elitelist_papers/test_tie_break.py`.

**Interfaces:** Class-level `leaderboard_columns` declarations and the spec's log payloads. Existing run/restore/checkpoint signatures remain unchanged. `_checkpoint` remains persistence plumbing; it must not call UI update code.

- [x] Add a scripted test calling EliteSearch.run directly inside Run, without example display setup: assert proposal descriptions appear before generation finishes, evaluation status before the batch ends, custom Operation/Parents cells, exact `_rank` tie order, and combined completion. Include multiple seeds and one repaired policy.
- [x] Extend the existing resume test to assert restored evaluated/discarded slots and leaderboard appear before new provider/evaluator work. Replayed events must not duplicate progress. Preserve the original seed reuse and saved metadata assertions.
- [x] Add a direct LineageSearch.run case covering planning, a family batch and early culling. Assert Family/Operation/Parent columns, attempt-budget total, and unknown progress during planning rather than a false generation percentage.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_progress tests.test_elitesearch tests.test_lineagesearch examples.elitelist_papers.test_resume examples.elitelist_papers.test_tie_break -v`; confirm new integration tests fail before instrumentation.
- [x] Emit events where each agent already mutates candidate/batch state. Start/finish surround the real run method; attempt IDs use native organism/trial IDs and repairs retain them. Emit leaderboard rows from native ranking and batch results. Emit restored state at loop entry, prior to scheduling. Distinguish terminal failure from repair and catch cancellation only to log/re-raise.
- [x] Remove repeated Progress/FileHandler setup and leaderboard printing from the two examples. Preserve checkpoint saves, prior callbacks, leaderboard.json, decomposition.json, summary.json, held-out assessment and video behavior. The paper application inherits this integration without extra display setup.
- [x] Run the focused tests; require PASS with unchanged optimizer outcomes and artifacts.

### Task 4: Integrate AlphaEvolve and ShinkaEvolve through their real loops

**Files:** Create `research/alphaevolve/search.py`, `research/shinkaevolve/search.py`. Modify `research/alphaevolve/original/agent.py`, `research/alphaevolve/paper/agent.py`, `research/alphaevolve/paper/pipeline.py`, `research/shinkaevolve/agent.py`, `examples/alphaevolve.py`, `examples/shinkaevolve.py`, `tests/test_progress.py`. Improved AlphaEvolve inherits shared instrumentation; change its file only if an overridden transition bypasses it.

**Interfaces:** Move existing `run_search` and `run_paper_search` implementations from examples/alphaevolve.py into research/alphaevolve/search.py; move Shinka's `run_search` likewise. Preserve algorithm parameters and return values; remove UI-only console arguments from these functions and update their callers to inject Console through Run instead. Examples re-export the imported function names, preserving their import locations. Move `_check_gym_evaluation` with its loop and import it where still needed. No snapshot callbacks or general search-loop base class.

- [x] Parameterize scripted progress tests over original/improved/paper AlphaEvolve and ShinkaEvolve. Assert direct research-loop invocation inside Run works automatically; validate each column declaration, intermediate proposal visibility and final progress while preserving scores and artifacts.
- [x] Extend paper pipeline tests with two attempts sharing a policy hash, an overlapping later batch, screening rejection and a repair changing source. Assert one resolved slot per attempt, persistent original batch identity, no seed/stage double count and no post-search held-out score leakage.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_progress tests.test_alphaevolve tests.test_shinkaevolve tests.test_paper_pipeline tests.test_paper_agent tests.test_variants -v`; confirm new assertions fail before instrumentation.
- [x] Relocate the actual loops, retaining persistence, objective checks, history grouping and provider/executor behavior. Emit search budgets at full-loop entry; standalone generate calls emit batch/class metadata with unknown run budget. Paper proposal groups use initial attempt allocation groups of evaluation_batch_size; do not regroup based on completion order or generate(1) calls.
- [x] Emit candidate updates at proposal, repair and final optimizer feedback transitions. Use existing attempt/revision records and update every pending attempt sharing a policy hash. Seed events remain diagnostics only. Declare optimizer-owned columns and emit ranked leaderboard rows using native objective/selection rules; do not assume reward when another objective is selected.
- [x] Rehydrate display from existing restored agent/history state at resumed loop entry. Do not introduce another checkpoint file, duration reconstruction or new algorithm resume support.
- [x] Remove old per-example ProgressHandler/log-handler lifecycle and intermediate score-table printing. Keep explicit artifact/report output where still needed.
- [x] Run the focused tests; require PASS, including screening, repair, variant behavior and existing resume coverage.

### Task 5: Finish generic-loop coverage, remove obsolete setup and verify Docker behavior

**Files:** Modify `rsikit/generation/__init__.py`, `examples/inner_loop/__init__.py`, `research/rewards.py` (only supplementary log fields), `rsikit/execution.py` (diagnostics and formatted child traceback), `tests/test_progress.py`, `tests/test_inner_loop.py`, `tests/test_execution.py`, `tests/container_smoke.py`, `README.md`, `research/README.md`, `docs/RUNS.md`. Update relevant optimizer docs only where their usage changes.

**Interfaces:** Generic generation emits the same candidate metadata through logging. The five-policy inner loop supplies its real fixed budget and terminal measurements as domain events. Optimizers own final candidate completion; shared reward/executor events never masquerade as final optimizer assessment.

- [x] Add an inner-loop scripted check proving default activation, five proposal plus five evaluation units despite multiple seeds, and resumed cached evaluation without artificial generation activity. Assert reports and stored episodes are unchanged.
- [x] Extend an executor failure test with an exception inside generated act: assert the diagnostic includes its candidate.py frame and exception chain, excludes local-variable dumps, retains PolicyError classification and reaches run.log. Include a large traceback within the existing payload limit to prove only the visible panel is shortened.
- [x] Run `rtk proxy .venv/bin/python -m unittest tests.test_progress tests.test_inner_loop -v`; confirm the new generic-loop assertions fail before instrumentation.
- [x] Replace inner-loop progress prints with readable structured logs; preserve its report output. Supply genuine timing/task identity only where available. Remove the old counter-based ProgressHandler API once all consumers migrate; retain show_scores only if a real report consumer remains.
- [x] Use traceback formatting without locals in the existing child error message payload, and retain exception information in parent error logs. Preserve transport bounds and error handling. Run `rtk proxy .venv/bin/python -m unittest tests.test_execution tests.test_progress tests.test_inner_loop -v`; require PASS.
- [x] Add a short documented optimizer example showing class-level custom columns and a standard domain log record. State explicitly that Run supplies the display automatically; no Dashboard construction, manual handler attachment or snapshot adapter appears in usage examples.
- [x] Run `rtk proxy rg -n 'ProgressHandler|show_scores|dashboard\.update|with Progress\(' examples research rsikit`; inspect every remaining hit. Poker and explicit reports may remain; standard search entry points must have no UI setup.
- [x] Run `rtk proxy ./scripts/run unittest discover -s tests -t .` and `rtk proxy ./scripts/run unittest examples.elitelist_papers.test_paper1 examples.elitelist_papers.test_resume examples.elitelist_papers.test_tie_break examples.elitelist_papers.test_discrete_actions`. Require PASS; use existing Docker permissions workflow if needed. Do not rerun costly benchmarks for a UI change.
- [x] Exercise `rtk proxy ./scripts/run tests.container_smoke runs/progress-smoke --mode create`, then the same command with `--mode resume`, and `rtk proxy ./scripts/run tests.container_smoke runs/progress-cancel --mode cancel` in a PTY. Choose unused output paths if these already exist. Repeat create with a different path and redirected output. Verify responsive live updates, final terminal restoration, clean plain logs, persistent full traceback and container cleanup. Include a narrow terminal resize during the PTY check.
- [x] Run Ruff on changed Python files and `rtk git diff --check`. Record exact commands/results and distinguish pre-existing failures. Keep pre-existing Docker changes untouched and do not commit the mixed workspace without a separate, explicit request.

## Completion and review

- [x] Recheck every acceptance item in the linked spec against tests or recorded PTY evidence.
- [x] Review the final diff for accidental algorithm changes, per-example UI wiring, duplicate file handlers, unbounded retained display history and new dependencies.
- [x] Follow the execution skill's required final review process, using the agreed scope and actual uncommitted diff; resolve important findings.
- [x] Deliver the updated preview, test evidence, and the unchanged user command to launch a standard run. Implementation was subsequently authorized by the user.

## Implementation results

Implemented on the user-provided `sandbox` branch, without committing the existing mixed workspace. All standard loops use Run-scoped logging automatically. Optimizers own column definitions and rankings; examples contain no dashboard setup.

Validation: 230 core tests passed in Docker after the final review fixes; 18 paper application tests and 22 separate poker tests passed. Real Docker terminal create/resume/cancellation returned 0/0/130, restored the cursor, and handled an 80×24 resize. Redirected execution produced plain logs and normal artifacts. Ruff passed on all changed Python files, and `git diff --check` passed.

A fresh read-only review found five important issues: native paper resume counts, repeated Shinka budgets, settled standalone batches remaining active, unsanitized exception reasons, and missing active evaluations at 80×24. All were reproduced and fixed with regression tests. The reviewer’s minor error-field omission was reclassified as important and fixed so complete structured errors reach run.log. No findings were deferred.

Decisions made during implementation:

- Preserve duck-typed paper generators; AlphaEvolve-specific history instrumentation applies only to AlphaEvolve. External generators must emit their own detailed candidate events.
- Move poker’s unchanged legacy handler into its own display module. Poker keeps its separate behavior and dependencies.
- Run the paper tests by explicit module names because its namespace directory cannot use the original unittest discovery command; this changes no product behavior.
- Restore only known accepted completions from a native paper population checkpoint; detailed Run history supersedes that aggregate. Population-only resume may retain unresolved historical attempts rather than invent their outcomes.
- Automatically settle known-size batches when every slot resolves. This keeps standalone generate/update loops current and relies on producers declaring their actual batch size.
- Keep prior Docker changes, independent poker behavior, detailed events for third-party generators, and new optimizer resume capabilities outside the dashboard review, matching the agreed scope.

Use the existing launcher unchanged, for example `./scripts/run examples.elitesearch --env BipedalWalker-v3` with the usual model and experiment options.
