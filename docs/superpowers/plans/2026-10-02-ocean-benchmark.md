# Ocean Benchmark Implementation Plan

**Goal:** Implement the approved `docs/OCEAN_BENCHMARK.md` experiment, with runnable correctness, capacity/replay and live-search commands.

**Architecture:** A pinned headless native 2048 binding feeds an experiment-owned panel evaluator. Existing Policy, Executor (reference), Measurement, Run and EliteSearch remain unchanged. Separate example commands measure capacity/replay and perform budgeted live searches.

**Constraints:** Python 3.10+, installed NumPy/Gymnasium, stdlib ctypes/compiler/processes; no new Python dependencies. Keep unrelated user edits intact. No paid model calls during implementation. Vendored upstream source retains license and pin. Cooperative execution only, documented explicitly.

**Execution:** Use subagent-driven-development for disjoint native, benchmark and search tasks; root implements shared panel execution and integration. Work in new task-owned files in the shared checkout; no existing user work is moved, committed or reverted.

## Interfaces

- `research.ocean.native.build() -> Path`: build/cache headless shared library; no network at runtime.
- `Batch(seeds, max_steps=2000)`: `.observations` float32 `(N,16)`, `.active` integer active slot indices; `.step(actions)` consumes actions for active rows; `.results` list of dicts with seed, score, max_tile, return, steps, ending; `.close()`; retain all initial slots in observations. Completed slots never step again.
- `OceanEnv(max_steps=2000)`: Gym env with observation `(1,16)` and action MultiDiscrete([4]); reset seeds native batch, step returns rewards/flags and info containing score/steps/ending. Native construction lazy in reset, so template is pickleable. Preserve reset-observation semantics of upstream.
- `PanelEvaluator(output, mode='batch', workers=1, batch_size=32, max_steps=2000, timeout=60)`: async context manager. `submit(policy, seeds=range(32), job_id=None) -> dict`; `evaluate(policies, seeds=range(32)) -> dict[policy_id, Measurement]`; `.events` job dictionaries; `.queued`, `.running` counters. Modes `reference`, `summary`, `batch`. Each result has `job_id`, `policy_id`, `status` (`ok`/`failed`), `error`, `results`, `steps`, and absolute parent monotonic timestamps `submitted`, `started`, `received`, `persisted`; seconds `queue_seconds`, `service_seconds`, `response_seconds`; worker phase durations when available. JSON measurements and policy source persisted before completion. No cache. Cooperative generated code; deadlines reap process groups.
- `research.ocean.baselines.policies() -> list[type[Policy]]`: eight deterministic row-independent reviewed baseline sources, explicitly not claimed LLM-generated. Support external source files in CLIs.
- `research.ocean.evaluator.rollout(source, seeds, batch_size, max_steps, trace=False) -> dict`: direct async rollout for focused correctness tests; results sorted to requested seed order, optional per-seed action histories, independent of batching.

## Tasks and checks

- [x] Native binding: vendor exact 2048 source and small required headers with license; use headless render shim; ctypes wrapper and Gym adapter. Check native terminal/truncation score capture, explicit seeding, ordering and invalid actions. Native source stays unchanged.
- [x] Panel execution and policies: fresh worker per panel, full-history reference through existing Executor, scalar summary and batch paths, finite shape/range validation, timeout/cancellation cleanup, persisted timings and failures. Test parity, row independence, malformed action, timeout and recovery.
- [x] Benchmark CLI: correctness/capacity/replay modes, measured queue/capacity gates, manifest and event/results artifacts, synthetic or JSON arrival trace, cold/warm separation, diagnostic timings, safe bounded admission with reported backpressure. Test arithmetic and a real tiny replay. Read interface above; own `examples/benchmark_ocean.py` and benchmark tests.
- [x] Search CLI: reuse EliteSearch and Run, independent sampling control with empty elite context, measured/budgeted provider, capped proposal/repair calls, validation and frozen test winner, event trace and generation tail metrics. Test with scripted provider; no paid calls. Own `examples/ocean_search.py` and search tests.
- [x] Integrate: correctness test suite, short real native capacity/replay smoke, appropriate existing suite and Ruff; update specification with actual commands, measured result and limitations. Review source and process boundaries before completion.

## Review focus

State leaking between seeds; native auto-reset replacing final score; external cap cases omitted; failure/queued jobs inflating capacity; retries/cancellation escaping budget or deadline. Each owning task must exercise these cases through behavior tests.

## Rulings

- User's “implement” authorizes this specification and necessary reversible implementation choices; proceed without another design-approval cycle.
- Full training, Breakout expansion and recursive controller evolution are explicitly follow-ups in the specification; implement 2048 and the fixed-search/independent controls now.
- Use reviewed generated-style policies when archived Ocean candidates do not exist. Clearly identify provenance in results.
- Native/source and independent whole-change review completed. Fixed queued-cancellation accounting and external-cap max-tile reporting with regressions; scoped re-review passed. Native game source remains byte-identical to pin.
- Real one-worker, eight-policy/32-seed measurement: 132.99 s reference vs 6.26 s batch, all per-seed results match. This smoke is not the full pilot.
- Final validation: 20 Ocean tests pass, Ruff lint/format pass. Full suite: 257 tests, four existing video/scientific errors documented in the benchmark guide (two worker exits, two missing optional control). No paid model calls or existing-source changes.
