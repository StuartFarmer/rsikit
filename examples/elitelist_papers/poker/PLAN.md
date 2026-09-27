# EliteTable poker implementation

Build the approved fixed-stack self-play league: breed from frozen elites,
evaluate challengers and incumbents together, retain the best current scores,
and preserve per-generation evidence. Keep the existing single-policy runner intact.

Use PokerKit 0.7.6 for full no-limit rules, including side pots and short raises.
Run independent table blocks concurrently in Docker. Keep each seat's generated
code in its own persistent process with private observations and a call deadline;
restart processes between duplicate rotations. Batch hands to amortize startup.

- [x] Game adapter, deterministic deals, legal actions, balanced scheduling and scoring.
- [x] Isolated table worker, bounded parallel Docker evaluation, failure attribution.
- [x] EliteSearch adaptation: generation barrier, fresh incumbent scores, complete-field
  scoring with reuse of unchanged blocks after repairs, immutable generation reports, resume and fixed benchmarks.
- [x] CLI, offline demo, docs, correctness checks and measured throughput.

Checks: chip conservation; duplicate seat symmetry; deterministic schedules;
arbitrary population sizes with equal exposure; illegal actions/timeouts;
incumbent reranking; historical scores; repairs invalidate affected table results;
resume; Docker isolation and cancellation; benchmark worker/hand batching.

Implementation stays under this example directory because the shared core files
already have unrelated edits. The approved conversational design is the brief;
no additional approval round is needed for this local implementation.

Validation: 27 poker, Docker integration and existing EliteSearch tests pass;
Ruff passes. The Docker integration includes a two-generation search with scripted
proposals. The real CLI demo completed 240 table hands with 12 players, equal
120-hand exposure and no policy failures. No paid model calls were made.

Independent review found two important issues: the one-chip all-in needed a
distinct action (-1), and Docker cleanup needed bounded waits and named removal
on interrupted launch. Both have regression tests that failed before their fixes.
Resume regressions also cover discarded invalid code and consumed repair budgets.

Performance: preloading reduced a 300-hand block from 24.14 to 5.08 seconds.
An equal-work benchmark measured 62.3 table hands/s with one worker and 222.7
with four. Full settings are saved in benchmark.json; these are single samples.

## Persistent evaluator container (approved architecture change)

Use one lazy-started `TablePool` per CLI run, shared by search and held-out
tournaments. Multiplex requests, telemetry, results and cancellation acknowledgments
over the container's existing stdin/stdout. A clean preloaded forkserver starts
trusted table supervisors and isolated player processes; each table has a killable
process group and distinct player UIDs. Temporary directories are removed before
acknowledging completion/cancellation. No changes to the shared rsikit core.

- [x] Add real Docker regressions for container reuse, concurrent table isolation,
  cancellation followed by reuse, and teardown.
- [x] Replace per-block Docker launches in `tournament.py` with `TablePool` in `pool.py`;
  add `service.py` and adapt `worker.py` process groups, UIDs and telemetry.
- [x] Share the pool across all CLI phases, update Docker packaging and docs.
- [x] Rebuild and run focused unit/Docker checks plus an equal-work timing sample.

Review startup cancellation, stale response IDs, worker crashes, per-block timeout,
and cleanup before reusing capacity. Keep policy failures repairable and infrastructure
failures fatal. Preserve fresh policy instances and Numba kernel reuse per rotation.

Independent review found interrupted cleanup was not retryable and failed Docker
removal could be reported as success. Both have failing-then-passing regression
tests; cleanup now survives cancellation and checks the removal exit status.
The user also requested 32 GiB RAM: this is the default shared container cap,
configurable with `--memory-gb`, including an explicit override on resume.

Final validation: 38 focused unit, shared optimizer and real Docker checks pass,
including a surviving concurrent table during cancellation, a per-block timeout,
subsequent reuse, no surviving player processes, and inspection of the actual
32 GiB memory limit. The same-container timing sample is in persistent-benchmark.json.
