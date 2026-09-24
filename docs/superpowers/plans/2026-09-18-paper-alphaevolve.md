# Paper-based AlphaEvolve Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and review the bounded tasks below.

**Goal:** Implement the published AlphaEvolve mechanisms with a real persistent population.

**Architecture:** Reuse policy generation/repair and isolated RSIKit evaluation. Add a
paper variant backed by a SQLite MAP-Elites/island database, structured evaluation,
and an asynchronous controller. Preserve historical baselines.

**Tech Stack:** Python, sqlite3, dataclasses, asyncio, existing Slick/Jinja and SQLModel.

**Spec:** `docs/superpowers/specs/2026-09-18-paper-alphaevolve.md`

## Global Constraints

- No new dependencies. No live model calls in tests.
- Shared workspace; edit only assigned files. No commits or changes to saved runs.
- Exact database algorithms/defaults are explicitly local choices, not paper claims.

## Tasks

- [x] Database: `alphaevolve/paper/database.py`, `tests/test_population.py`.
  Implement Candidate and Database with persistent evaluated records, per-island
  metric/niche elites, sampling, migration, source deduplication, and state storage.
  Write/run regressions before implementation; verify reopen and diversity behavior.
- [x] Evaluation/controller: `alphaevolve/paper/evaluation.py`,
  `alphaevolve/paper/pipeline.py`, `tests/test_paper_pipeline.py`.
  Implement structured results, threshold cascades, optional feedback graders and
  a bounded producer/consumer pipeline accepting a batched async evaluator callback.
  Verify overlap, pruning, failed policies, cancellation and bounded pending work.
- [x] Agent integration: `alphaevolve/paper/agent.py`, `__init__.py`, prompts,
  `examples/alphaevolve.py`, `alphaevolve/edits.py`, integration/repair tests.
  Bind existing generation to database sampling, structured updates, persistence,
  evaluated seeds, and rich prompts. Use paper default and explicit mode selection.
- [x] Documentation and review: document coverage and local choices in
  `docs/ALPHAEVOLVE.md` and README; run relevant and full offline unittest suites,
  lint, inspect the final diff, and independently review integration.

## Execution record

- User explicitly requested implementation of paper mechanisms, superseding the
  earlier narrow fix. Worktree creation was sandbox-denied; work continues in the
  current workspace. The pre-existing failing duplicate-class regression is retained.
- Paper read directly from arXiv because the alpha CLI is unavailable.
- Database, pipeline and agent reviewed independently. Fixed canonical duplicate
  guidance credit, restored proposal counters/failure context, upfront integer
  validation, final-seed score display, and asynchronous snapshot completion.
- Final suite: 87 tests ran successfully with one Docker smoke class skipped
  (worker image unavailable). Ruff check, format, and git diff checks pass.
  A deterministic concurrency regression verifies revision-scoped failure
  diagnostics survive overlapping generation and repair. No paid search or
  performance reproduction was run.
