# LineageSearch implementation plan

**Goal:** Implement the approved family-aware experiment search through stagnation.

**Architecture:** A separate optimizer owns the search loop and local Slick
prompts. RSIKit retains ownership of execution and SQLite storage.

**Tech stack:** Existing Python, Slick, Pydantic, SQLModel, Gymnasium dependencies.

**Spec:** `docs/superpowers/specs/2026-09-18-lineage-search.md`

## Constraints

- Python >=3.10; no new dependencies.
- Preserve existing workspace changes.
- Generated code executes only in the caller's evaluator, never in the optimizer.
- Finite, comparable measurements determine fitness. Provider failures propagate.
- Fixed attempt budgets bound generation, including rejected and duplicate trials.

## Tasks

- [x] Add `tests/test_lineagesearch.py` using `tests.providers.ScriptedProvider`.
  Exercise a weak founder whose descendants beat the early leader, then assert
  all families complete. Add plateau/pivot, uncertainty, budget, failure and
  persistence checks. Run `.venv/bin/python -B -m unittest tests.test_lineagesearch -v`
  and confirm the missing optimizer fails.
- [x] Add `lineagesearch/agent.py`, `records.py`, `__init__.py`, and local prompts.
  Public API: `LineageSearch(task, provider, evaluate, *, context, config, seed,
  on_checkpoint)` with `await agent.run()`, `agent.best`, `agent.records()`.
  Evaluator: `await evaluate(policies) -> dict[policy_id, Measurement]`, where
  `Measurement` contains per-seed scores and feedback. Implement named phases:
  discover families, propose experiments, generate policies, measure, update
  frontier and patience, schedule additional work. Use existing policy parsing
  and edit-boundary checks. Re-run the targeted tests.
- [x] Add `examples/lineagesearch.py` and `docs/LINEAGESEARCH.md`; include package
  in `pyproject.toml` and link from README. CLI configures templates/provider,
  uses Docker-backed Run evaluation with fixed search seeds, saves typed records,
  and evaluates the final best on disjoint held-out seeds. Exercise CLI using
  the existing fake sandbox and scripted provider, without API spend.
- [x] Run repository unittest discovery, Ruff lint/format checks, CLI help, and
  inspect the final diff. Report checks and empirical limitations.

## Initial validation

- Repository suite: 96 tests, OK, one existing Docker check skipped.
- Nine LineageSearch tests, including the CLI with a scripted provider/fake sandbox.
- Repository Ruff lint and formatting checks pass; CLI help works.
- Built the wheel with cached Hatchling dependencies (read-only cache access),
  verified all six templates ship, and rendered all five operations from the
  extracted wheel outside the repository.
- Independent read-only review found no consequential issues and additionally
  exercised candidate failure and cancellation handling.
- No paid model experiment or empirical superiority claim.

## Repair and Rich progress follow-up

- [x] Add a shared per-policy repair allowance for malformed generation and
  execution failures; keep failed revisions and repair diagnostics, and reevaluate
  only repaired candidates. Retry invalid decompositions with explicit feedback.
- [x] Reuse the existing Rich generation/evaluation handler, add attempt/family
  progress and score/status tables, and save both optimizer and executor logs.
- [x] Expose `--max-repairs` (default 2), document the bounds, and verify the CLI.
- [x] Verify 14 LineageSearch tests, including malformed JSON followed by runtime
  repair, exhausted repairs with successful siblings, original parent boundaries,
  persisted repair history, and captured Rich output. Scoped Ruff checks pass.
- [x] Address review finding: an invalid replacement must clear the previous
  policy identity; the old identity remains in its saved revision.

The broader suite encountered two unrelated AlphaEvolve resume failures in the
concurrently changed `tests/test_variants.py` (the CLI lacked `--resume` during
that run). Its formatting check also failed. These files were left untouched.

## Typed decomposition and initial tree follow-up

- [x] Verify the reference repo is clean on `extensions`, commit `85e7310`,
  including `TaxonomyGlobalScheme` and the proposal/discriminator control loop.
- [x] Port pinned typed facets, `2k-1` proposal sampling and ahead-by-k voting,
  preserving bounded malformed-output feedback and explicit fallback evidence.
- [x] Plan every initial family and approach before implementing policies; save
  unused plans independently of the executable attempt budget.
- [x] Add Rich planning/vote progress, initial tree display and `decomposition.json`.
- [x] Verify lead-vs-total voting, invalid votes, bounded fallback, planning before
  execution under a small budget, persisted CLI tree, and existing repair behavior.

Validation: 109 repository tests run, 108 passed and one Docker check skipped;
repository Ruff lint and scoped formatting pass; CLI help exposes both decomposition
settings. The wheel contains all nine LineageSearch templates. Independent review
found no actionable issues and checked elected plans/tallies through SQLite.
No paid provider or live Docker experiment was run for this change.

## Decomposition throughput follow-up

- [x] Trace the live 25-family run: serial sampling/voting/planning ignored model
  concurrency, and thirteen failed outputs added 159,076 characters to later prompts.
- [x] Run independent partitions and families concurrently under one shared model
  call limit; ingest discriminator votes in bounded waves as in MAKER.
- [x] Isolate per-proposal repair budgets and keep only the latest failed output
  in correction prompts. Preserve full raw history separately and cancel/await
  nested work on provider failure or cancellation.
- [x] Add exact expected/actual count diagnostics, concise partition output guidance,
  configurable output tokens, and a completed-model-call progress counter.
- [x] Exercise global concurrency, family overlap, feedback isolation, cancellation,
  the initial planning barrier and existing repair/selection behavior in scripted tests.

Validation: 113 repository tests run (112 passed, one Docker group skipped), Ruff
passes, and all ten lineage templates ship in the wheel. Independent review found
no actionable issues. With a scripted 20 ms provider delay, planning 25 families
with 25 approaches made the same 208 calls in 4.851 seconds at concurrency 1 versus
0.410 seconds at concurrency 25 (11.8x); this is orchestration evidence, not a live
provider benchmark. No paid model calls were made.
