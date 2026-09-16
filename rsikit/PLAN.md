# RSIKit stateful strategy refactor

Approved design: `candidates = await strategy.generate()` followed by
`await strategy.update(candidates, evaluations)`. Each strategy owns its state,
including any history it needs. The caller owns evaluation and the experiment loop.

- [x] Replace the monolithic optimizer with `HillClimb`: incumbent, pending
  candidate, strict selection, and internal feedback history.
- [x] Preserve source edit validation, fixed evaluation results, and the optional
  local script evaluator. Keep Slick generation as an injected callable.
- [x] Move source materialization, stopping, and optional persistence into the demo.
- [x] Replace optimizer tests with checks of generate/update state transitions,
  invalid edits, mismatched results, retry behavior, and cancellation.
- [x] Update usage documentation for internal state and caller-owned execution.
- [x] Run focused tests, offline demo, Ruff, and independent review.

Validation: all 12 focused tests pass, Ruff check/format pass, and independent
review found no actionable issues. The offline demo reduced MSE from 0.0037795926
to 4.2943375e-13 across three supplied revisions and saved its source and history.

No shared history service, strategy base class, mandatory runner, automatic resume,
population implementation, or resource framework. Add these only for actual research
needs. Local evaluation remains for trusted experiments, not a security sandbox.

The earlier repository-wide suite ran 897 tests with eight errors and seven skips.
Those errors concern missing `prompt_optimization` modules/catalogue files and
optional `sklearn`/`metagen` dependencies; none imports RSIKit.

## Bounded repair during generation

Approved extension: compose proposal generation with an independent check–repair
loop before creating the pending candidate. Check initial source and each repair;
stop on validity or the repair budget. Keep repair traces with the proposer.

- [x] Add `repair_until_valid` with explicit results, diagnostics, and bounded repair.
- [x] Add `RepairingProposer`, checking edit boundaries against the original parent.
- [x] Record exhausted proposals as failed search attempts without pending work.
- [x] Preserve infrastructure error/cancellation propagation and external evaluation.
- [x] Document composition and demonstrate syntax repair offline.
- [x] Verify focused tests, demo, Ruff, and independent review.

All 18 focused tests and Ruff checks pass. The offline demo checks a broken
proposal, repairs and rechecks it, then evaluates the repaired source; it saves
separate repair traces and reaches MSE 4.2943375e-13. Independent review found no
actionable issues.

## Complete circle-packing integration experiment

User requested a complete benchmark to run the optimizer against: initial solution,
fixed evaluator, system/task prompts, repair, measurement, selection, and results.

- [x] Add ten-circle unit-square packing, maximizing sum of radii from a baseline of 1.0.
- [x] Keep generated artifacts as a function returning literal circle triples;
  parse without executing generated source and verify all geometric constraints.
- [x] Use OpenRouter GPT-OSS 120B Nitro for proposals and diagnostic-driven repairs.
- [x] Save sources, check/evaluation logs, histories, score summary, and packing SVGs.
- [x] Add offline integration checks for repair, improvement, regression, and failure evidence.
- [x] Finish full focused verification and independent review. Live paid calls are not part of this check.

All 23 RSIKit tests and Ruff checks pass. The CLI help runs, and independent review
found no actionable issues. The circle-packing integration test verifies a repaired
improvement from 1.0 to 1.25 and rejection of a subsequent 1.1 candidate.

## Generation plots and animation

- [x] Save a labeled two-row SVG after every generation, including baseline zero.
- [x] Plot each candidate packing above candidate and best-so-far score lines.
- [x] Render a GIF using final-run axis limits, preserving regressions and labeling rejections.
- [x] Provide a CLI to render saved histories without model calls and document plotting dependencies.
- [x] Verify 27 focused tests, Ruff, independent review, and visually inspect an exported frame.

Backfilled `runs/circle-packing-2cbf2442`: six SVG frames and `progress.gif`, with
fixed score limits spanning the baseline 1.0 through final best 1.38. No model calls.

## Additional search recipes and timestamped runs

User authorized AlphaEvolve, EoH, and DGM-style choices for the same packing task,
and UTC timestamp directory names replacing random suffixes.

- [x] Factor the shared single-candidate lifecycle into an abstract strategy class.
- [x] Add single-objective island/cell search with inspirations and reseeding.
- [x] Add seeded EoH population cycles with all five operators and thought/source output.
- [x] Add explicitly named DGM archive adaptation, including regressions and diagnosis.
- [x] Keep repair, evaluation, and artifacts in the existing packing experiment.
- [x] Record selections/thoughts/diagnoses and label visualizations with strategy names.
- [x] Use UTC timestamps with microseconds in all RSIKit CLI default run names.
- [x] Complete focused checks and independent review; no paid model calls required.

All 36 tests and Ruff checks pass. CLI strategy dispatch and timestamp formatting
were verified without network calls. Review identified a descriptor-callback
half-commit; admission now completes before history/best updates, with a regression
test proving safe retry. Independent review verified the correction.
