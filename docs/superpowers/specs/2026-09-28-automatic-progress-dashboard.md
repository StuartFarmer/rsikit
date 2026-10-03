# Automatic optimization progress dashboard

## Agreed direction

The user supplied a terminal layout and confirmed that Generation Progress covers proposals and evaluations together. Integration must be automatic in the optimization loop, like logging. Application scripts must not construct dashboards, supply snapshot adapters, or call dashboard updates. Each optimizer can declare its own leaderboard columns.

This document records that agreed direction and the concrete implementation decisions below. Poker retains its separate application and display.

## Ownership and lifecycle

- Entering a synchronous or asynchronous `Run` context provides scoped logging and display support; exiting or closing it releases that support. Creating or opening a Run without entering it does not take over the terminal.
- Optimization events activate the display lazily. Reading a saved Run for analysis stays silent. Ordinary logs before activation may still be written to the run log; entering an otherwise unused context produces no terminal output.
- The optimization loop reports domain events using Python logging. A handler in `rsikit/progress.py` consumes the same records used for readable logs. There is no event transport, separate service, UI callback, or second persistence system.
- Use a `ContextVar` to associate asynchronous work with its Run. Logging in concurrent runs must not bleed into another run's file or display. Restore prior logging configuration when the last Run context exits. A single terminal has one live-display owner; additional simultaneous runs use plain output, while keeping separate log files.
- `Run.create(..., console=None)` and `Run.open(..., console=None)` permit an injected Rich Console for existing callers/tests. Default construction is automatic. UI code stays out of the optimizer methods and example scripts.
- Capture only `rsikit` and `research` log namespaces. Do not change root logging, generated child-process output capture, or poker's display.
- Rich runs in the existing application container. The Docker launcher requires no protocol or communication changes.

## Display

The leaderboard top border shows environment, optimizer, completed-generation progress and total elapsed time; its bottom border shows the absolute run directory; separate proposal and evaluation progress bars embedded in their panel titles; full-width leaderboard; proposals and evaluations side by side; event log with errors and tracebacks.

Standard leaderboard columns are Rank, ID, Name, Description, Score and Gen. Optimizer columns follow them. Proposal descriptions wrap; leaderboard descriptions ellipsize. Policy text is literal text, never Rich markup or executable terminal controls. Keep full descriptions and errors in existing saved records/logs.

Render at most four times per second, independent of event frequency, using the available terminal height. Proposals are a table with Time, ID, Name and Description, containing only completed (`generated`) proposals. Evaluations are a table with Time, ID, Name, Score and Duration, containing only completed (`evaluated`) results. Time is the UTC completion timestamp (`HH:MM:SS`). Insert each completion at the top, retain immutable result snapshots in a bounded queue, and discard oldest results from the bottom. In-flight, repair and failure events remain in the event log; they do not create evaluation-table rows. Keep bounded recent proposal/evaluation/log buffers, with all logs written to `run.log`. At widths below 100 columns, stack proposals and evaluations. Unknown score, duration or ETA is `—`; do not invent worker identities.

Assign a random unused color to each policy within the Run, shared by its leaderboard row and proposal/evaluation log messages. Keep that assignment while rows move; no hash-to-color mapping or cross-process assignment persistence. Prefer distinct 256-color swatches, extending with unused RGB colors for longer runs. The Docker launcher advertises 256-color support and forwards host COLORTERM for true-color detection; 256-color terminals can approximate additional RGB colors to the same shade.

Preserve real exception tracebacks, including generated-code frames. The executor currently reduces child exceptions to messages; include formatted exception chains without local-variable dumps in its existing bounded error payload. The dashboard shows a recent excerpt and run.log retains the full received traceback. Do not change error classification, timeout behavior or existing transport size limits.

TTY output uses Rich Live; redirected output uses readable logs without cursor-control sequences. Finishing, errors and Ctrl-C restore the terminal and leave a final summary. Elapsed Time is this invocation's duration; resumed runs are labelled resumed. Historical durations absent from saved records remain unknown.

## Structured logging contract

Use existing loggers and messages with `extra={"progress": payload}`. The payload's `kind` identifies one of the following events. These are in-process dictionaries; Rich Column objects need not be serialized. Existing `event` extras may remain temporarily for other consumers, but the new dashboard counts only `progress` records.

| kind | Required data | Meaning |
| --- | --- | --- |
| `search_started` | `optimizer`, `total_candidates: int or None`, `columns: dict[str, rich.table.Column]`, `resumed: bool` | Declare metadata; repeated declarations update metadata without clearing state. |
| `environment` | `name` | Emitted by Rollouts from environment spec ID, falling back to the environment class name. May precede search start. |
| `batch_started` | `batch_id: str`, `label: str`, `total_candidates: int or None` | Establish an actual generation or stable proposal group. Repeating a batch ID is idempotent. Optional optimizer/columns metadata permits standalone generation to activate the display without overriding an enclosing search budget. |
| `candidate` | `batch_id`, `attempt_id: str`, `revision: int`, `status`, `proposal_done: bool` | Replace current attempt state. Optional `policy_id`, `name`, `description`, `score`, `duration`, `error`, `worker`, `extras`. |
| `leaderboard` | `rows: list[dict]` | Replace ordered leaderboard rows. Each row has `id`, `name`, `description`, `score`, `generation`, `extras`. |
| `batch_finished` | `batch_id`, `status` | Mark completed, stopped, failed or cancelled; never fabricate missing candidate completions. |
| `search_finished` | `status`, `reason` | Freeze search progress and report the outcome. Post-search held-out/video logs can continue. |

Candidate statuses: planned, generating, generated, queued, evaluating, repairing, evaluated, discarded, failed, cancelled. `failed` is terminal only when the loop has decided no repair follows; temporary evaluation failures use repairing. `proposal_done` becomes true once valid source exists or the attempt is permanently discarded/failed; it stays true through runtime repair.

`search_started` may include `completed_candidates_before`, a conservative count of settled historical candidates from the optimizer's existing checkpoint. Exact replayed history supersedes this aggregate rather than adding it twice. Native paper searches with only a population checkpoint recover known accepted completions; they do not assume every historical attempt finished. The application loop also replays its detailed Run history when available.

A known-size batch settles automatically once all its proposal and evaluation slots resolve. This also supports standalone generate/update loops; later explicit batch-finished records remain idempotent. Unknown-size batches require an explicit finish record.

Validate contract fields at the handler boundary. Malformed display payloads yield a readable diagnostic and retain the original log; they must not corrupt counts or crash an otherwise valid search. File logging failures must be surfaced, not silently ignored. Unknown progress kinds remain ordinary readable logs.

## Counts, identity and overlap

- N planned candidates means 2N units: one proposal and one complete policy evaluation each. The confirmed example is `(44 + 26) / (50 + 50) = 70%`.
- Evaluation means the optimizer's complete assessment, including all seeds and screening stages. Per-seed reward events and executor timings never increment candidate counters.
- Key current state by `(batch_id, attempt_id)`, not policy hash. Repairs change revision/source without creating extra work slots. Duplicate source in separate attempts remains separate planned work. Repeated terminal events cannot advance counts twice; ignore stale revisions.
- Evaluated, permanently discarded and terminal failed attempts resolve the evaluation slot. A proposal that cannot be generated resolves its proposal and skipped evaluation slots. Clearly distinguish success, failure and skipped evaluation in displayed status and logs. Cancellation leaves unfinished slots unfinished.
- Retain original batch identity through overlapping work and repairs. The generation panel follows the oldest unfinished batch; total progress covers all planned candidates across the search. Streaming optimizers use a Batch label and stable allocation groups, not a generation reset on every `generate(1)` call.
- Total Run Progress uses the combined units across the declared search budget. Early termination shows its actual percentage and stopped/target-reached status. Unknown budgets remain indeterminate. A zero-work successful search shows complete without division by zero.
- ETA uses recent completed-batch durations for the run and settled candidate durations for a batch. Show unknown until there are at least two relevant observations; reset samples on resume. Combined progress counts work units, not predicted time.

## Optimizer integration and custom columns

Define `leaderboard_columns` once on each optimizer as an ordered dictionary of existing Rich Column objects. Emit it at search start; candidate/leaderboard events provide matching `extras` values. Missing values display `—`; duplicate standard-column keys are rejected with a diagnostic. The dashboard preserves supplied ranking and does not recompute optimizer fitness.

EliteSearch emits events inside `run`, `_generate`, `_measure` and `_promote`. Its columns are Operation and Parents; retain native organism identities for ancestry, not fabricated policy hashes. LineageSearch emits from its existing internal loop and uses Family, Operation and Parent. AlphaEvolve uses Island and Parent; ShinkaEvolve uses Island, Model, Patch and Parents. Actual column values come from existing optimizer records.

AlphaEvolve and ShinkaEvolve currently put their complete multi-generation loops in example modules. Move those existing loops into `research/alphaevolve/search.py` and `research/shinkaevolve/search.py`; example modules import the same functions, preserving their public entry points. These are the real loops, not dashboard adapters. Preserve scoring, repair, persistence and resume semantics. Existing callbacks continue serving persistence only.

Full search loops emit budget/start/finish events. Standalone `generate`/`propose` calls inside an active Run still activate default progress through their own domain events, with unknown overall budget when necessary. A batch-start event may activate the display with optimizer class metadata if no enclosing search-start event exists.

Generic `rsikit.generate` and the inner-loop example use the same contract; their actual five-policy loop declares its work budget through logs. Seed-level measurement is supplementary evidence; the loop consuming the complete measurement emits terminal candidate status. Post-search held-out evaluation must not change completed search progress or replace leaderboard training scores.

Resume re-emits current candidate and leaderboard state from the optimizer's existing restore path before scheduling new work, using the same idempotent events. The handler never reads optimizer-specific database tables. Existing historical runs need no schema migration. Do not add resume support to algorithms that currently lack it.

## Acceptance

All standard optimization entry points display the same layout by running their ordinary loop inside Run; none constructs a Dashboard, attaches Rich handlers or translates state into UI snapshots. New optimizer integration consists of domain log events and optional column declarations, not changes to the renderer. Existing run artifacts, scoring, seed reuse, cancellation and poker behavior remain intact.

Validate with deterministic/scripted providers, no paid model calls. Cover repairs, duplicate sources, multiple seeds, overlapping batches, resume, custom columns, unsafe display text, redirected output and cancellation. Exercise the actual Docker launcher with a terminal and without one.
