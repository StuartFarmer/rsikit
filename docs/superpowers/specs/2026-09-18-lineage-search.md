# LineageSearch

Approved direction: executable experiments, scored by a fixed evaluator, guide a
diverse search over approach families. MAKER's taxonomy/ancestry pattern supplies
the proposal design; no runtime dependency on the sibling MAKER checkout.

An independent `lineagesearch` optimizer owns generation, family frontiers,
sampling, experiment history, and completion. The caller owns isolated evaluation
and persistence through RSIKit. Python >=3.10; no new dependencies. Prompts live
in the package's `prompts/` directory and use Slick structured outputs.

Generate named families with distinct mechanisms, then a quota of concrete
hypotheses per family. Each experiment records its family, parent, hypothesis,
mechanism, intended test, implementation, measured per-seed scores, and outcome.
Expansion prompts receive family evidence, earlier failures, and sibling coverage.
Reject duplicate proposal mechanisms within a batch and equivalent Python ASTs
across the entire run. Semantic mechanism diversity remains a prompt heuristic.

Each unfinished family gets one batch per round. Extra batches sample families
by score rank and recent confirmed improvement. Within a family, keep a bounded
frontier with the confirmed incumbent and alternatives; an exploration fraction
can sample the full measured archive, including older ancestors. A pivot changes
an assumption within its family and retains its stagnation history.

All scores maximize. The evaluator returns finite per-seed measurements on the
same seed panel. Promote an incumbent only on a positive paired mean difference
exceeding a configurable standard-error margin. Reset patience only when the
gain over the saved progress checkpoint also exceeds `min_delta`; small gains
can accumulate. This is an uncertainty heuristic, not a statistical guarantee
under adaptive multiple testing. Final held-out evaluations never enter search.

Complete a family after `patience` full, valid expansion batches without progress,
including a pivot batch since the last progress. Empty/invalid batches have a
separate generation-exhaustion allowance. Partial batches at the global attempt
limit never establish stagnation. Infrastructure errors propagate and preserve
state; failed candidate executions are recorded without scores. Terminate when
all families complete, or report budget exhaustion separately. No claim of a
global optimum. Save ancestry and all outcomes for inspection; automatic search
resume and learned model weights are outside this version.

Use deterministic scripted-provider tests, including a delayed breakthrough,
noise rejection, cumulative gains, duplicates, partial budgets, failure
propagation, and persistence. Provide a Docker-backed Gymnasium CLI and document
comparison under equal experiment/model budgets; do not claim empirical gains
from scripted tests.

Follow-up: bounded self-repair is part of the runnable search. Default to two
correction calls for malformed decompositions and two repair calls per policy,
shared across malformed generation and execution failures. Preserve failed
revisions and diagnostics; reevaluate only repaired policies, keep successful
siblings' scores, and never repair using held-out results. Expose `--max-repairs`.
The CLI uses the existing Rich progress handler for generation/evaluation and
adds attempt/family progress, repair messages, score tables and file logging.

Follow-up: use the typed global taxonomy adaptation on MAKER's `extensions`
branch (`85e7310`), ported to Slick. Pin levels to mechanism family, experimental
approach, policy. Sample `2k-1` valid partitions, validate quota/duplicate invariants,
and elect by first-to-ahead-by-k discriminator votes with a bounded plurality
fallback. Preserve raw calls, partition candidates, tallies and outcomes. Initial
families and all their approaches must be planned, persisted and displayed before
implementation/evaluation; unexecuted plans survive a smaller execution budget.
Later refinement and pivot partitions use the same election, while executable
measurements alone determine fitness. Initial planning exhaustion retires a family
without consuming executable attempts. No dependency on the adjacent checkout.
