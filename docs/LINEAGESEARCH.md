# LineageSearch

LineageSearch generates executable approaches across distinct research families,
then uses measured rewards to decide which parents to expand. It ports the typed
global taxonomy decomposition from the sibling MAKER repo's **`extensions` branch,
commit `85e7310`** (`maker/decompose/taxonomy.py` and `recursive.py`) to Slick,
without requiring that checkout at runtime.
An LLM proposes families, experimental hypotheses, and implementations; it never
assigns their fitness.

The pinned levels are **mechanism family → experimental approach → policy**.
Startup elects the family partition, then elects the approaches beneath *every*
family, before generating or evaluating any policy. This initial tree is printed
in Rich and saved as `decomposition.json`. All initial approaches are planned even
when `max_attempts` only permits executing part of the tree; unexecuted plans remain
in the family records and do not count as attempts.

Each family partition and approach partition (including later refinements/pivots)
samples `2k - 1` valid proposals, then uses independent discriminator votes until
one leads its nearest rival by `k`. The default `--decomposition-k 3` requests
five proposals. Discriminators judge coverage, distinctness, consistent taxonomy
levels, ancestry, and experiment quality. They never assign measured fitness.
Malformed proposals receive diagnostic feedback; invalid votes are discarded.
At `--decomposition-max-votes 40`, a contested election falls back to the plurality
(first-voted candidate breaks ties), or the first proposal if there are no valid
votes, matching MAKER's bounded fallback. A single surviving proposal needs no
election. `--decomposition-k 1` disables competing partitions.

This adaptation pins the facets and equal initial quotas from configuration,
uses structured JSON with bounded repair feedback. Independent partitions and
families run concurrently. Discriminator calls run in waves of up to `k`, ingesting
the complete wave before deciding, as in MAKER. All model calls share one global
`--generation-concurrency` limit (default 100), including implementation and repair.
Independent families generate policies while earlier batches evaluate. Evaluation
batches share a single evaluator, whose separate `--concurrency` limit controls
episode workers. Each family completes its batch before choosing its next parents;
the next round and score-weighted bonus batches wait for the current round.
Set generation concurrency to 1 for serial execution. Provider failure or cancellation cancels and awaits
outstanding work before saving the final interrupted state.
MAKER's extension can discover facet schemas and allocate variable quotas; those
are not used here. Fixed family/approach counts conserve the configured initial
quota. Creative policy leaves are implemented once and repaired as needed;
there is no exact-text voting on policies or claim of MAKER's error guarantee.

## Run

Install the repository's `openrouter` extra, build the existing Docker worker,
and set `OPENROUTER_API_KEY` as described in the README. This command makes paid
model calls and executes generated policies in Docker:

```sh
.venv/bin/python -B -m examples.lineagesearch \
  --env CartPole-v1 --families 10 --initial 10 --batch-size 10 \
  --patience 3 --max-attempts 500 --seeds 0 1 2 3 4
```

The terminal uses Rich for decomposition, the attempt budget, completed families,
generation and episode evaluation progress. It displays partition/vote progress,
the initial tree, generated policy names, repair diagnostics,
batch score tables, and a final family status table. The same log messages, including
RSIKit's per-seed results, are retained in `run.log`.
The completed model-call counter advances during planning; the executable attempt
counter starts when implementations are reserved. Generation totals accumulate
across overlapping batches, and both generation and evaluation bars remain visible.
Logs identify the family and
partition being sampled, repaired or elected, even when families overlap.

`--max-output-tokens` defaults to 16,384 per call. It is adjustable for larger
partitions; prompts also request concise fields. For example, 25 families with
25 founders need 625 executable attempts just to finish the initial evaluation,
so a 500-attempt budget leaves no room for refinement.

Defaults explore 100 founders across 10 families. Each unfinished family gets
one batch per round. Two additional batches sample families by descending score
rank, with extra weight for a recent confirmed gain. The finite attempt budget
may stop the search before all families receive enough trials to stagnate.

After all batches in a round finish evaluation and repair, the optimizer pools
previous survivors with newly scored candidates **across all active families**.
`--cull-percent 90` (the default) drops the lowest-scoring 90% globally, keeping
the top 10%, rounded up to at least one. For example, 250 scored founders contract
to 25 survivors regardless of their families. A family that loses every candidate
is marked `culled` and receives no further expansions. Each surviving family samples
a surviving parent for the next decomposition, which generates `--batch-size`
children. Those children compete with previous survivors at the next global cull.
Each optional bonus batch also triggers a global cull. Ties favor older trials.

Parent selection uses descending rank weights; `--exploration 0.2` makes 20% of
selections uniform among survivors. Culled candidates cannot return as parents,
but their source, scores and ancestry remain in the archive and decomposition
evidence. The confirmed incumbent and stagnation checkpoint remain separate from
this score-ranked parent pool, preserving uncertainty-aware progress tracking.
An entirely failed round leaves survivors unchanged; families with no successful
candidates still follow the bounded failure/repair rules. Completed families do
not compete for active survivor slots. `--cull-percent 0` disables culling; values
must be below 100. `--frontier` optionally caps each family's survivors further
(no cap by default). Founder quotas apply until a family produces a measured
incumbent; later expansions use `batch_size`.

Expansion alternates refinements and pivots, forcing a pivot at the patience
limit. The decomposition sees the selected parent's implementation, all family
trial summaries and failure evidence, and sibling family mechanisms. Every child
records its hypothesis, mechanism, change, and intended test. Tests are hypotheses
about the fixed evaluator's measurements, not model-written replacement evaluators.

## Scores and stopping

All scores maximize. Every candidate must have finite rewards for the same seed
panel. A candidate replaces the family incumbent only when its paired mean gain
exceeds `uncertainty * standard_error` (default multiplier 2). Single-seed
evaluation assumes deterministic scores; use multiple comparable cases for noisy
tasks. This margin is a heuristic and does not correct for adaptive multiple testing.

Patience resets only after a confirmed gain also exceeds `min_delta` relative to
the last progress checkpoint. Smaller confirmed gains accumulate against that
checkpoint. A new pivot or a renamed approach never resets patience.

A full expansion batch with at least half its requested trials successfully
measured counts toward stagnation. Complete a family after `patience` such batches
without meaningful progress, with at least one evaluated pivot batch since the
last progress. A full batch with fewer measurements advances a separate failure
counter instead; `patience` consecutive such batches completes the family as
`generation_exhausted`. That status covers invalid/duplicate generation and
unusable candidate executions, not infrastructure outages.

- `completed`: all families are `stagnated`, `culled` or `generation_exhausted`.
- `budget_exhausted`: unfinished families remain when `max_attempts` is reached.
- `generation_exhausted`: family discovery failed its bounded attempts.
- `error` / `cancelled`: the run was interrupted; inspect saved evidence.

Partial batches at the attempt limit cannot establish stagnation. Infrastructure
and provider failures propagate without retiring a family. Generated-output
rejections during search consume attempted slots. For `N = 2k - 1` proposals,
family discovery uses at most `N * discovery_attempts` sampling calls (default
`5 * 3`). Approach decomposition uses at most `N * (1 + max_repairs)` sampling
calls, followed by up to `decomposition_max_votes` discriminator calls. Each of the
N proposals has its own correction allowance, stopping at its first valid result;
unused repairs are not transferred to another proposal. A smaller valid pool can
still be elected when some proposals exhaust their repairs. Only the latest failed
output and diagnostic feed that proposal's correction call. Fresh proposals and
elections receive no rejected-output history; the complete history remains in
`study.calls`. Repairs are instructed to preserve usable entries and correct the
count/fields, returning a complete partition. If no initial approach partition survives, that family
is `generation_exhausted` before any attempt is reserved. Later decomposition
exhaustion rejects that batch's reserved slots and advances the failure counter.

Policies have a shared `max_repairs` allowance (default 2) for malformed JSON,
invalid Python/interface/boundary changes, duplicate programs, and candidate
execution failures. Repairs keep the trial's family, hypothesis and parent, and
must pass the same syntax, boundary and global duplicate checks. Successful siblings
keep their measurements; only repaired candidates are evaluated again. Exhausting
the allowance discards the policy and lets the search continue. Provider or worker
infrastructure failures still propagate. Held-out evaluation never repairs a policy,
since that would feed test information back into search.

Dependency availability is determined by execution in the sandbox, not an import
allowlist. Missing-module errors feed into the same bounded policy repair loop,
which evaluates the replacement and preserves the original failure in history.
The standard image includes NumPy, SciPy, python-control, CVXPY with
OSQP/Clarabel/SCS, scikit-learn and CPU-only PyTorch. Shared prompt guidance
describes useful APIs and CPU/episode constraints during decomposition,
implementation and repair. Slycot-dependent synthesis is not advertised.
Rebuild `rsikit-sandbox:local` after updating these dependencies; the build tests
numerical operations and saves installed versions in `/opt/worker/libraries.json`.

Repair calls do not consume new proposal slots or reset family patience. Their
separate counts and failed versions are saved in each trial's `repairs` and
`revisions` fields; raw responses and diagnostics remain in the study call log.
Use `--max-repairs 0` to disable policy repair and extra approach sampling calls.
Model usage is bounded by `1 + max_repairs` implementation calls per trial, plus
the decomposition and election budgets above. Planning calls do not consume the
executable attempt budget.

The CLI evaluates the final incumbent on disjoint held-out seeds (100–104 by
default), records those scores separately in `summary.json`, and never passes them
back to the optimizer. Search scores remain in the lineage trial records even
though Run also stores the incumbent's held-out episodes. Reusing that held-out
panel to tune later runs would make it validation data rather than a final test.

## Python API

Configure Slick's process-global template root once at application startup:

```python
from pathlib import Path
from slick import prompts
from research import lineagesearch
from research.lineagesearch import Config, LineageSearch, Measurement

prompts.TEMPLATE_ROOT = Path(lineagesearch.__file__).parent / "prompts"

# Inside an async function, with an existing provider and RSIKit Run:
seeds = (0, 1, 2, 3, 4)


async def evaluate(policies):
    await run.evaluate(policies, seeds=seeds)
    return {
        policy.id: Measurement({seed: run.scores(policy)[seed] for seed in seeds})
        for policy in policies
    }


agent = LineageSearch(
    task="Maximize cumulative episode reward.",
    context=environment.instructions,
    provider=provider,
    evaluate=evaluate,
    config=Config(max_attempts=500, min_delta=1.0),
    seed=0,
    on_checkpoint=lambda current: run.save(*current.records()),
)
study = await agent.run()
best_policy = agent.best  # None when nothing could be measured.
```

Use one agent per study. The evaluator returns exactly the requested policy IDs.
`Measurement(scores, feedback="", failure=None)` carries per-seed evidence and
optional textual diagnostics. To reject a broken candidate without aborting
siblings, return `Measurement({}, failure=diagnostic)`; infrastructure errors must
raise. The CLI's `measure()` shows the adapter for RSIKit's `PolicyError.failures`.
Generated implementations are never executed by the optimizer itself.

Programmatic callers supply valid configuration: positive family, batch, optional frontier,
patience, concurrency, discovery, decomposition-k and vote counts; positive generation timeout;
nonnegative attempt/bonus/repair budgets, minimum gain and uncertainty; exploration in
`[0, 1]`. The CLI checks its user-facing ranges. Deterministic evaluator results
and a scripted provider make the search RNG reproducible; real providers can
vary even with the same search seed.

## Evidence and limits

`run.sqlite` contains three additional typed tables:

| Table | Evidence |
| --- | --- |
| `lineagesearch_study` | Task/configuration, phase, stop reason, attempt counts, raw calls/errors, partition candidates, tallies and election outcomes |
| `lineagesearch_family` | Mechanism, initial approach plans, incumbent, progress checkpoint, frontier and completion counters |
| `lineagesearch_trial` | Parent/family IDs, hypothesis, implementation, per-seed measurements, repair counts and failed revisions |

```sql
SELECT f.name, f.status, t.name AS incumbent, t.score
FROM lineagesearch_family f
LEFT JOIN lineagesearch_trial t ON t.id = f.best_id;

SELECT id, family_id, parent_id, kind, hypothesis, score, status, error
FROM lineagesearch_trial ORDER BY id;
```

Exact AST duplicates are rejected globally, including programs changed only by
formatting, comments, or the external policy name. Family and experiment mechanism
labels are checked for exact duplicates within each generated partition. Semantic
diversity remains model-guided: this version does not use embeddings or behavior
clustering, and code changes that preserve behavior can still pass deduplication.

The full archive and raw model output stay in memory and are checkpointed through
Run. This supports inspection, not automatic optimizer resume. Cross-family
crossover and model-weight training are outside this version.

The scripted regression includes an initially weak family that later beats the
early leader, then verifies completion at stagnation. It checks control flow, not
empirical search superiority. To measure value, compare against flat sampling and
the existing optimizers on the same tasks, evaluation panels, proposal limits,
model-call/token budgets, and multiple search seeds. Keep final test cases separate.
