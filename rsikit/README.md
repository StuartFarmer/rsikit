# RSIKit — Recursive Self Improvement Kit

Start with a program and a fixed evaluator. A strategy generates candidates and
updates itself from their evaluations. It owns whatever state it needs, including
history; your experiment owns the loop.

```python
from rsikit import Candidate, HillClimb


async def optimize(initial_source, propose, evaluate):
    initial = Candidate(id=0, source=initial_source)
    strategy = HillClimb(
        initial_source,
        await evaluate(initial),
        propose,
        objective="mse",
        maximize=False,
    )
    for _ in range(20):
        candidates = await strategy.generate()
        evaluations = [await evaluate(candidate) for candidate in candidates]
        await strategy.update(candidates, evaluations)
    return strategy.best
```

Here `evaluate(candidate)` is your async adapter returning an `Evaluation`.
It can write the source to a file, submit it to a worker, or measure it directly.
`propose(parent, history)` is an async callable returning replacement source.
There is no state argument, shared history service, or required runner.
This replaces the prototype's `LLMProgramOptimizer.run()` API.

## Complete optimizer benchmark

Use the [circle-packing experiment](examples/circle_packing/README.md) for an
end-to-end integration run with OpenRouter GPT-OSS 120B Nitro. It includes a fixed
ten-circle problem, initial program, geometry evaluator, task/generation/repair
prompts, search strategies, saved results, per-generation SVGs, and a packing/score animation.
One persistent Docker sandbox executes `pack_circles()` once per proposed version;
repairs may submit new versions. NumPy and SciPy are available to generated code.
Scoring and plots reuse the returned circles without executing the function again.

```sh
uv pip install --python optimizer/.venv/bin/python -r rsikit/examples/circle_packing/requirements.txt
docker build -t rsikit-sandbox:local rsikit/sandbox
export OPENROUTER_API_KEY='your-key'
optimizer/.venv/bin/python -B -m rsikit.examples.circle_packing --iterations 5
```

## Try it offline

From this checkout, using the existing environment:

```sh
optimizer/.venv/bin/python -B -m rsikit.examples.sine.demo
```

The [demo](examples/sine/demo.py) evaluates the [initial program](examples/sine/initial.py)
and three supplied Taylor-polynomial revisions with a [fixed evaluator](examples/sine/evaluate.py).
The first proposal contains a deliberate syntax error; a supplied repair fixes it
before fitness evaluation. The demo makes no model calls. The experiment writes
candidate files, evaluator logs, `best.py`, `history.json`, and `repairs.json` to a
fresh directory under `runs/`.
Use `--output PATH` to select a new directory. Persistence is ordinary example code;
the strategy never creates files. Automatic resume is not implemented.

The evaluator accepts arithmetic expressions and measures sine approximation MSE
over a public grid. This is a working harness, not a generalization benchmark.

## Try OpenRouter GPT-OSS 120B Nitro

The [OpenRouter example](examples/sine/openrouter.py) uses Slick's direct
`OpenRouterAPI` provider with model
[`openai/gpt-oss-120b:nitro`](https://openrouter.ai/openai/gpt-oss-120b:nitro).
The `:nitro` suffix requests OpenRouter's
[throughput-oriented routing](https://openrouter.ai/docs/guides/routing/model-variants/nitro).

From this checkout, install the provider's optional SDK in the existing environment,
set your key, and run:

```sh
uv pip install --python optimizer/.venv/bin/python 'openai>=2,<3'
export OPENROUTER_API_KEY='your-key'
optimizer/.venv/bin/python -B -m rsikit.examples.sine.openrouter --iterations 3
```

This makes paid model calls using your OpenRouter account. Both proposals and
repairs use the selected model. To deliberately exercise the repair path in a
short run:

```sh
optimizer/.venv/bin/python -B -m rsikit.examples.sine.openrouter --repair-demo --iterations 1
```

That option supplies a broken first proposal (`return x - x**3 /`); the model
repairs it from syntax diagnostics. Later iterations, if requested, use model
proposals normally. Without the option, repair happens only when a model proposal
fails its edit-scope or syntax check. The fixed evaluator subsequently checks the
arithmetic interface, finite outputs, and MSE; those later failures are recorded
as invalid candidates, without a second repair loop.

The terminal prints baseline MSE, each generation's repair count and score, and
the best score so far. A fresh `runs/rsikit-openrouter-*` directory contains the
candidate sources, evaluator logs, `best.py`, `history.json`, and `repairs.json`.
`--output PATH` selects a new directory. There is no guarantee of improvement.

Defaults: three generations, up to two repairs per generation (at most nine model
calls), 8,192 output tokens per call, and a 180-second deadline for each entire
generation including repairs. Change these with `--iterations`, `--max-repairs`,
`--max-tokens`, and `--timeout`; `--model` accepts another OpenRouter model ID.
Transport retries are disabled. Error/cancellation preserves completed histories
and the current best source. Partial repair traces need callback logging as
described below. Local execution uses the restricted arithmetic evaluator and
process timeout; it is not an OS sandbox for arbitrary programs.

## Strategy contract

All five strategies share `generate()` / `update()` and keep state internally.
`SequentialStrategy` is their abstract base for common single-candidate bookkeeping;
selection and population/archive admission live in the concrete classes.

| Strategy | Search policy |
| --- | --- |
| `HillClimb` | Always propose from the best measured candidate. |
| `AlphaEvolve` | Island/cell champions, exploration, cross-island inspirations, and periodic weaker-half reseeding. |
| `ShinkaEvolve` | Weighted island parents, bounded novelty resampling, adaptive model selection, and periodic meta recommendations. |
| `EoH` | Rank-weighted parents; fixed-population E1/E2/M1/M2/M3 cycles; elite selection after a full cycle. |
| `DGMArchive` | Keep all feasible candidates, including regressions; sample by normalized quality and admitted-child count. |

These adapt the local [AlphaEvolve](../alphaevolve/agent.py), [EoH](../eoh/agent.py),
[DGM](../dgm/agent.py), and [ShinkaEvolve](../shinkaevolve/agent.py) policies without
runtime imports from sibling agents.
AlphaEvolve here uses one configured objective and serial generation, without
ensembles, meta-prompt evolution, or evaluation cascades. EoH starts with the
provided seed and fills the remaining population. `DGMArchive` adapts archive
search only; it does not execute evolving agents or test self-modification.
See [NOTICE](NOTICE) for attribution and the [packing comparison commands](examples/circle_packing/README.md#compare-strategies).

Proposal callables retain the `(parent, history)` signature. They can additionally
read `strategy.context`: the selected `operation`, `parents`, and optional
`inspirations` / `island`. The packing example uses this to dispatch prompt methods.

```python
from rsikit import EoH


async def propose(parent, history):
    return await generator(parent, history, context=strategy.context)


strategy = EoH(initial_source, baseline, propose, objective="score", population_size=4)
```

Here `generator` is your operation-aware callable. `strategy.selections` records
completed attempt IDs, operations, and parent/inspiration/island choices. Concrete
state is exposed as `AlphaEvolve.islands/events`, `EoH.population/offspring/cycles`,
and `DGMArchive.archive`. The global `best` updates immediately, including during
partial EoH cycles. `DGMArchive.score_bounds` maps the objective to `[0, 1]` for
sampling; minimizing reverses quality. Configure bounds appropriate to your task.

`HillClimb` starts with source and its already measured, valid evaluation. It owns
`best`, `history`, and `pending`. Its history contains `Candidate` records with
source, ID, parent ID, and evaluation; generated candidates initially have no
evaluation. IDs are local to the strategy instance.

- `await generate()` returns a list containing one candidate. Blank, unchanged,
  or out-of-scope edits are recorded as invalid and return an empty list.
- `await update(candidates, evaluations)` incorporates the matching batch.
  Strict improvements replace the incumbent; ties, regressions, and invalid
  evaluations remain in history without replacing it.
- Use one operation at a time per instance. Update a generated batch before
  generating again. Empty updates are allowed only when nothing is pending.
- Mismatched updates and missing objective metrics raise without changing state.
  Explicit `ProposalRejected` exceptions record a failed attempt and return `[]`.
  Other generation exceptions and cancellation propagate without admitting a
  candidate or changing completed history. Sampling can advance RNG and context.
- Evaluation happens outside the strategy. If it fails, the pending candidate
  remains available for retry, or you can update with an explicit invalid result.

The caller controls stopping, generation deadlines, retries, scheduling, and
logging. This strategy imposes no generation timeout. For example, wrap its call
in `asyncio.wait_for(strategy.generate(), timeout=120)` when needed.
Other strategies can own different state and generation logic without changing
this loop. The provided abstract base is optional for independent strategy implementations.

## ShinkaEvolve

`ShinkaEvolve` owns the archive and measured search decisions. `ShinkaProposer`
owns separate diff, full-rewrite, crossover, novelty-judge, and meta-scratchpad
prompts. The host still evaluates candidates and calls `update()`:

```python
from pathlib import Path
import rsikit
from slick import prompts

prompts.TEMPLATE_ROOT = Path(rsikit.__file__).resolve().parent.parent


def make_shinka(initial, baseline, task, provider, embed=None, ensemble=()):
    operations = rsikit.ShinkaProposer(task, provider, ensemble=ensemble)

    async def propose(parent, history):
        return await operations(parent, history, context=strategy.context)

    strategy = rsikit.ShinkaEvolve(
        initial, baseline, propose,
        objective="score", maximize=True,
        models=len(operations.models),
        embed=embed,
        novelty=operations.assess_novelty,
        reflect=operations.summarize,
    )
    return strategy, operations


async def optimize(strategy, evaluate, iterations=100):
    for _ in range(iterations):
        candidates = await strategy.generate()
        evaluations = [await evaluate(candidate) for candidate in candidates]
        await strategy.update(candidates, evaluations)
    return strategy.best
```

The initial `baseline` is an already measured RSIKit `Evaluation`. Its metrics and
feedback, like all RSIKit evaluations, are public prompt context; keep private and
held-out measurements outside these records. Sources use RSIKit's existing edit
boundary convention, including tolerance for one final LF/CRLF.

Default parent weights are sigmoid fitness relative to the island median, divided
by one plus the parent's submitted offspring count. `parent_selection="power"`
uses fitness rank to `-power_alpha`; `"uniform"` and `"best"` are also available.
All policies support `maximize=False`. Each island retains at most `archive_size`
programs: top elites plus random stepping stones. Ring migration copies
nonchampions from a snapshot; it never exports an island champion.

`context` includes the selected `model`, `patch` (`diff`, `full`, or `cross`),
parents, inspirations, objective/direction, guidance, and retry diagnostics.
Use `models=len(operations.models)` when supplying an ensemble. `ShinkaProposer`
selects that provider; providers may have different decoding settings. Repairs
use the base provider; `novelty_provider` and `meta_provider` optionally separate
the auxiliary models. No providers are created by the library.

Model rewards use improvement above the better of the parent and initial score.
The same pooled-maximum normalization before the exponential reward as the
[standalone example](../shinkaevolve/README.md) keeps rewards bounded and invariant
to positive fitness rescaling. UCB scores become sampling weights; untried models
go first. Every completed attempt, including a rejected generation, counts once.
Failed/non-improving attempts receive zero reward. Infrastructure failures and
cancellation propagate without committing history or model credit.

Supply async `embed(mutable_source) -> Sequence[float]` for embedding filtering.
Vectors are cached, normalized, and compared within the selected island. Above
`novelty_threshold`, async `novelty(source, nearest_candidate, similarity)` returns
`(accepted, reason)`; `operations.assess_novelty` implements the LLM judge. Without
a judge, high-similarity proposals are rejected directly. Without embeddings,
only exact duplicates are filtered. There is no implicit embedding service.

`max_proposals` bounds source-generation calls per `generate()`; rejected edits,
novelty failures, and invalid generated judge JSON share that allowance. Exhaustion
records one invalid history entry and returns `[]`. No rejected proposal reaches
the external evaluator. If you compose `RepairingProposer`, use syntax/interface
checks that do not execute candidates when evaluation savings matter; the novelty
gate runs after that proposer returns its repaired source.

When configured, meta reflection runs before generation after every
`meta_interval` completed attempts. Thus stopping does not buy an unused final
reflection. `operations.summarize(recent, previous, objective=..., maximize=...)`
receives recent attempts plus their parents and the seed, and replaces the scratchpad with
at most `max_recommendations` items. Malformed generated recommendations preserve
the previous scratchpad; infrastructure failures propagate. Meta calls have their
own records and do not earn mutation-model reward. Scratchpad/cached embeddings
may update even if a subsequent provider call fails; completed search history does
not. A retry reuses already completed meta analysis.

Inspect `islands`, `offspring`, `model_gains`, `scratchpad`, `proposals`, and
`events` alongside the usual `history`, `pending`, and `selections`.
`operations.records` retains raw model calls and parsing failures. Improvement
history uses `Decimal` to avoid overflow before normalization; serialize it as
strings if needed. State remains in memory, and all timeouts and call budgets
belong to the host.

## Repair during generation

Wrap a proposer to check and repair its source before `generate()` returns:

```python
from rsikit import HillClimb, RepairingProposer


def make_strategy(initial_source, baseline, propose, check, repair):
    proposer = RepairingProposer(propose, check, repair, max_repairs=3)
    return HillClimb(initial_source, baseline, proposer)
```

`check(source) -> Evaluation` and `repair(source, evaluation) -> str` are async
callables. The check reports diagnostics in `feedback`; the repair returns the
complete revised source. Supply syntax checks, interface checks, or tests for your
task, and a repair callable implemented with Slick or another tool. Include any
fixed task context or original-source reference the repairer needs in that callable.
Checks that execute code own their execution isolation, just like evaluators.

The wrapper checks blank/unchanged source and edit boundaries against the original
parent before every task check. Each repaired revision is checked again. With
`max_repairs=3`, there are at most three repair calls and four checks; zero means
check only. It stops at the first valid result. Check metrics never select the best
candidate: fitness evaluation still happens after generation.

An explicit invalid check triggers repair. Infrastructure exceptions and
cancellation propagate; they are not treated as source bugs. Deadlines remain with
the caller. On exhaustion, `HillClimb` records the final failed source and diagnostic
as one search attempt, returns `[]`, and leaves no candidate pending.

`RepairingProposer.history` holds completed repair runs, including exhausted runs.
Each `RepairResult.attempts` contains the source and check result for every checked
revision. These traces are separate from the strategy's candidate history. Interrupted
calls do not append a completed trace; callers needing partial failure logs should
record them in their check/repair callbacks.

The same loop works independently of any proposer or strategy:

```python
from rsikit import repair_until_valid

# Inside your async function:
result = await repair_until_valid(source, check=check, repair=repair, max_repairs=3)
checked_source = result.require_valid_source()  # Raises RepairExhausted on failure.
```

The standalone loop runs exactly the checks you supply; the proposer wrapper adds
edit-scope checks. Passing establishes only those checks. Bugs found later during
fitness evaluation remain ordinary invalid evaluations; this flow does not modify
an already pending candidate or repair valid candidates merely for a low score.

## Connect Slick

`Proposer` is the abstract base class for proposal implementations. Subclasses
implement `async __call__(parent, history) -> str`; the base class imposes no
provider, prompt, or history-window policy. `HillClimb` also accepts plain async
callables, as used by the offline demo.

Use the concrete `SlickProposer(Proposer)` as the `propose` argument above:

```python
from pathlib import Path

import rsikit
from slick import prompts


def make_proposer(provider):
    prompts.TEMPLATE_ROOT = Path(rsikit.__file__).resolve().parent / "prompts"
    return rsikit.SlickProposer(
        "Improve approximate(x) on [-1, 1]. Minimize mean squared error. "
        "Use one return expression containing only x, numbers and arithmetic.",
        provider,
    )
```

The proposer includes the incumbent source and eight recent outcomes in each
independent model call. Describe the objective, direction, and program interface
in the task. No model is selected implicitly. Slick's template root is currently
process-global; agents using different roots need separate processes.

Runtime dependencies are Slick, Pydantic, and the standard library. No sibling
agent is required. In this checkout, use the adjacent Slick source as described
in the root README.

## Reusable API-only improvement modules

All model weights stay fixed. The editable state is instruction text, selected
context, measured reflection, and the host's candidate archive. The existing
`generate()` / `update()` loop and two-argument proposal callback still apply.

| Building block | Responsibility |
| --- | --- |
| `PromptProposer` | Separate initialization, crossover, mutation, simplification, inspiration, diagnosis/modification and repair prompts |
| `recent_context(history, limit=8)` | Explicit recent-outcome selection |
| `ReflectionMemory` | Bounded guidance from measured pairs or failed attempts |
| `better`, `top_candidates`, `rank_parents`, `lineage_weights` | Reusable score comparison, elite survival and parent selection |
| `PromptSearch` / `PromptTrial` | Evolve an instruction through fresh downstream comparisons |

For these **new** templates, configure Slick's root to the checkout root once
at application startup. Existing `SlickProposer` keeps its original root above.

```python
from pathlib import Path
import rsikit
from slick import prompts

prompts.TEMPLATE_ROOT = Path(rsikit.__file__).resolve().parent.parent

def make_strategy(initial, baseline, task, provider):
    operations = rsikit.PromptProposer(
        task, provider, instructions={"mutate": "Prefer small, testable changes."}
    )

    async def propose(parent, history):
        return await operations(parent, history, context=strategy.context)

    strategy = rsikit.HillClimb(initial, baseline, propose)
    return strategy, operations
```

The same closure works with `EoH`, `AlphaEvolve` or `DGMArchive`. Operation names
are `mutate`, `INIT`, `E1`, `E2`, `M1`, `M2`, `M3`, `alphaevolve`, and
`dgm-archive`. Instructions are plain text, never evaluated as templates. The
trusted task, edit boundaries and output schema stay outside the editable block.
Generation returns `Draft(description, source)` internally and source to the
strategy. `operations.records` retains rendered prompts, raw responses, lineage
and errors, including failed parsing. Invalid generated drafts become rejected
attempts; provider failures propagate. `operations.repair` composes with the
existing `RepairingProposer`.

After `update()`, optionally call
`await memory.observe(parent, completed, objective=strategy.objective,
maximize=strategy.maximize)` and supply `context["guidance"] = memory.texts` on
the next proposal. Valid unequal scores produce pair reflections; invalid attempts
with diagnostics produce failure reflections. Ties use no API call. Completed
guidance is bounded by `max_items`; `attempts` retains the full call trace. Each
independent trial starts with fresh memory. Reflection errors propagate after the
candidate update; a failed reflection does not undo measured search progress.

Selection helpers consume valid measured candidates. For example,
`top_candidates(population, 4, objective="error", maximize=False)` returns the
four lowest-error candidates. They do not replace archive admission policies:
EoH keeps cycle-end elites, DGM retains feasible stepping stones, and AlphaEvolve
keeps champions in distinct cells.

### Evolve one instruction

```python
search = rsikit.PromptSearch(task, provider, evaluate_instruction)
best = await search.run(
    "Prefer small, testable changes.",
    development=("case-a/repeat-0", "case-a/repeat-1"),
    selection=("case-b/repeat-0", "case-b/repeat-1"),
    revisions=3,
)
```

The application supplies `async evaluate_instruction(instruction, cases)`, returning
`PromptTrial(case_ids, utilities, feedback)` in exactly the requested order.
Utilities are finite and higher is better; negate a minimization objective.
Include failed outputs using a declared finite penalty rather than dropping them.
Construct fresh strategy/proposer/memory state for every independent case/repeat.

Each revision must strictly beat a freshly measured incumbent on development
**and** selection. Evaluation order alternates. Selection feedback never enters
the revision prompt; callers supply disjoint cohorts and withhold final tests.
Blank/unchanged revisions and ties retain the incumbent. Incomplete comparisons
cannot promote; exceptions preserve `best`, `history` and partial `measurements`.
This is measured instruction hill climbing, not a full reproduction of
Promptbreeder or GEPA.

Run the complete offline composition from the checkout:

```sh
optimizer/.venv/bin/python -B -m rsikit.examples.prompt_search --output /tmp/prompt-search.json
```

It reuses the packing geometry evaluator and parses literal source without executing
it. Scripted outputs demonstrate a development winner that loses selection, followed
by a winner on both. The cases are independent repeats of the same packing task;
this checks the decision plumbing, not generalization or live model quality.

The example owns `BudgetedProvider`: every generation, repair, reflection and
instruction-revision dispatch consumes the same allowance, including failed API
calls. Per-trial caps include repair and reflection. Before a pair, capacity is
checked for the worst-case complete development and selection comparison.
Exhaustion saves progress and retains the accepted instruction. Transport retries
must be disabled; hidden provider retries are not counted. Token/currency usage is
saved as unknown. JSON also records model/settings, task/evaluator identity, case
IDs, instructions, raw calls, candidate/check evidence and partial comparisons.

For live comparisons, use the same model, decoding settings, evaluator and resource
allowance for static instructions, static+reflection, evolved instructions and
existing EoH operators. Record actual resource use. Evaluate the selected instruction
once on untouched final cases; never reuse selection/final evidence in reflection
memory construction for development trials. Applications still own isolation,
timeouts and persistence. Templates are configured for checkout use; installed
package loading is not provided by this increment.

## Evaluation contract

```python
from rsikit import Evaluation

evaluation = Evaluation(
    valid=True,
    metrics={"mse": 0.0012, "runtime_seconds": 0.03},
    feedback="All required cases passed; most error is near the endpoints.",
)
```

Metrics must be finite numbers. Every valid evaluation must include the strategy's
objective. `maximize=False` minimizes it; other metrics are retained as diagnostics.
Return `Evaluation(valid=False, feedback="...")` for invalid candidates, and raise
for infrastructure failures. An invalid candidate cannot win on score. Repeated
measurements and aggregation belong in the evaluator. Keep held-out answers out
of feedback and the fixed evaluator and its data outside candidate control.

`LocalEvaluator` is an optional adapter for a fixed script:

```python
from pathlib import Path
from rsikit import LocalEvaluator

evaluate_file = LocalEvaluator(Path("evaluate.py"), timeout=30)
# After your experiment writes candidate.source to a unique candidate_path:
# evaluation = await evaluate_file(candidate_path)
```

It invokes the script with absolute paths:

```sh
python /absolute/evaluate.py --program /absolute/candidate.py --output /absolute/evaluation.json
```

The script writes JSON and exits zero:

```json
{"valid": true, "metrics": {"mse": 0.0012}, "feedback": "Measured on the evaluation set."}
```

The working directory is the candidate's directory. Use a separate directory per
candidate; locate fixed assets relative to the evaluator's own `__file__`.
Standard output and error go to log files. Nonzero exit, missing output, or invalid
JSON raises `EvaluationError`; timeout returns an invalid evaluation. Cancellation
terminates the process group and propagates. Previous evaluation output is removed
before execution so stale results cannot count as fresh measurements.

**LocalEvaluator is not a security sandbox.** It runs on macOS/Linux with the
current user's filesystem, network, and environment access. Process cleanup does
not isolate permissions. Use it for trusted experiments; arbitrary generated code
needs an evaluator backed by an appropriately isolated container or remote worker.

## Edit scope

An unmarked file is fully editable. Optional `EVOLVE-BLOCK-START` and
`EVOLVE-BLOCK-END` comment lines restrict edits to their contents. Multiple blocks
are supported; markers must be balanced and non-nested. Surrounding text and marker
lines must remain identical, except for an optional final LF/CRLF. The validator
ignores that final line ending when comparing sources and preserves the original
candidate text for evaluation. These are lexical boundaries, not a sandbox or syntax
checker. The evaluator checks the program's actual task contract.

## Development

```sh
optimizer/.venv/bin/python -B -m unittest discover -s rsikit/tests
../slick/.venv/bin/ruff check rsikit
../slick/.venv/bin/ruff format rsikit --check
```

Tests and the offline prompt-search demonstration reuse `tests/providers.py` for
scripted Slick calls. The library has no dependency on that helper. No paid model
calls are needed.
