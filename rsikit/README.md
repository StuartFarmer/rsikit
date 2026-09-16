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

All four strategies share `generate()` / `update()` and keep state internally.
`SequentialStrategy` is their abstract base for common single-candidate bookkeeping;
selection and population/archive admission live in the concrete classes.

| Strategy | Search policy |
| --- | --- |
| `HillClimb` | Always propose from the best measured candidate. |
| `AlphaEvolve` | Island/cell champions, exploration, cross-island inspirations, and periodic weaker-half reseeding. |
| `EoH` | Rank-weighted parents; fixed-population E1/E2/M1/M2/M3 cycles; elite selection after a full cycle. |
| `DGMArchive` | Keep all feasible candidates, including regressions; sample by normalized quality and admitted-child count. |

These adapt the local [AlphaEvolve](../alphaevolve/agent.py), [EoH](../eoh/agent.py),
and [DGM](../dgm/agent.py) policies without runtime imports from sibling agents.
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

Tests reuse `tests/providers.py` for scripted Slick calls. The runtime and offline
example have no dependency on that helper. No paid model calls are needed.
