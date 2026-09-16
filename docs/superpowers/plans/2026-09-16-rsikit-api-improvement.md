# RSIKit API-only Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prompt operations, context, reflection and measured instruction evolution reusable in RSIKit while every LLM remains a fixed API model.

**Architecture:** Preserve the experiment-owned `generate()` / `update()` loop and existing proposal callback. Add a reusable prompt proposer and optional reflection memory, extract concrete selection decisions, and evaluate editable instruction variants in an explicit outer experiment. No training layer or universal agent runtime.

**Tech Stack:** Existing Python, Slick, Pydantic, Jinja and unittest; current injected API providers and task evaluators.

**Spec:** [API-only improvement design](../specs/2026-09-16-rsikit-api-improvement-design.md). Also read the [mechanism investigation](../../../outputs/rsikit-improvement-modules-comparison.md) for donor distinctions.

## Global constraints

- Use fixed pretrained models through injected API providers; no model-weight updates, fine-tuning, soft-prompt embeddings, gradient access or training backends.
- Keep `generate()` / `update()` and the existing two-argument proposal callback compatible.
- Keep task evaluation, execution isolation, run persistence and total budgets application-owned.
- Keep the trusted task objective, evaluator, edit boundaries and output schema outside the editable prompt block.
- Use existing Python, Slick, Pydantic and Jinja dependencies; add no dependency for this work.
- Keep prompt operations in separate local Jinja files; select operations in Python and do not put branches in Jinja.
- Preserve existing strategies' parent sampling, RNG order, admission rules and update timing during extraction.
- Keep fixed model identity and decoding settings within each prompt comparison; record model/settings and rendered prompt provenance.
- Count proposal, critique, repair and prompt-revision calls against the same application-owned call allowance.
- Default new examples to offline scripted verification; live API comparisons are explicit runs, never part of the tests.

## Delivery map

| Increment | Tasks | Working result |
|---|---|---|
| A. Reusable prompts | 1–2 | Existing strategies can use one task-independent, operation-aware proposer |
| B. Evidence-guided proposals | 3–4 | Optional reflection and independently reusable selection/admission decisions |
| C. Prompt improvement | 5–6 | A fixed API model proposes instruction revisions; downstream trials decide whether to retain them |

Execute sequentially. There is no need for parallel implementation or a repository-wide refactor. Commit only the reviewed task's files when integrating; `rsikit/` was untracked during planning, so do not use `git add .` or absorb unrelated work.

## Files and ownership

| Path | Responsibility |
|---|---|
| `rsikit/proposer.py` | Keep existing proposers; add `Draft`, `PromptProposer`, recent-context selection and proposal traces |
| `rsikit/prompts/operations/*.j2` | Task-independent operation templates and shared context envelope |
| `rsikit/reflection.py` | New bounded run-local evidence-to-guidance helper |
| `rsikit/prompts/reflection/*.j2` | Pair comparison and explicit-failure reflection |
| `rsikit/selection.py` | New pure helpers extracted from concrete strategies where reuse is real |
| `rsikit/population.py` | Use extracted helpers while retaining each recipe's sequencing/state |
| `rsikit/prompt_search.py` | New prompt-trial records and explicit measured instruction search |
| `rsikit/prompts/prompt_search/revise.j2` | Revise one plain-text instruction using development evidence |
| `rsikit/examples/circle_packing/experiment.py`, `__main__.py`, `README.md` | Opt-in use of generic prompts/reflection; preserve legacy baseline |
| `rsikit/examples/prompt_search.py` | Offline integration example, experiment-owned root/per-trial call allowance, saved comparison records |
| `rsikit/tests/test_prompt_components.py`, `test_prompt_search.py` | New focused behavior checks using existing `tests.providers.ScriptedProvider` |
| `rsikit/tests/test_population.py`, `test_circle_packing.py` | Extend existing checks only where integration changes |
| `rsikit/__init__.py`, `README.md` | Public exports and runnable examples |

No new package per conceptual module. `Candidate`, `Evaluation`, `RepairingProposer`, source validation and execution adapters remain the existing boundaries.

## Task 1: Reusable prompt operations and explicit editable text

**Files:** modify `rsikit/proposer.py`, `rsikit/__init__.py`; create `rsikit/prompts/operations/context.j2`, `mutate.j2`, `initialize.j2`, `diverse.j2`, `shared.j2`, `structure.j2`, `settings.j2`, `simplify.j2`, `inspirations.j2`, `diagnose.j2`, `modify.j2`, `repair.j2`; create `rsikit/tests/test_prompt_components.py`.

**Interfaces:**

```python
class Draft(BaseModel, extra="forbid"):
    description: str
    source: str

def recent_context(history: tuple[Candidate, ...], *, limit: int = 8) -> tuple[Candidate, ...]:
    return history[-limit:] if limit else ()

# New class, existing SlickProposer remains unchanged.
PromptProposer(task, provider, *, instructions=None, history_size=8)
await proposer(parent, history, context=None)     # -> str; context is keyword-only
await proposer.repair(source, evaluation)        # -> str, for RepairingProposer
proposer.records  # list[dict]: attempted calls and completed drafts
```

`instructions` maps operation names to editable plain text, defaulting to empty additional guidance. Existing static operator intent and task/output contracts remain in templates. Context keys: `operation`, `parents`, `inspirations`, `island`, `guidance` (tuple of reflection strings), and `evidence` (application-supplied display data). Normalize missing optional context values in Python, not Jinja. Do not mutate caller dictionaries.

- [ ] Add one async scripted-provider check: an E2 request receives both parents, custom instruction and task; returns only the draft source; records description, operation and parent IDs. Add the corresponding invalid-draft assertion to the same focused test group.
- [ ] Run `rtk proxy optimizer/.venv/bin/python -B -m unittest rsikit.tests.test_prompt_components -v`; confirm failure because the new proposer does not exist.
- [ ] Implement named decorated methods and a Python operation map:

```python
operations = {
    "mutate": self.mutate,
    "INIT": self.initialize,
    "E1": self.explore_diverse,
    "E2": self.explore_shared,
    "M1": self.modify_structure,
    "M2": self.tune_settings,
    "M3": self.simplify,
    "alphaevolve": self.use_inspirations,
}
# dgm-archive explicitly calls diagnose, then modify.
```

Each method uses its own external template and `output_type=Draft` where appropriate; diagnosis and repair remain prose/source. Use repository-root-relative names such as `rsikit/prompts/operations/shared.j2` for all new decorators/includes. Configure the repository root once in new examples/tests; do not change existing decorators or existing consumers' root choice. Validate generated nonblank source and description in the generated-output boundary. Blank/malformed drafts become explicit `ProposalRejected`; provider failures propagate. Preserve source bytes. `repair` consumes failed source and `Evaluation.feedback`, returns source only, and performs no evaluation; the existing wrapper checks original-parent edit boundaries.

- [ ] Render the shared context envelope once per selected operation. Its editable block is ordinary text:

```jinja2
Task and required interface:
{{ instance.task }}

Additional method instruction:
{{ instruction }}

Measured parent and inspiration records:
{{ parents | tojson }}
{{ inspirations | tojson }}

Prior outcomes and attributed guidance:
{{ outcomes | tojson }}
{{ guidance | tojson }}
{{ evidence | tojson }}

Return JSON matching this schema:
{{ schema | tojson }}
```

Add fixed source/interface/edit-envelope instructions from the current proposer to the generation templates. The diagnosis template has a prose response contract, not the draft schema. Include operator-specific directives from the existing EoH semantics; use task-neutral wording, not packing geometry.

- [ ] Capture attempt index, proposed candidate ID, operation, all parent/inspiration IDs, instruction snapshot, raw response/error and parsed draft. A small provider-recording wrapper captures raw results before Slick parsing; do not add a global tracing service.
- [ ] Check literal `{{ untrusted_text }}` in editable instructions is passed as text rather than rendered a second time, and render every operation to catch missing template variables.
- [ ] Run the focused checks and existing `rsikit.tests.test_rsikit`; update exports and review the diff. This task deliberately introduces generic prompt wording and a structured response; it is not an exact prompt-text-preserving migration.

Representative assertion body for the new async test, using existing unittest conventions and setting the repository template root in test setup:

```python
parent = Candidate(id=0, source="def f(): return 0", evaluation=Evaluation(valid=True, metrics={"score": 0}))
other = Candidate(id=1, source="def f(): return 1", evaluation=Evaluation(valid=True, metrics={"score": 1}))
provider = ScriptedProvider(['{"description":"combine ideas","source":"def f(): return 2"}'])
proposer = PromptProposer("Improve f", provider, instructions={"E2": "Keep it short"})
source = await proposer(parent, (parent, other), context={"operation": "E2", "parents": (parent, other)})
assert source == "def f(): return 2"
assert "Keep it short" in provider.calls[0]
assert other.source in provider.calls[0]
assert proposer.records[-1]["parent_ids"] == [0, 1]
```

## Task 2: Use the same proposer in existing experiments

**Files:** modify `rsikit/examples/circle_packing/experiment.py`, `__main__.py`, `README.md`, `rsikit/README.md`; extend `rsikit/tests/test_prompt_components.py` and `test_circle_packing.py`.

**Consumes:** task 1's `PromptProposer`, unchanged strategy `context`, `RepairingProposer` and external packing evaluator.

**Produces:** opt-in `--prompt-mode modular` with legacy as the default comparison baseline; optional `--instruction-file PATH` for the `mutate` operation. File text is data, not a Jinja template or code. Programmatic callers can supply per-operation instructions directly.

- [ ] Add an offline routing check for HillClimb and EoH using the same generic proposer and the existing fake measurements. Check candidate lineage, pending/update behavior and EoH cycle timing against existing population assertions.
- [ ] Preserve the closure used by the existing proposal contract:

```python
async def propose(parent, history):
    context = dict(strategy.context)
    context["evidence"] = measured_context(parent, context)
    return await operations(parent, history, context=context)
```

`measured_context` is a local experiment function returning packing data for selected parent/inspiration IDs; it performs no evaluation. For non-packing callers, omit `evidence`.

- [ ] Extract the packing task/interface description for the generic proposer; leave geometric measurement, execution caching and repair checks in the experiment. Pass `operations.repair` from task 1 to the existing repair wrapper in modular mode; retain the original packing repair in legacy mode.
- [ ] Resolve template composition explicitly. In modular mode, configure the repository root once and use the `rsikit/prompts/...` paths defined in task 1 for every generic operation. Leave legacy-only mode and `SlickProposer` roots untouched. Never switch the process-global root during a call or run. These examples run from the checkout; installed-distribution template loading is outside this increment.
- [ ] Save `proposal_records.json` alongside existing candidate, selection and repair records. Preserve initial/final best artifacts and failure saving.
- [ ] Run offline routing/template checks. Run Docker integration only when explicitly enabled via the existing test environment flag. Document that modular prompts are a new experimental variant.

## Task 3: Add measured reflection with bounded memory

**Files:** create `rsikit/reflection.py`, `rsikit/prompts/reflection/pair.j2`, `failure.j2`; modify exports, the modular experiment and `test_prompt_components.py`.

**Interfaces:**

```python
ReflectionMemory(task, provider, *, max_items=8)
await memory.observe(parent, candidate, objective="score", maximize=True)
memory.records  # completed reflections with candidate/parent IDs and evidence
memory.texts    # tuple[str, ...] containing at most max_items guidance strings
```

- [ ] Add one focused sequence test: an improvement is reflected in worse→better order; the next proposal includes its guidance; a second observation obeys the configured memory bound. Repeat the ordering assertion with a minimizing objective. Skip ties without a model call.
- [ ] Implement two explicit reflection paths. Valid unequal scores use the measured pair; invalid completed attempts with diagnostics use failure reflection. Reject missing objective evidence through the existing `EvaluationError` semantics; never supply invented scores.
- [ ] Use task-neutral prompts: pair reflection identifies concrete differences supported by measured outcomes; failure reflection identifies a next change supported by diagnostics. Neither declares new correctness or modifies the evaluator.
- [ ] Append memory only after successful nonblank reflection generation. Treat empty model reflection as an invalid generated response; propagate provider/cancellation failures. Memory remains unchanged on failure.
- [ ] Integrate after candidate completion:

```python
before = len(strategy.history)
candidates = await strategy.generate()
evaluations = [await evaluate(candidate) for candidate in candidates]
await strategy.update(candidates, evaluations)
for candidate in strategy.history[before:]:
    parent = strategy.history[candidate.parent_id]
    await memory.observe(parent, candidate, objective=strategy.objective, maximize=strategy.maximize)
```

The example's cached evaluator remains its existing implementation. A `generate()` infrastructure exception does not reach reflection. A recorded `ProposalRejected` may reach failure reflection with its real diagnostics. Successful strategy updates stay committed if reflection fails; the runner saves them in its existing `finally` path.

- [ ] Add `context["guidance"] = memory.texts` before subsequent generation; reset memory for independent runs. Export records separately from candidate fitness.
- [ ] Verify rejected attempts, ties and reflection failure with scripted calls; rerun population tests to establish unchanged admission behavior. Add opt-in `--reflect`; keep it off for the static baseline.

## Task 4: Extract reusable parent and survivor decisions

**Files:** create `rsikit/selection.py`; modify `rsikit/population.py`, exports and `rsikit/tests/test_population.py`.

**Interfaces:**

```python
def better(candidate, incumbent, *, objective, maximize=True) -> bool:
    left = candidate.evaluation.metrics[objective]
    right = incumbent.evaluation.metrics[objective]
    return left > right if maximize else left < right

def top_candidates(candidates, count, *, objective, maximize=True) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.evaluation.metrics[objective], reverse=maximize)[:count]

def rank_parents(population, count, *, rng) -> list[Candidate]:
    size = len(population)
    return rng.choices(population, weights=[1 / (rank + size) for rank in range(1, size + 1)], k=count)

def lineage_weights(archive, *, objective, maximize=True, score_bounds=(0.0, 1.0)) -> list[float]:
    lower, upper = score_bounds
    children = Counter(c.parent_id for c in archive)
    weights = []
    for candidate in archive:
        score = candidate.evaluation.metrics[objective]
        quality = min(1.0, max(0.0, (score - lower) / (upper - lower)))
        if not maximize:
            quality = 1 - quality
        weights.append(1 / (1 + math.exp(-10 * (quality - 0.5))) / (1 + children[candidate.id]))
    return weights
```

These consume measured, valid candidates selected by the strategy. Do not turn invalid execution into a comparable scalar. The `lineage_weights` implementation is the existing sigmoid quality divided by one plus admitted-child count, with current normalization and minimizing reversal.

- [ ] Extend the existing policy test with one regression that fails elite survival, remains in DGMArchive, and occupies an empty AlphaEvolve cell, while global best remains unchanged. Reuse existing test setup.
- [ ] Extract the functions and use them only where the existing expression has identical semantics. Preserve call order and RNG draws; do not sort an island differently before a uniform choice.
- [ ] Keep archive admission, cycle advancement and reset sequencing inside the concrete strategies. Do not replace them with a shared `accept()` switch.
- [ ] Run `rtk proxy optimizer/.venv/bin/python -B -m unittest rsikit.tests.test_population -v`; compare selected IDs for existing fixed seeds. Update documentation with independently usable selection examples.

## Task 5: Measure and evolve an instruction through downstream outcomes

**Files:** create `rsikit/prompt_search.py`, `rsikit/prompts/prompt_search/revise.j2`, `rsikit/tests/test_prompt_search.py`; modify exports and README.

**Interfaces:**

```python
@dataclass(frozen=True)
class PromptTrial:
    case_ids: tuple[str, ...]
    utilities: tuple[float, ...]  # higher is better; includes failures
    feedback: tuple[str, ...]    # same ordered cases

    @property
    def mean(self) -> float:
        return statistics.fmean(self.utilities)

EvaluateInstruction = Callable[[str, tuple[str, ...]], Awaitable[PromptTrial]]

search = PromptSearch(task, provider, evaluate, before_comparison=None)
await search.run(initial_instruction, development=development, selection=selection, revisions=3)  # -> str
search.best       # accepted instruction text
search.history    # revision/pair measurements and decision records
```

The ordered `case_ids` represent case/repeat pairs (for example `case-a/repeat-0`), so repeats are explicit. Validate callback evidence at the runtime boundary: exact requested case IDs/order, matching lengths, nonempty measured cohort and finite utilities. Bad evidence raises `EvaluationError`; it is not a bad candidate score. Failure utility is defined by the task callback and recorded there. Initial failure utility must be finite and comparable with valid utility.

`before_comparison` is an optional synchronous application callback `() -> None`, invoked after a nonblank changed instruction is produced and before either development arm is evaluated. The budgeted example uses it to ensure capacity for the worst-case complete paired comparison; it raises `BudgetExhausted` if insufficient. The search class does not own budgets. Caller-supplied development and selection cohorts must be disjoint, and final test cases are never supplied.

- [ ] Add the deterministic promotion check below using `ScriptedProvider` and a local async evaluator:

```python
async def evaluate(instruction, cases):
    value = 2.0 if instruction == "challenger" and cases == ("dev",) else 1.0
    return PromptTrial(cases, (value,), ("measured",))

provider = ScriptedProvider(["challenger"])
search = PromptSearch("Improve the mutation instruction", provider, evaluate)
best = await search.run("incumbent", development=("dev",), selection=("select",), revisions=1)
assert best == "incumbent"  # development improves; selection tie does not promote
assert search.history[-1]["decision"] == "retain"
```

- [ ] Implement a plain-text revision prompt using incumbent instruction and completed development measurements. Preserve one-editable-block semantics. Reject blank or unchanged proposals as retained attempts without downstream evaluation; retain the raw revision response.
- [ ] Measure the incumbent once initially for revision context. For each challenger, invoke `before_comparison` when supplied, then freshly measure **both** incumbent and challenger on matching development cases; alternate evaluation order by revision parity. Require strict mean improvement. If it passes, measure both on matching selection cases with the same rules and require strict improvement there. Existing initial measurements are context, not cached comparison evidence.
- [ ] Only completed pairs can produce a promotion decision. Save partial measurements if a callback fails; retain incumbent and propagate infrastructure errors. No selection feedback enters the revision prompt. Do not expose final test cases to this class.
- [ ] Set `search.best` only after the full promotion gate. Record prompt strings, revision attempt, split/case identity, measurements and decision. Return the incumbent even when every challenger fails or ties.
- [ ] Add assertions for successful promotion, minimization normalized by the callback, invalid-output penalties retained in the denominator, mismatched case evidence and incomplete comparison. Keep these as one small state-transition sequence rather than a framework of fixtures.
- [ ] Run `rtk proxy optimizer/.venv/bin/python -B -m unittest rsikit.tests.test_prompt_search -v`; document that this is measured instruction hill climbing, not weight training or a full Promptbreeder/GEPA reproduction.

## Task 6: Bounded end-to-end example and comparison record

**Files:** create `rsikit/examples/prompt_search.py`; modify README; extend `test_prompt_search.py`.

**Consumes:** `PromptSearch`, `PromptProposer`, `ReflectionMemory`, the unchanged strategy/repair/evaluation interfaces and existing scripted provider.

**Produces:** an offline `python -B -m rsikit.examples.prompt_search` example that demonstrates instruction revision → generated candidate → fixed evaluation → retain/promote. It saves prompt versions, candidate evidence, completed/partial comparison records and call counts. It makes no network calls. Real applications inject their existing provider/evaluator into the same experiment functions.

- [ ] Implement an application-owned `BudgetedProvider` with `acall(context, *, tools=None, tool_results=None)`, delegating the exact provider response and incrementing `calls` before dispatch. Exceeding root or active-trial allowance raises `BudgetExhausted` before dispatch. API failures consume calls. Inject the same object into every prompt operation, reflection and repair. Its sequential-run helpers are `ensure_capacity(calls)`, `start_trial(max_calls)` and `end_trial()`; trial accounting records the root counter at start and clears in a `finally` block.
- [ ] Define a finite experiment allowance before running. Supply `before_comparison=lambda: provider.ensure_capacity(2 * (len(development) + len(selection)) * max_trial_calls)` to `PromptSearch`. This checks worst-case capacity without charging unused calls; serial execution prevents other work from consuming the allowance during the pair. `max_trial_calls` includes all configured generation, diagnosis, repair and reflection calls within one case/repeat. Initial context measurement and meta-revision consume the same root counter separately. If insufficient allowance remains, stop with `budget`, save partial progress and return `search.best`; do not compare an incomplete pair.
- [ ] Implement `evaluate_instruction(instruction, cases)` using fresh strategy/proposer/memory per independent case, the same initial artifact and settings, and declared finite failure utility. Use existing `Evaluation` for artifacts. Return a complete `PromptTrial` with every requested case/repeat represented. Keep all rejected artifacts in evidence.
- [ ] Create a scripted scenario where a challenger improves development but regresses selection, then a later challenger improves both. Assert the first is rejected and the second is retained. No live-model quality assertion is made.
- [ ] Add one call-accounting check that exhausts the allowance during a repair/reflection path; ensure no extra underlying API call is dispatched and the incumbent survives. Record unavailable token/currency usage as unknown.
- [ ] Save run metadata: model identifier/settings when supplied, evaluator/case-set identifier, initial source, instructions, rendered operation inputs, call counts, evaluations and decision. Reuse ordinary JSON/file persistence; no database or resume framework.
- [ ] Document future live comparison recipes: static instruction; static+reflection; evolved instruction; existing EoH operators. Keep model/settings fixed, record all resource counts and evaluate the selected instruction once on untouched final cases. Selection and final cases must not be reused in memory construction.
- [ ] Run focused verification:

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest discover -s rsikit/tests
rtk proxy optimizer/.venv/bin/python -B -m rsikit.examples.prompt_search
rtk proxy ../slick/.venv/bin/ruff check rsikit
rtk proxy ../slick/.venv/bin/ruff format rsikit --check
```

Scope linting to changed files if pre-existing failures occur; report unrelated failures rather than silently fixing them. Docker checks retain their existing opt-in status. Review task diffs and document the exact checks run.

## Follow-on experiments after the first three increments

These are sequenced extensions, not unfinished requirements of tasks 1–6.

| Order | Extension | Concrete donor / integration | Gate before expanding |
|---|---|---|---|
| 1 | Select verified examples and preserve cross-task textual insights | ExpeL/OPRO: development-case evidence → insight records → explicit context selection | Demonstrate benefit on tasks excluded from memory construction; exact scan before a vector service |
| 2 | Evolve different prompt components or crossover instructions | GEPA/EvoPrompt: one named instruction block at a time, component lineage and matched trials | Track per-case results; retain instruction identity; separate generation gains from extra evaluation cost |
| 3 | Evolve the mutation instruction itself | Promptbreeder-style text-only outer loop | Score through the same downstream evaluator; all nested calls share root accounting |
| 4 | Allocate effort adaptively among operation prompts | AdaEvolve/QUBE ideas implemented as host-side statistics | Credit measured offspring results, not parent fitness; compare under equal resource allowance |
| 5 | Richer diversity and specialist archives | In-context QD, EoH-S, MEOH | Add the relevant descriptors/case matrices/objective semantics; do not collapse them into one unexplained score |

Executable agent/controller self-rewriting is deferred, not required to obtain text-based self-improvement. Numeric policy coevolution and training-based methods are excluded. No adapters for training APIs, model checkpoints, soft embeddings or gradients should be created.

## Plan review and completion

- [ ] Tasks 1–2 demonstrate one proposer reused by multiple unchanged strategies.
- [ ] Task 3 shows measured feedback affects the next prompt with explicit memory scope.
- [ ] Task 4 preserves EoH, island and stepping-stone semantics under extraction.
- [ ] Tasks 5–6 distinguish prompt fitness from candidate fitness, use complete matched evidence and respect the total allowance.
- [ ] The public examples still work with existing callback signatures.
- [ ] All new templates render from the documented single root, including combined reflection and prompt search.
- [ ] No model training or weight access exists anywhere in the delivered path.

Planning validation: inspected current strategy, proposer, repair, packing integration, scripted provider and Slick prompt-rendering interfaces. The implementation steps and tests above are proposed work; no new code or live trials were run during planning.

Self-review completed: all ten global constraints match the design verbatim; six tasks cover the three increments; relative document links resolve; no incomplete planning markers remain. Clarified one template root for the new composed path and an application-owned capacity callback before paired instruction comparisons. Training-related extensions are excluded rather than deferred.
