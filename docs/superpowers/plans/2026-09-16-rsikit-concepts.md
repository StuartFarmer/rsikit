# RSIKit Concept Modules Implementation Plan

> Execute inline with `superpowers:executing-plans`, checking each deliverable before proceeding.

**Goal:** Make selection, variation, archives/QD and islands independently composable.
**Architecture:** Ordinary sampler functions, two small bandit state owners, explicit archive
admission, and snapshot migration. Demonstrate composition in an offline experiment rather
than introduce a universal agent runtime. Existing recipes retain their policies.
**Tech stack:** Existing Python/Slick/Pydantic; stdlib RNG and data structures.
**Spec:** [Approved conceptual map](../../../outputs/rsikit-conceptual-modules-comparison.md).

## Global constraints

- Fixed API models; no LLM training or new dependencies.
- Preserve legacy strategy behavior, RNG order and generation/update contracts.
- Runtime measurements are validated; annotated caller configuration is trusted.
- Separate prompt files and explicit Slick generated-output contracts.
- Selectors choose; recipes define reward and when to observe; archives define retention.
- Work stays in the current branch alongside earlier and concurrent changes; no blanket staging.

## 1. Selection and credit

Files: `rsikit/selection.py`, `rsikit/tests/test_selection.py`.

- [x] Add direct selectors `uniform`, `weighted`, `tournament`, `softmax`,
  `epsilon_greedy`; keep existing rank sampling and survival helpers.
- [x] Add `UCB1.choose(arms)` / `.observe(arm, reward, attempt_id=..., outcome=...)`.
  Cold arms are explored first; ties use seeded RNG; rewards must be finite in [0,1].
- [x] Add `ThompsonSampling` with the same calls, explicit Beta prior, binary reward.
  None records an unscored event without updating beliefs. Each policy owns role,
  RNG, observations and per-arm statistics. Eligible arms supplied at each choice.
- [x] Check cold starts, seeded decisions, invalid measurements leaving state unchanged,
  role separation, ineligible arms and posterior updates. Run focused tests red then green.

## 2. Variation vocabulary and lineage

Files: `rsikit/proposer.py`, `rsikit/strategies.py`, operation template,
`rsikit/tests/test_prompt_components.py`.

- [x] Add a distinct `crossover` operation/template with at least two distinct parents.
  Preserve existing EoH operations and exact source bytes.
- [x] Add `Candidate.parent_ids` alongside compatible primary `parent_id` and populate
  it for successful and rejected strategy attempts. Test crossover source/lineage
  and edit rejection relative to the primary parent.

## 3. Archives and QD

Files: `rsikit/archives.py`, `rsikit/tests/test_archives.py`.

- [x] Implement `EliteArchive`, `SteppingStoneArchive`, `QDArchive` with `.add(candidate)`
  and `.candidates`; keep score direction explicit. Reject unmeasured/missing scores;
  invalid candidates return false. Stable ties retain incumbents, IDs identify candidates.
- [x] Add `FeatureGrid.locate(features)` and cell count; QD admission computes the
  measured cell before mutation. Test boundaries, novel weaker niches, failed descriptor
  atomicity and maximizing/minimizing survival.

## 4. Island exchange

Files: `rsikit/islands.py`, `rsikit/tests/test_archives.py`.

- [x] Implement snapshot `migrate(archives, routes, select)` over archive objects.
  Select all outgoing candidates before any destination admission; preserve IDs;
  no bandit credit for migration. Return admission events.
- [x] Verify ring routing cannot cascade arrivals within the same migration step and
  recipient archive policy decides admission. Keep allocation outside this function.

## 5. Working compositions and public guide

Files: `rsikit/examples/concepts.py`, README, exports, tests.

- [x] Compose a standard population and adaptive QD islands using the same variation
  and parent samplers; separate island/operator bandits, explicit binary improvement
  reward, fixed evaluator, shared scripted provider, recorded decisions and lineage.
- [x] Demonstrate mutation/crossover selection, archive coverage and snapshot migration
  without executing generated source or making network calls.
- [x] Run full RSIKit suite, both offline examples, Ruff and diff checks; review source
  boundaries and failure accounting. Document concrete scope and later variants.


## Completion evidence — 2026-09-16

All five deliverables are implemented in the existing working branch. No new
runtime abstraction or dependency was introduced. Existing recipes remain intact;
the new example composes existing SequentialStrategy with independent samplers,
archives and migration. Thompson sampling uses binary Beta-Bernoulli observations;
UCB1 takes measured rewards in [0,1]. None records an unscored event.

Checks completed:

- Full RSIKit unittest discovery: 68 discovered, 62 passed, 6 opt-in Docker tests skipped.
- Concepts example: elite population and adaptive QD islands each used six scripted
  calls, exercised crossover and invalid geometry, and retained best sum_radii=1.25.
  Each QD island ended with both behavior niches occupied.
- Earlier prompt-search example: still completes retain/promote with 20 scripted calls.
- Ruff correctness/import checks and formatting cover the RSIKit package.
- No paid API calls, generated-source execution or model-weight updates were used
  by the new example.

Read-only review corrections were verified with failing-then-passing checks:
finite-extreme softmax/grid arithmetic, descriptor bounds covering the full legal
packing domain, and distinct cancellation attribution without invented reward.
The initial demo seed was changed to exercise both variation operators in six
calls; no sampler policy was changed to force a desired outcome.

New files: archives.py, islands.py, examples/concepts.py, crossover.j2, and focused
selection/archive/composition tests. Public exports and README document contracts,
lineage, exactly-once credit ownership and migration partial-commit behavior.
Discounted/windowed bandits, non-Bernoulli posterior models, Pareto archives and
arbitrary tree/racing schedulers remain separate future variants.
