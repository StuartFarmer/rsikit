# Ocean search controllers: GEPA, RRSI and alternatives

Investigated 2026-10-02. This is a source-backed integration recommendation,
not an Ocean performance comparison. No competitor was installed or run against
Ocean, and no paid model calls were made.

## Recommendation

Use **GEPA as the first outer optimizer of controller source**, compare it with
**RRSI's structured harness-editing approach**, and include **ShinkaEvolve as a
practical program-search challenger**. Use a small, fixed generate/evaluate/revise
controller as the common starting artifact. EliteTable should be an optional
historical baseline, not the architecture every method inherits.

There are two different experiments:

| Experiment | Artifact being edited | One evaluator call does |
| --- | --- | --- |
| Direct policy search | Code that plays an Ocean game | Run that policy on the fixed game-seed panel |
| Controller optimization | Code that searches for game policies | Run a whole fresh, budgeted policy search; audit its final policies |

GEPA and ShinkaEvolve can occupy either position, with different evaluator
adapters. RRSI is especially relevant to the second. Putting an optimizer around
a controller is not automatically recursive proposer improvement: the evolved
controller must actually participate in producing its successors for that stronger
experiment. This placement is our proposed Ocean integration, not an upstream result.

## Comparison

| Method | Useful mechanism | Proposed Ocean role | Main limitation |
| --- | --- | --- | --- |
| GEPA | Reflection on execution feedback, candidate history, per-case/objective frontier selection | First outer optimizer; also a direct policy-search baseline | Nested costs and suite-level ratio aggregation need our adapter |
| RRSI | Whole-harness edits, failure analyst, edit history, leakage critic, noise/cost-aware promotion | Structured outer controller evolution | Default selector and token accounting differ from our objectives |
| ShinkaEvolve | Code evolution, archives/islands, novelty filtering, optional prompt evolution | Direct policy-search challenger; alternative outer engine | Extra novelty/meta/model-selection calls must be metered |
| OpenEvolve | Code evolution with MAP-Elites/islands and population hooks | Alternative when population control is useful | More search machinery than the first experiment needs |
| ADAS / Meta Agent Search | Fixed meta-agent writes and tests agent workflows | Small interpretable outer baseline | Research script needs execution isolation and accounting |
| Darwin Gödel Machine | Modified parent agent performs its next self-edit | Later recursive treatment | Benchmark-specific research harness, substantial adaptation |

Mechanisms in this table are supported by the primary sources below. The ordering
is an engineering judgment; no source establishes which wins on our three Ocean objectives.

## GEPA: the smallest established outer-engine integration

The original GEPA paper studies reflective prompt evolution: use execution traces
to diagnose failures, propose improvements and retain complementary candidates.
Its reported sample efficiency is on the paper's tasks, not an Ocean speedup.
[Paper](https://arxiv.org/abs/2507.19457)

The inspected implementation goes beyond prompts. `optimize_anything` accepts a
source string or named text components, and an evaluator returning a scalar score
plus diagnostic information. We can submit a whole `controller.py` without moving
the application into DSPy. The inspected API uses `OptimizeAnythingConfig`; older
examples use compatibility names. Its optional test evaluation is outside the
optimization budget and still belongs in our experiment ledger.
[API source](https://raw.githubusercontent.com/gepa-ai/gepa/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/optimize_anything.py)

The adapter should run a fresh controller under the standard Ocean budget and
return one of our three objective values, alongside compact diagnostic feedback:

- Per-task final scores and incumbent improvement curves.
- Output tokens and evaluation submissions, including failed attempts.
- Invalid-policy rate, repeated errors, timeout examples and repair expenditure.
- Which proposal operators improved the incumbent and which consumed resources.

GEPA's selection machinery can retain per-case and explicit objective champions.
However, ordinary mutation acceptance still uses a scalar improvement criterion.
Returning three metrics does not automatically create three independent searches;
run the performance, token-ratio and evaluation-ratio campaigns separately.
[State/frontier implementation](https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/core/state.py),
[launcher and acceptance wiring](https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/gepa_launcher.py)

Begin with one source file and merging disabled. Built-in merge exchanges named
component values; it is not a semantic source-code merge of arbitrary Python files.
The normal outer proposer remains fixed as candidate controllers evolve.
[Merge implementation](https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/proposer/merge.py),
[reflective proposer](https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py)

Two budget names need care: `max_evals` counts candidate/example evaluator calls,
not distinct controller revisions or inner game-policy proposals; `max_token_cost`
means proposer USD spend, not output-token count. Parent reevaluation, child tests
and validation can all launch expensive nested searches. Grouped evaluation can
also exceed the library limit at a batch boundary, so keep strict admission in our
own coordinator.
[Configuration](https://raw.githubusercontent.com/gepa-ai/gepa/15ee314f9c7d34ec153b809d401f42f55c4dcd76/src/gepa/oa/config.py),
[evaluation server](https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/oa/eval_server.py)

For exact efficiency objectives, an evaluator case should contain the **whole
task suite across the prescribed search replicates**. Compute `S`, `T` and `E`
exactly as defined in the meta-experiment, then return `S`, `1,000,000 × S/T`, or
`100 × S/E` for the chosen track. Averaging separate per-environment ratios would
change the objective. Include per-environment results as diagnostics. GEPA's
optimization-time `valset` is still development data because selection consults
it; it must not contain the benchmark's frozen validation or final test cases.
Start with one outer trial at a time owning the four Ocean workers.

## RRSI: closest to whole-harness improvement

RRSI edits the prompts, control flow, tools and memory around a frozen model. Its
regularization constrains edit size, remembers failed hypotheses, screens
benchmark-specific edits, and penalizes unnecessary cost or instability. Those
are useful ideas for avoiding controller complexity that only fits development
cases. Its published results concern coding, workspace and engineering tasks.
[Paper](https://arxiv.org/abs/2609.24972)

The released loop has separate analysis, proposal, critic, smoke-check, evaluation
and selection stages. Each candidate is isolated in a git worktree and a winning
revision advances the incumbent branch. A `Domain` supplies task splits, run/score
functions, traces and guards. For Ocean, one domain trial should be a **complete
policy-search run**; its score comes from the resulting policy's private audit.
We need none of the original coding/workspace benchmark runners.
[Round loop](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/loop.py),
[domain interface](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/domain.py)

Its default selection is not our ratio objective. It requires a score near the
historical best, applies a relative-cost rule to clear improvements, allows a
weighted score/cost/novelty tradeoff within the noise band, then chooses the
highest-score admissible candidate. Changing its cost field to output tokens
does not turn the final selection into `score / output_tokens`.
[Exact selection code](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/selection.py)

Run an unchanged RRSI selector as a named method baseline if desired. For our
three campaigns, explicitly label the adapted versions: performance without
the soft cost preference; token-ratio selection; and evaluation-ratio selection.
Retain the common hard budgets and quality criteria from the benchmark.

Existing adapters commonly use input-plus-output or total tokens. Aggregation
omits unavailable/nonpositive token observations, and missing relative cost can
be treated as neutral. The outer LLM client returns text without retaining usage
and has retries. We therefore need complete metering of inner and outer calls;
missing cost cannot earn an efficiency rank. Also, the inspected outer client
uses `AnthropicVertex` directly, despite broader provider wording in the README.
Our OpenRouter route would need adaptation.
[Aggregation](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/evaluate.py),
[coding adapter](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/domains/coding/adapter.py),
[LLM client](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/llm.py)

Despite the name, the inspected release keeps the outer proposer fixed: its file
tools can change the designated harness directory, not the external RRSI proposer,
critic or selector. It supports iterative harness improvement directly; it does
not by itself implement our stronger evolving-proposer comparison. A leakage
critic is also not a security boundary for executing candidate code.
[Editable-path enforcement](https://github.com/google-research/rrsi/blob/be50316e1db05914068a973f322770ef08ed7ba1/rrsi/propose.py)

## Other credible options

**ShinkaEvolve** is the practical code-search alternative I would test first.
Its package exposes `ShinkaEvolveRunner`, with parent/inspiration archives,
parallel evaluations and islands. Novelty filtering and adaptive mutation-model
selection are central mechanisms; current configuration also supports prompt
evolution and meta recommendations. Use a single fixed model initially so model
selection cannot confound algorithm quality. Count novelty judgments, meta calls,
rejected candidates and prompt edits in token usage. Its USD cap is useful but
does not substitute for our output-token ledger.
[Repository](https://github.com/SakanaAI/ShinkaEvolve),
[paper](https://arxiv.org/html/2509.19349v1),
[configuration](https://github.com/SakanaAI/ShinkaEvolve/blob/main/docs/configuration.md)

**OpenEvolve** offers a callable evaluator through `run_evolution`, plus MAP-Elites,
islands and population-management hooks. It is a reasonable alternative engine
if those controls prove useful. Its OpenAI client does expose provider usage, but
per-client `last_usage` is not a durable ledger of every request/retry; manual
answers can have no comparable usage record. Its default outer engine remains fixed.
[Repository/API](https://github.com/algorithmicsuperintelligence/openevolve),
[provider implementation](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/llm/openai.py)

**ADAS / Meta Agent Search** is a useful minimal baseline: a fixed meta-agent
proposes Python agent workflows using an archive of earlier designs and results,
reflects, tests and debugs them. The published scripts expose an evaluator function
for adaptation. Reuse that algorithmic pattern rather than copying their direct
`exec` and incomplete usage accounting into our benchmark.
[Repository](https://github.com/ShengranHu/ADAS),
[search implementation](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/search.py)

**Darwin Gödel Machine** is the clearest fit for the later evolving-proposer
treatment. Its source applies the selected parent's accumulated code patches
before invoking that modified agent with `--self_improve`. Its `no_selfimprove`
control skips applying those patches for the editing step. That is a concrete
mechanism for the improving agent to participate in generating descendants, while
the external scheduler remains fixed. The released harness is tied to coding
benchmarks and Docker, so borrowing the controlled comparison is simpler than
porting the full stack.
[Official repository](https://github.com/jennyzzt/dgm),
[self-improvement execution](https://github.com/jennyzzt/dgm/blob/main/self_improve_step.py)

## Recommended experimental order

1. **Check direct search first:** independent generation, a simple fixed
   generate/evaluate/revise controller, GEPA on policy source, and ShinkaEvolve
   on policy source. Keep EliteTable as an optional reference. Measure the three
   objectives with the same model, panels, resource ceilings and held-out audit.
2. **Optimize the controller:** use the same simple starting controller for
   GEPA and RRSI. Give each authority over the same code/prompts. Separate native
   method baselines from adaptations of their objective/selection rules.
3. **Test recursion:** compare the fixed-editor treatment with the DGM-style
   treatment where the promoted editor creates successors. Match model, archive,
   feedback and resource ceilings; keep scoring and budgets external.

The [meta-experiment](OCEAN_META_EXPERIMENT.md) still defines the benchmark, but
its earlier eight-revision sizing is not a GEPA runtime estimate. Budget **actual
outer trial invocations** and all nested policy evaluations; reevaluating a parent
consumes resources too. Outer discovery costs and deployed-controller costs must
remain separate in every method's report.

Use GEPA's diagnostics and RRSI's disciplined edits together only after measuring
them separately. Combining engines first would make it hard to identify what
actually improved search. No new general optimizer framework is required.

## Standalone EliteTable run, with no outer optimization

The existing `research/elitesearch` implementation matches the corresponding
agent/generation/healing files in the sibling EliteTable repository at inspection.
The Ocean runner now exposes its population, generation, elite, repair and episode
seed settings. The search algorithm itself is unchanged.

Only **Ocean 2048 (`g2048`) is runnable today**. Breakout and Maze are proposed
adapters, and the earlier six-environment timing was a 2048 cost projection.

From this checkout, set `MODEL` to your OpenRouter model ID, `SPEND_CAP` to your
chosen USD limit, and `INPUT_PRICE`/`OUTPUT_PRICE` to conservative USD prices per
million tokens for that model. Supply `OPENROUTER_API_KEY` through the environment
or the `.env` file used by `scripts/run`. Then:

```sh
./scripts/run examples.ocean_search \
  --env g2048 --arm elite \
  --population 50 --generations 10 --elites 10 --max-repairs 2 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --workers 4 --generation-concurrency 4 \
  --model "${MODEL:?Set MODEL}" \
  --spend-cap "${SPEND_CAP:?Set SPEND_CAP}" \
  --input-price "${INPUT_PRICE:?Set INPUT_PRICE}" \
  --output-price "${OUTPUT_PRICE:?Set OUTPUT_PRICE}" \
  --output runs/ocean-elitetable-2048
```

This is 500 proposal slots and nominally 5,000 search games before failed attempts
and repair reevaluations. It also validates up to five finalists on 128 cases each
and tests the frozen winner on 512 cases. It evolves game policies, not its controller.
Use a new output directory for each run. `winner.py`, `summary.json`, usage events,
policy sources and per-seed results are retained there.

Default call/token reservation limits now scale with the requested workload:
`50 × 10 × (1 + 2) = 1,500` possible calls, including repairs, and 122,880,000
reserved input-plus-output tokens at the default per-call limits. These are ceilings,
not predicted usage. The chosen spend cap can stop the run earlier: reservations
are conservative and unrefunded, even when actual responses use fewer tokens.
At default limits, permitting all 1,500 calls requires reservation headroom of
`98.304 × INPUT_PRICE + 24.576 × OUTPUT_PRICE` USD. This formula uses your supplied
price ceilings; it is not a quoted provider price or expected bill.

For a local dependency-equipped checkout, replace `./scripts/run` with
`.venv/bin/python -m`; the local path does not enforce the Docker CPU/RAM allocation.
No GEPA, RRSI or other new package is needed for this standalone run.

### Runner verification

All 21 Ocean tests and targeted Ruff lint/format checks pass. A scripted-provider
run exercised the real native evaluator with four workers, two populations of two
policies, custom search seeds, elite edits, finalist validation and the full
512-case winner test. Six invalid CLI configurations were rejected before any
generation. No paid model calls were made.

The full local suite ran 258 tests with the same four existing errors recorded
before this change:

- `ApplicationEpisodeTests.test_replay_records_best_completed_policy_without_changing_original_scores`
  and `ApplicationEpisodeTests.test_run_records_real_video_artifact`: video workers
  exited without a result.
- `ApplicationEpisodeTests.test_scientific_libraries_in_episode_process` and
  `ScientificLibrariesTests.test_numerical_operations`: optional `control` package missing.

Full test log: `/private/tmp/ocean-options-full-suite.log`. Native smoke artifacts:
`/var/folders/zl/54zd_7b91f76wj0dp6dyfjfh0000gn/T/ocean-elite-cli-p9z834ee/run`.

## Evidence limits

RRSI source was pinned to `be50316e1db05914068a973f322770ef08ed7ba1`.
GEPA API/config were inspected at `15ee314f9c7d34ec153b809d401f42f55c4dcd76`,
with additional internals inspected on `main`; this is not a claim of verified
latest HEAD or a tested package combination. Other repositories were read at the
linked branches. Pin complete source versions before implementation or comparison.

Open questions are empirical: which searcher wins, how much reflection helps,
whether results transfer across tasks, and how isolation changes evaluation cost.
None of the published language/coding benchmarks establishes Ocean state of the art.
