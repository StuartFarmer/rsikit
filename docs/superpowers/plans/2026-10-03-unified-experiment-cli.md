# Unified Experiment CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run policy search or evaluate existing policy files through one `rsikit` command, using an environment name/file, an optimizer name/file, ordinary flags, or a complete configuration file.

**Architecture:** The environment definition supplies the evaluator and policy instructions. The optimizer consumes the existing asynchronous policy-to-`Measurement` callback. A thin research-level runner owns configuration, resources, budget accounting, held-out evaluation, and artifacts; Gymnasium and upstream Ocean objects retain their own APIs.

**Tech Stack:** Python >=3.10, argparse, json, importlib, existing Pydantic, existing Run/Executor/Rollouts/Measurement, upstream PufferLib bindings. PyYAML >=6,<7 handles config files; JSON remains the result/log format.

**Spec:** The requirements below are the specification for this plan. Preserve the upstream workload described in [OCEAN_BENCHMARK.md](../../OCEAN_BENCHMARK.md).

## Global constraints

- Keep `rsikit` core independent of `research`, `examples`, and individual optimization algorithms.
- Reuse `research.rewards.Measurement`; do not invent another result model or force this flow through the older propose/update `rsikit.optimization.Optimizer` protocol.
- Keep Gymnasium environments directly usable. Do not add a mandatory Environment superclass or convert Ocean into Gymnasium VectorEnv.
- Each Ocean seed identifies one upstream batch reset; batch width and horizon are part of the workload. Scalar Gymnasium seeds identify episodes.
- Optimizer integration is written once per optimizer, never once per optimizer/environment pair.
- Keep existing example commands working while the unified command is introduced. Do not silently change their historical defaults.
- Docker remains outside the Python runner. Preserve existing isolation, cancellation, process cleanup, budget reservations, and raw evidence.
- This is a plan only; implementation and paid experiments are not part of writing it.

## Requirements

### 1. Commands and ordinary flags

The target interface is:

```bash
rsikit run --env ocean:g2048 --optimizer elite \
  --model gpt-oss-120b:nitro \
  --population 50 --generations 10 --elites 10 --max-repairs 2 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --batch-size 32 --max-steps 2000 \
  --workers 4 --generation-concurrency 25 \
  --spend-cap 100 --input-price 0.01 --output-price 0.01 \
  --output runs/ocean-elitetable-2048-1

rsikit run --config experiments/ocean-2048.yaml
rsikit run --config experiments/ocean-2048.yaml --population 100 --workers 2

rsikit evaluate --env ocean:g2048 --policy runs/example/winner.py \
  --seeds 2000 2001 --batch-size 32 --max-steps 2000 \
  --workers 4 --output runs/evaluate-winner
```

The price values above illustrate user-supplied ceilings, not verified model pricing. Keep the existing USD-per-million-token units.

- `run` requires an environment, optimizer, and model after resolving config/defaults. `evaluate` requires an environment and one or more policy files, with no optimizer, model, API key, or generation budget.
- `--policy` accepts one or more paths. Config uses `policies: ["..."]`; optimizer/model sections are invalid for `evaluate` rather than silently ignored.
- `--optimizer-kwargs` is unnecessary. Elite exposes `--population`, `--generations`, `--elites`, `--new-fraction`, `--remix-fraction`, `--remix-parents`, `--max-repairs`, and `--target-score` directly.
- Shared flags include model/provider settings, budget limits, output, search seed, generation concurrency/timeout, evaluation workers/timeout, seeds, horizon, score, and held-out selection settings.
- `--env-shoes-per-episode 24` configures Blackjack. Environment-specific constructor options use the `--env-` prefix so they cannot collide with optimizer options. Ocean `--batch-size` is an evaluation setting, not an environment constructor argument.
- `--workers` replaces the inconsistent evaluation `--concurrency` spelling in the new CLI. `--generation-concurrency` remains distinct. Do not enforce Ocean's old benchmark-specific four-worker maximum in the general runner.
- Use `--timeout-per-seed` for evaluation timeout. For Ocean, explain that the current candidate process receives `number_of_seeds * timeout_per_seed`; it is not a separate timer around every seed. Gymnasium applies the timeout to each episode process.
- `rsikit run --optimizer elite --help` shows Elite's options. Adding `--env Blackjack` also shows Blackjack options. Plain help works without Ocean installed or provider credentials.
- Unknown flags, conflicting option registrations, and unsupported options fail before run creation or model calls. Disable argparse abbreviation.

### 2. Complete YAML configuration

Use YAML, as requested after the initial implementation, with PyYAML SafeLoader and duplicate-key rejection. No config inheritance, includes, or interpolation language.

Example `experiments/ocean-2048.yaml`:

```yaml
version: 1
env: ocean:g2048
optimizer: elite
model: gpt-oss-120b:nitro
output: ../runs/ocean-elitetable-2048-1
search_seed: 0
environment: {}
optimizer_options:
  population: 50
  generations: 10
  elites: 10
  max_repairs: 2
  new_fraction: 0.2
  remix_fraction: 0.4
  remix_parents: 3
generation:
  provider: openrouter
  concurrency: 25
  timeout: 120
  max_input_tokens: 65536
  max_output_tokens: 16384
budget:
  spend_cap: 100
  input_price: 0.01
  output_price: 0.01
  max_calls: 1500
  max_tokens: 122880000
evaluation:
  seeds:
  - 0
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
  workers: 4
  timeout_per_seed: 60
  batch_size: 32
  max_steps: 2000
  score_key: merge_score
selection:
  finalists: 5
  validation_seeds:
    start: 1000
    stop: 1128
  test_seeds:
    start: 2000
    stop: 2512
videos:
  top: 0
  workers: 1
```

For Blackjack, use `"env": "Blackjack"`, `"environment": {"shoes_per_episode": 24}`, and `"score_key": "return"`; omit `batch_size`, and omit `max_steps` for complete episodes. For CartPole use `"env": "CartPole-v1"`. The same optimizer options apply to all three.

Configuration requirements:

1. Precedence is **explicit CLI values > config values > selected component defaults > shared defaults**. Parse flags with suppressed defaults so omitted flags cannot overwrite config. Apply defaults only after selecting environment/optimizer.
2. Flags override individual fields. `--population 100` updates only `optimizer_options.population`. Lists replace lists, never concatenate. `--validation-seeds` and `--test-seeds` accept zero or more integers, so an explicitly empty list disables that panel. Search seeds must be nonempty.
3. File-only invocation must be fully functional. Every runnable setting, including environment/optimizer options, budgets, evaluation, selection, and videos, has a config field. API credentials are deliberately excluded: use existing environment variables.
4. Seed fields accept either an explicit list or `{start, stop}` with an exclusive stop. Expand before validation and persistence. Reject duplicates, booleans, negative/noninteger seeds, invalid ranges, and overlap between search/validation/test panels. Apply Ocean's additional width-dependent signed-integer constraint to every panel.
5. `version` defaults to 1 when omitted; reject unsupported versions. Reject unknown fields at every level, duplicate YAML keys, non-finite YAML numbers, invalid types, and invalid component combinations. Error messages identify the dotted field path. Validate numeric bounds and cross-field constraints, including `elites <= population` and valid new/remix fractions.
6. Resolve config-origin paths relative to the config file, and flag-origin paths relative to the invocation directory. This covers output, policy files, component files, and explicitly declared path options. Do not guess that arbitrary string options are paths.
7. If flags change the config's optimizer/environment selector, validate the resulting options against the newly selected component. Reject stale incompatible fields; never silently discard them.
8. `--print-config` loads and validates configuration and prints canonical YAML without requiring credentials, constructing environments, starting workers, creating a run, or calling models. Custom Python modules are imported to read their definitions, so their top-level code still executes; document that files are executable extensions.
9. Save canonical, fully defaulted configuration as `config.yaml` before generation. Save protocol, versions, resolved seed lists, relevant source hashes, upstream binding hash, platform, effective workload, and available container metadata in `manifest.json`. Redact credentials and do not serialize provider objects.
10. Saved `config.yaml` must be accepted as input; callers supply a new output path to rerun. Existing nonempty output directories fail under existing `Run.create` behavior. No implicit resume or overwrite.

Defaults for the new command: search seeds 0–9, workers 4, timeout per seed 60 seconds; generation concurrency 4 and timeout 120 seconds; videos disabled; validation/test panels empty and finalists 1. Empty held-out panels are explicitly labeled “not evaluated” in output. Ocean defaults remain batch size 32, horizon 2000, and merge_score for g2048 / return for Breakout. Gymnasium defaults to cumulative episode reward and its existing episode termination/time limits. Elite defaults follow its existing Config except the shared generation settings. Do not silently inherit the old Ocean 128/512 held-out panels; the example config requests them explicitly.

For paid search, require spend cap and input/output price ceilings as the Ocean command does today. Preserve max-call/token reservations. Elite may derive omitted max_calls as population × generations × (1 + max_repairs), then max_tokens from per-call limits; a custom optimizer must supply max_calls when no such bound is known. The runner must not infer other optimizers' call counts from Elite settings.

### 3. Environment and optimizer contracts

Keep integration in `research`, where both evaluation conventions and optimizer algorithms already live. Install a `rsikit` console entry point targeting `research.cli:main`; include `research` and its prompt/data resources in the wheel. This avoids making core `rsikit` import research algorithms or executable examples.

An environment definition is a small data object containing its option model, argument registration, description function, evaluator-opening function, evaluation defaults, and supported evaluation fields. It lives alongside the environment integration, not on the Gymnasium/PufferLib instance. The shared runner uses:

```python
async with definition.open_evaluator(options=env_options, evaluation=eval_config, run=run) as evaluate:
    measurements = await evaluate(policies, seeds)
```

`evaluate(policies: Sequence[type[Policy]], seeds: Sequence[int]) -> dict[str, Measurement]` is asynchronous, safe for concurrent calls, and returns exactly the requested policy IDs. Successful results contain every requested seed and finite scores; candidate failures become rejected Measurements; infrastructure errors propagate. The session owns evaluator resources. The runner binds search seeds before giving the optimizer its one-argument callback.

- Gymnasium definitions use existing `Executor`, `Rollouts`, and `measure_rewards`.
- Ocean definitions use existing `PanelEvaluator`, unchanged upstream reset/step, and environment-specific instructions moved out of `examples/ocean_search.py`.
- Reuse `rsikit.envs.tasks` presets. Unknown registered Gymnasium IDs may fall back to `gym.make` and space-based instructions; arbitrary task/scoring descriptions can come from a custom definition.
- Unsupported combinations, such as scalar Blackjack with `batch_size=32`, fail. No implicit SyncVectorEnv wrapping or claim that all environment types have the same workload.
- Capability/default metadata must be readable without importing native Ocean bindings; runtime pin/binding checks occur during preflight before paid generation.

Optimizer modules export `Options` (a Pydantic model with forbidden extra fields), `add_arguments(parser)`, and:

```python
async def optimize(*, task, provider, evaluate, run, options, seed) -> list[type[Policy]]:
    ...
```

The returned list contains unique finalists ranked best first using search evidence only. It may be shorter than requested, or empty when no proposal succeeded. The module owns its algorithm, checkpoints and prompt setup. Elite maps public `population`/`elites` to existing `Config.population_size`/`elite_size`; no algorithm rewrite. New optimizers implement this once and work with every compatible policy-generating environment definition. This standardizes integration, not an assertion that every arbitrary existing optimizer already conforms.

Use a small explicit built-in name mapping and `importlib` file loading. `--env ./my_environment.py` exports `environment` (the same definition object used by built-ins); `--optimizer ./my_optimizer.py` exports the module functions/model above. Resolve absolute paths and hash file contents; use path-derived unique module names to avoid same-basename collisions. Missing exports produce actionable preflight errors. No plugin discovery framework or per-pair adapters.

Argument registration uses ordinary argparse calls with `default=argparse.SUPPRESS`. Pydantic models perform the authoritative config validation after merging. Do not build a general type-annotation-to-CLI generator. First parse selectors/config, then register only selected components' flags, then parse the entire argv strictly. Environment options are prefixed; optimizer options use short direct flags. Reject collisions instead of choosing a winner.

### 4. Shared evaluation, selection, and artifacts

- Search receives only search-panel measurements. The runner separately evaluates up to `selection.finalists` returned policies on validation seeds, chooses the highest mean successful result, freezes `winner.py` and `selection.json`, then evaluates that winner on test seeds. Break exact validation ties by incoming finalist rank.
- With no validation panel, select the first successful returned finalist. With no candidates or no successful validation results, report `no_valid_candidate`; do not quietly substitute a baseline. With no test panel, record “not evaluated.” Never repair using validation/test feedback.
- Save `summary.json` for successful, budget-exhausted, failed, and cancelled runs. Unexpected errors produce nonzero exit status. Cancellation cleans up immediately and does not launch held-out work. Budget exhaustion may select already completed candidates after draining admitted work, as the current Ocean search does.
- `evaluate` writes per-policy Measurements and summaries without ranking them into a search winner. Evaluation failures remain distinguishable from zero scores.
- Reuse `Run` for policy/score storage and existing per-backend evidence. Record requested candidates, actual evaluation submissions, seed runs, Ocean transitions, model calls, token usage, and reservations separately. Cached results do not count as fresh execution; failed/repaired executions do. Record unavailable actual token usage as null, not zero. Keep search/validation/test/rendering costs separate.
- Preserve budget-provider behavior by extracting it from the Ocean example rather than implementing a second accounting path. Preserve current Ocean timing diagnostics in its old benchmark command; the generic runner need not emulate Elite-specific timing reports for arbitrary optimizers.
- Video configuration is accepted uniformly but validated by capability. Initially support the existing Blackjack + Elite generation-video workflow only, through its current checkpoint/export machinery. Ocean video requests fail before paid work. Preserve current video behavior and export format; adding universal rendering is separate work.

## Review focus

- Config values must survive omitted CLI flags, including false/zero/empty-list values (Task 1).
- File selectors and paths must work from a different working directory and under the container launcher (Tasks 1 and 5).
- Optional Ocean installation must not break Gymnasium runs or help (Tasks 2 and 5).
- Held-out data and failures must not leak into search or silently become success (Task 4).
- Budget exhaustion, subprocess failure, and cancellation must retain evidence and release resources (Tasks 3 and 4).

## File map

| File | Responsibility |
|---|---|
| `research/cli.py` (new) | argparse, YAML loading/merging, file loading, config validation, entry point |
| `research/experiment.py` (new) | environment definition type, shared evaluation config, lifecycle, selection, summaries |
| `research/environments.py` (new) | Gymnasium definitions and built-in environment mapping |
| `research/ocean/environment.py` | Ocean definitions, options/instructions, evaluator opening; retain upstream metadata |
| `research/providers.py` (new) | Move reusable budget/usage implementation from Ocean example |
| `research/elitesearch/cli.py` (new) | Elite options, argument registration, optimizer entry callable |
| `examples/ocean_search.py`, `examples/elitesearch.py`, `examples/blackjack_videos.py` | Reuse extracted code; retain old CLIs and video compatibility |
| `pyproject.toml`, `scripts/run`, `Dockerfile` if necessary | Console entry, wheel contents, container invocation |
| `tests/test_cli.py`, `tests/test_experiment.py` (new) | Config and end-to-end contracts |
| Existing Ocean/launcher/package-boundary tests | Regression checks and packaging boundaries |
| `experiments/ocean-2048.yaml`, `experiments/blackjack-elite.yaml` (new), `README.md` | Executable configs and usage |

Do not create a separate spec document or schema framework. Existing option models, argparse registration, and this document are sufficient.

## Implementation tasks

### Task 1: Config loading and dynamic flags

**Files:** `research/cli.py`, `tests/test_cli.py`.

**Interfaces:** `load_component(selector: str, *, kind: str, base_dir: Path) -> object`; `parse_config(argv: Sequence[str]) -> dict` returns canonical validated data with resolved paths. Component defaults/option validation are injected via loaded definitions; no native evaluator creation during parsing.

- [x] Write failing unittest cases: file population 50 becomes 100 only with `--population 100`; file workers 4 survives omission; max_repairs 0 survives; `--test-seeds` clears the panel; wrong/duplicate keys, booleans as seeds, overlapping panels, and invalid fractions fail.
- [x] Add path cases: config-relative `../runs/x` resolves beside the config, CLI-relative output resolves against cwd; two same-basename optimizer files remain distinct; switching optimizer rejects leftover unsupported options. Help must not require credentials.
- [x] Run `rtk proxy python -m unittest tests.test_cli` and confirm failures identify missing behavior.
- [x] Implement the two-pass argparse flow, safe YAML loader, selective field merge, Pydantic validation, and canonical `--print-config`. Support the limited seed range object only; no general config expression language.
- [x] Rerun the same command; all cases pass. Review the diff as the first independently testable deliverable.

### Task 2: Environment-owned evaluator sessions

**Files:** `research/experiment.py`, `research/environments.py`, `research/ocean/environment.py`, `tests/test_experiment.py`; extract instructions from `examples/ocean_search.py`.

**Interfaces:** `EnvironmentDefinition` contains `Options`, `add_arguments`, `describe(options, evaluation) -> str`, `open_evaluator(*, options, evaluation, run)` returning an async context manager yielding the two-argument evaluation callable, `evaluation_defaults`, and `supported_evaluation_fields`. `EvaluationConfig` holds resolved seeds/workers/timeout_per_seed/batch_size/max_steps/score_key.

- [x] Write failing checks comparing direct existing evaluator output with the session output for a deterministic CartPole policy and an Ocean g2048 baseline; require identical per-seed scores, policy IDs, and Ocean transition counts for equal settings.
- [x] Add a Blackjack complete-episode check, unsupported batching rejection, and Gym-only loading with Ocean unavailable. Validate all configured seed panels before runtime execution.
- [x] Run `rtk proxy python -m unittest tests.test_experiment` and confirm the new cases fail.
- [x] Implement definitions as data plus existing callable composition. Use `AsyncExitStack` to own environment/Executor/PanelEvaluator lifetimes. Keep scalar and batch policy instructions accurate and describe score/horizon semantics in the optimizer task.
- [x] Rerun the checks, plus `tests.test_ocean_evaluator` and `tests.test_ocean_upstream`; pass before continuing.

### Task 3: Reusable Elite entry and budget provider

**Files:** `research/elitesearch/cli.py`, `research/providers.py`, `examples/ocean_search.py`, `examples/elitesearch.py`, `tests/test_experiment.py`, existing search tests.

**Interfaces:** The optimizer module contract above; `BudgetProvider`/`UsageOpenRouter`/`BudgetExceeded` retain their existing public behavior. Shared generation settings are passed into the optimizer's validated options under generation_concurrency/generation_timeout; the config source of truth remains `generation`.

- [x] Write a scripted-provider check: population 2, generations 1, elites 1 exercises generation, Measurement feedback, checkpoint persistence, and ranked finalist return. The same entry must work with Gymnasium and Ocean callbacks.
- [x] Reuse existing budget tests to pin reservation limits, missing usage, and cancellation behavior; add a shared-runner case proving invalid config triggers zero model calls.
- [x] Run `rtk proxy python -m unittest tests.test_experiment tests.test_ocean_search tests.test_elitesearch` and observe the new entry-point cases fail.
- [x] Move reusable provider code intact, compose EliteSearch with the callback, retain budget-aware stopping/draining, and save current record types. Extract video coordination from the example only as needed; do not make a research module import `examples`.
- [x] Rerun the same command; preserve old search behavior and verify rejected policies still follow existing repair limits.

### Task 4: Unified run/evaluate lifecycle

**Files:** `research/experiment.py`, `research/cli.py`, `tests/test_experiment.py`, `examples/blackjack_videos.py` if needed for config compatibility.

**Interfaces:** `async run_experiment(config: dict) -> dict`; `async evaluate_policies(config: dict) -> dict`. Both return the persisted summary and own resources through context managers. The CLI maps failure status to exit status.

- [x] Write an integration check recording callback seed panels: optimizer sees only search seeds; finalists see validation seeds; only the frozen winner sees test seeds. Check stable tie handling, empty panels, all-invalid candidates, and test failure without winner reselection.
- [x] Add interruption/budget cases: cancellation leaves summary/evidence and no live workers; exhausted generation may select completed candidates; infrastructure failure returns nonzero. Check evaluation-only execution never constructs a provider.
- [x] Check `config.yaml` round-trips with a new output; manifests distinguish Ocean batches from scalar episodes; fresh execution counts exclude cache hits and include failures. Video requests reject Ocean early and preserve the supported Blackjack/Elite path.
- [x] Run `rtk proxy python -m unittest tests.test_cli tests.test_experiment` and confirm new orchestration cases fail.
- [x] Implement preflight, Run creation, canonical artifacts, search/selection/test lifecycle, and evaluation-only path using Tasks 1–3 interfaces. Persist winner choice before opening the test panel.
- [x] Rerun the same tests; inspect one scripted run's config/manifest/summary/winner files manually.

### Task 5: Installable command, container entry, and migration examples

**Files:** `pyproject.toml`, `scripts/run`, `Dockerfile` if required, `tests/test_container_launch.py`, `tests/test_package_boundaries.py`, `experiments/*.yaml`, `README.md`.

**Interfaces:** Installed `rsikit = research.cli:main`; source invocation `python -m research.cli`; primary invocation `rsikit run --config experiments/ocean-2048.yaml`.

- [x] Extend launcher tests for `research.cli` and package-boundary checks to ensure orchestration stays in research and algorithms remain independent. Permit Elite's narrow dependency on shared provider code only if the extracted integration needs it; do not broadly weaken the boundary test.
- [x] Add wheel smoke checks: installed help and custom file loading work outside the checkout, Elite prompt templates are present, and Gym-only help does not import Ocean. Include all runtime research prompt/context resources, not tests or example scripts.
- [x] Keep the launcher simple: initially route `research.cli` through the existing Ocean-capable amd64 image, as unittest already does, rather than reimplement config parsing in bash. Document that this container choice includes Gym-only runs; native Python remains available. Preserve absolute/relative path behavior inside the container by using `/app` project paths and documenting project-relative inputs. External files require explicit read-only mounts; do not pretend arbitrary host paths already work in Docker.
- [x] Add both complete config examples and direct-flag/config-only/override/evaluate-only commands. Document the old-to-new flags, units, held-out defaults, video limits, and custom module exports. Have video export read canonical config or emit the existing compatible metadata rather than losing replay support.
- [x] Run `rtk proxy python -m unittest tests.test_cli tests.test_experiment tests.test_container_launch tests.test_package_boundaries tests.test_ocean_search tests.test_ocean_evaluator tests.test_ocean_upstream`.
- [x] Build/install the wheel in a temporary environment and run help/config smoke checks; run scripted-provider Gymnasium and Ocean searches under Docker, then `rtk proxy ./scripts/run unittest discover -s tests`. No paid model calls are necessary.
- [x] Run the repository's lint/format checks on changed files and `rtk git diff --check`. Distinguish any unchanged baseline failures from regressions. Review the full diff and report the verified commands.

## Completion criteria

The same Elite entry runs Blackjack, CartPole, g2048, and Breakout with environment-selected evaluation; a custom optimizer and custom environment file run through the same CLI; direct flags and equivalent YAML resolve identically; saved configs rerun with a new output; held-out protocol and costs are explicit; existing example commands still work. No custom Ocean binding, new simulator abstraction, or meta-optimizer is needed for this milestone.


## Implementation outcome — 2026-10-03

Implemented and verified. Usage and extension contracts are in [CLI.md](../../CLI.md); complete configs are in `experiments/ocean-2048.yaml` and `experiments/blackjack-elite.yaml`.

- `./scripts/run unittest discover -s tests`: **278 tests passed** in the final packaged Docker image.
- Installed wheel console command and custom optimizer ran a real CartPole evaluation outside the checkout; no model calls.
- Real Blackjack generation video export and cached replay passed; rendering execution counts are separate.
- Changed Python files pass Ruff lint and formatting; `git diff --check` passes.
- All generation tests used scripted providers; no paid experiment was launched.

Implementation decisions: work remained in the existing `unify-envs` checkout alongside the prior upstream migration, without committing or relocating existing changes. The execution ledger stayed in temporary storage. Shared optimizer runtime settings are passed as reserved `options.generation` and `options.videos` dictionaries rather than duplicating them into each public options model; custom modules must not declare those fields. Video implementation moved into the installed research package with compatibility imports for the original examples. Local checks used `.venv/bin/python` because system Python lacks project dependencies.

A fresh reviewer found and verified fixes for missing video metadata, reserved-option collisions, and omitted rendering counts. The proposed automatic host-directory mounting was not added: Task 5 explicitly retains project-relative Docker paths and requires explicit mounts for external files. Native CLI paths follow their config-file/command-line origins. All requested code remains uncommitted on the existing branch.

User-requested update: configuration input, saved config, print-config output, and example files now use YAML. The primary invocation is `rsikit run`, using the existing installed console entry point. JSON measurements and result logs retain their existing format.
