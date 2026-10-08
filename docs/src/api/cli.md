# CLI and YAML configuration

The installed `rsikit` command runs the shared experiment runner. It supports
`run`, `evaluate`, and `resume`. Use the [existing-system example](../examples/existing-system.md)
for a complete workflow and [runs and recovery](../guide/runs.md) for output files.

```sh
rsikit --help
rsikit run --env CartPole-v1 --optimizer elite --help
rsikit evaluate --env CartPole-v1 --help
rsikit resume --help
```

Help is component-aware: provide `--env` and `--optimizer` to include their flags.
The four built-in optimizer selectors are `alphaevolve`, `shinka`, `elite`, and
`lineage`. Environment selectors include the built-in task presets, registered
Gymnasium IDs, `PriceSeries`, and optional `ocean:NAME` integrations. A `.py` path
loads a [custom adapter](../guide/customization.md#cli-adapters).

## Commands

```sh
rsikit run --env CartPole-v1 --optimizer elite --model YOUR_OPENROUTER_MODEL_ID \
  --population 2 --elites 1 --generations 1 --max-repairs 0 \
  --seeds 0 1 --workers 1 --max-steps 100 --max-calls 2 \
  --output runs/cartpole-elite

rsikit evaluate --env CartPole-v1 --policy runs/cartpole-elite/winner.py \
  --seeds 100 101 --workers 1 --max-steps 100 --output runs/cartpole-test

rsikit resume runs/cartpole-elite
```

`run` requires the `openrouter` extra and `OPENROUTER_API_KEY`. `evaluate` needs no
model credentials. Outputs must be new directories for `run` and `evaluate`.
`resume` continues only a saved unified CLI EliteSearch, with its existing
configuration and budget ledger. `--policy` accepts one or more source paths.
The command prints a JSON summary; statuses other than `completed` and
`budget_exhausted` return a nonzero exit code. Budget exhaustion can still produce
a winner if valid evaluated candidates exist.

## Configuration rules

Pass a complete YAML file with `--config`. Explicit flags override corresponding
fields in the file, and model defaults fill the rest. `--print-config` validates
and prints the resolved configuration without evaluating policies or making model
calls; loading adapters still executes their Python modules.

```yaml
version: 1
command: run
env: CartPole-v1
optimizer: elite
model: YOUR_OPENROUTER_MODEL_ID
output: runs/cartpole-elite
search_seed: 0
optimizer_options:
  population: 2
  elites: 1
  generations: 1
  max_repairs: 0
evaluation:
  seeds: [0, 1]
  workers: 1
  max_steps: 100
generation:
  concurrency: 1
budget:
  max_calls: 2
selection:
  finalists: 1
  test_seeds: [100, 101]
```

```sh
rsikit run --config cartpole.yaml --print-config
rsikit run --config cartpole.yaml
```

Relative YAML paths use the configuration file's directory; paths supplied on the
CLI use the current working directory. This applies to output/policy paths, local
adapter selectors, and declared `Path` fields in component options. Unknown
fields, duplicate YAML keys, and wrong scalar types are rejected. A file's
`command` must match the requested command. Numeric strings are not numeric
configuration values.

For `evaluate`, replace `optimizer`, `model`, and search-only sections with
`policies: [path/to/winner.py]`. Both commands accept `environment` and `evaluation`
sections. A seed panel accepts a list or `{start: 0, stop: 10}` (exclusive stop).
Seeds must be nonnegative, unique, and nonempty for evaluation; validation and
test panels may be empty. Search, validation, and test panels must be disjoint.

## Common evaluation options

| Flag | YAML field | Default / meaning |
| --- | --- | --- |
| `--env` | `env` | Required environment selector. |
| `--output` | `output` | Required new run directory. |
| `--seeds ...` | `evaluation.seeds` | Seeds 0–9; PriceSeries defaults to `[0]`. |
| `--workers` | `evaluation.workers` | 4 process workers. |
| `--timeout-per-seed` | `evaluation.timeout_per_seed` | 60 seconds per worker episode. |
| `--max-steps` | `evaluation.max_steps` | Unset; environment's episode limit applies. |
| `--batch-size` | `evaluation.batch_size` | Unset; used by Ocean, unsupported for standard Gymnasium integration. |
| `--score-key` | `evaluation.score_key` | `return`; allowed values come from the environment adapter. |

`--batch-size` is environment batching; optimizer proposal batching uses
`--proposal-batch-size`. PriceSeries does not support `--max-steps`. Ocean requires
positive int32 batch size and horizon, and nonnegative int32 seeds.

## Generation and budget options

These apply to `run` only.

| Flag | YAML field | Default / meaning |
| --- | --- | --- |
| `--model` | `model` | Required provider model ID. |
| `--search-seed` | `search_seed` | 0; optimizer randomness, separate from evaluation seeds. |
| `--provider` | `generation.provider` | `openrouter`, currently the only CLI provider. |
| `--generation-concurrency` | `generation.concurrency` | 4 concurrent generation requests. |
| `--generation-timeout` | `generation.timeout` | 120 seconds. |
| `--max-input-tokens` | `generation.max_input_tokens` | 65536 per-call input reservation. |
| `--max-output-tokens` | `generation.max_output_tokens` | 16384 output reservation and request limit. |
| `--max-calls` | `budget.max_calls` | Unset; cumulative model-call cap. |
| `--max-tokens` | `budget.max_tokens` | Unset; cumulative reserved token cap. |
| `--spend-cap` | `budget.spend_cap` | Unset; cumulative reserved spend cap. |
| `--input-price` | `budget.input_price` | Unset; input price ceiling per million tokens. |
| `--output-price` | `budget.output_price` | Unset; output price ceiling per million tokens. |

A spend cap requires both prices, covering the selected model/routing. Full
input/output allowances are reserved before each call; failures, cancellations,
and connection retries consume reservations. Actual billed cost may remain unknown
when usage metadata is absent. The input bound uses UTF-8 byte length plus framing
allowance for text-only requests. A token cap below one full reservation cannot
admit a call.

## Final selection and video

| Flag | YAML field | Default / meaning |
| --- | --- | --- |
| `--finalists` | `selection.finalists` | 1 ranked candidate eligible for final selection. |
| `--validation-seeds ...` | `selection.validation_seeds` | Empty; when supplied, select by validation mean. |
| `--test-seeds ...` | `selection.test_seeds` | Empty; evaluate the selected winner without reselecting. |
| `--video-top` | `videos.top` | 0; optional generation video count. |
| `--video-workers` | `videos.workers` | 1 rendering worker. |

Videos currently require `--env Blackjack --optimizer elite` and full episodes
(no `--max-steps`). Rendering is additional execution work, reported separately.

## Environment-specific options

Blackjack accepts `--env-shoes-per-episode` (`environment.shoes_per_episode`,
default 24). PriceSeries requires `--env-data-path` and accepts
`--env-price-column`, `--env-time-column`, `--env-asset-name`, `--env-fee-rate`,
`--env-initial-cash`, `--env-start-time`, and `--env-end-time`. Their YAML names drop
`env_` and use underscores. Other environment adapters register their own flags.

## Optimizer-specific options

Flags use kebab-case; YAML uses the corresponding field name in
`optimizer_options`. For example, `--proposal-batch-size 4` sets
`optimizer_options.proposal_batch_size`. The models below are the source of truth
for defaults and constraints. Python optimizer `Config` objects use their own
field names; see the [research optimizer reference](optimizers.md).

### EliteSearch CLI options

<!-- api: research.elitesearch.cli.Options
{members: true, show_root_heading: false}
-->

### AlphaEvolve CLI options

`variant` selects `paper` (default), `original`, or `improved`. Archive fields
`objective`, `elite_fraction`, `migration_interval`, and `migration_count` require
`paper`; variant-specific validation applies to its configuration.

<!-- api: research.alphaevolve.cli.Options
{members: true, show_root_heading: false}
-->

### ShinkaEvolve CLI options

<!-- api: research.shinkaevolve.cli.Options
{members: true, show_root_heading: false}
-->

### LineageSearch CLI options

<!-- api: research.lineagesearch.cli.Options
{members: true, show_root_heading: false}
-->

