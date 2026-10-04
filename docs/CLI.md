# Unified experiment CLI

`rsikit run` searches for policies. `rsikit evaluate` measures existing policy files without a model. Both use the environment's evaluator: Gymnasium episodes for Blackjack/CartPole, upstream native batches for Ocean.

Activate your virtual environment and install the project (`pip install -e '.[openrouter]'`) to get the `rsikit` command, or use `python -m research.cli` from the checkout. Ocean additionally requires the [Ocean installation](OCEAN_BENCHMARK.md). Commands below are the new interface; existing example commands remain available with their original defaults.

## Run from a complete config

```bash
rsikit run --config experiments/ocean-2048.yaml
rsikit run --config experiments/blackjack-elite.yaml
```

Set `OPENROUTER_API_KEY` through the existing `.env`/environment mechanism. Budgets are optional and unlimited by default. The example configs omit `budget`.

Inspect configuration without creating a run or making model calls:

```bash
rsikit run --config experiments/ocean-2048.yaml --print-config
```

Override individual settings with ordinary flags:

```bash
rsikit run --config experiments/ocean-2048.yaml \
  --population 100 --generations 5 --workers 2 \
  --output runs/ocean-another-search
```

Explicit flags override config fields; config fields override defaults. Omitted flags leave file values intact. Lists replace lists. Config-origin paths resolve relative to that config file; flag-origin paths resolve against the working directory. A saved `config.yaml` is runnable again with a new output path; existing directories are never overwritten or implicitly resumed.

## Run with flags

```bash
rsikit run --env ocean:g2048 --optimizer elite \
  --model gpt-oss-120b:nitro \
  --population 50 --generations 10 --elites 10 --max-repairs 2 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --batch-size 32 --max-steps 2000 \
  --workers 4 --generation-concurrency 25 \
  --output runs/ocean-elitetable-2048-1
```

This command requests search only. The checked-in Ocean config additionally requests five finalists, 128 validation seeds, and 512 test seeds. These are different workloads. To set held-out panels with flags, use `--validation-seeds 100 101` and `--test-seeds 200 201`. Supplying either flag without values clears that panel.

Use `--env Blackjack --env-shoes-per-episode 24` for finite-shoe Blackjack, or `--env CartPole-v1` for Gymnasium CartPole. Omit `--batch-size` for scalar environments; it is rejected there. Other registered Gymnasium IDs use `gym.make` and a description of the observation/action spaces. Existing named task presets keep their settings, including continuous windy LunarLander.

`rsikit run --env Blackjack --optimizer elite --help` shows shared flags and the selected components' options. Environment constructor flags have an `--env-` prefix. For example, Ocean supports `--env-reward-scaler 2` and `--env-use-sparse-reward`, with `--no-env-use-sparse-reward` to turn it off. Breakout supports its geometry and speed settings, frameskip, and continuous actions.

For a single local price CSV, use `--env PriceSeries --env-data-path prices.csv`.
It defaults to the `price` column and one seed. Column names, timestamp ranges,
fees, and starting cash are configurable; see [PriceSeries](PRICE_SERIES.md).

## Choose an optimizer

| Selector | Main controls |
| --- | --- |
| `alphaevolve` | `--variant paper|original|improved` (paper default), `--proposals`, `--proposal-batch-size`, `--islands` |
| `shinka` | `--generations`, `--proposal-batch-size`, `--islands`, `--archive-size` |
| `elite` | `--generations`, `--population`, `--elites` |
| `lineage` | `--max-attempts`, `--families`, `--initial-per-family`, `--proposal-batch-size`, `--patience` |

All four use the same `rsikit.search` runner, per-seed `Episode` feedback,
search seed panel, budget provider, and validation/test selection. Options belong
to the selected algorithm; switching selectors with incompatible saved options
raises during validation. Use that selector's `--help` for the full list.
`--proposal-batch-size` controls optimizer rounds; `--batch-size` continues to
control Ocean episode batches. Shared `--generation-concurrency` and
`--generation-timeout` configure every optimizer. Provider ensembles, embeddings,
and custom grading remain Python API choices.

New manifests contain `optimization_schedule: round-v1`: complete generation,
then evaluation, then update, with concurrency within each stage. Repair rounds
settle before the next generation or family cull. This changes feedback timing
from historical overlapping implementations.

## Configuration fields

YAML is the supported config format. See the complete [Ocean](../experiments/ocean-2048.yaml) and [Blackjack](../experiments/blackjack-elite.yaml) files.

| Field | Meaning |
|---|---|
| `version` | Schema version, currently 1 |
| `env`, `optimizer` | Built-in name or Python file path |
| `model`, `output`, `search_seed` | Model identifier, new run directory, optimizer random seed |
| `environment` | Selected environment constructor options |
| `optimizer_options` | Selected optimizer settings, e.g. population/generations/elites |
| `generation` | Provider (currently OpenRouter), concurrency, timeout, per-call token bounds |
| `budget` | Optional spend, call, and reserved-token caps; omitted limits are unlimited |
| `evaluation` | Seeds, workers, per-seed timeout allowance, score, optional batch width/horizon |
| `selection` | Finalist count, validation seeds, test seeds |
| `videos` | Top generation leaders and rendering worker count; disabled by default |

Seed panels accept lists or an exclusive range: `{"start": 1000, "stop": 1128}`. All panels must be internally unique and mutually disjoint. Duplicate/unknown YAML keys, unsupported fields, invalid types, non-finite numbers, and unsupported environment/option combinations fail before paid generation. API keys stay in environment variables, outside config files and saved manifests.

Delete the entire `budget` section (or use `budget: {}`) to disable cumulative budget limits. Limits are never inferred from optimizer settings. Population, generations, repair limits, and per-call `generation` token bounds still apply. Usage logging remains enabled; reserved cost is null without both prices.

To opt in, set only the caps you want, for example `budget: {max_calls: 1000}` or `--max-calls 1000`. A `spend_cap` also requires `input_price` and `output_price`, in USD per million tokens. Reservations use the full per-call token bounds, including failed calls, and are not a provider billing guarantee. Omitted or null individual caps are unlimited; zero is invalid.

Defaults: search seeds 0–9; four evaluation workers; 60 seconds of timeout allowance per seed; four concurrent generation calls with 120-second timeouts; no validation/test panels; one finalist; no videos. Elite's default population/generations/elites are 50/20/10. Ocean defaults to at most 32 concurrent episodes and a 2000-step cap per episode, scoring merge_score for 2048 and return for Breakout. Other environments score cumulative episode reward.

Each Ocean seed runs one episode, stopping at termination or the step cap. Batch size limits concurrent episodes without multiplying them. Finished lanes freeze. `return` sums rewards for that episode; other metrics read the final state, including capped games. Ten seeds with batch size 32 and max steps 2000 therefore request ten episodes and **at most 20,000 transitions per candidate**. See [the Ocean protocol](OCEAN_BENCHMARK.md) for the upstream compatibility patch and seeding details.

`--workers` controls evaluation concurrency, while `--generation-concurrency` controls model calls. `--timeout-per-seed` applies per Gymnasium episode; Ocean's candidate process receives `number_of_seeds × timeout_per_seed` for its full panel.

## Resume an interrupted search

```bash
rsikit resume runs/my-search
# Equivalent without the installed entry point:
python -m research.cli resume runs/my-search
```

Resume currently supports the built-in Elite optimizer. It loads the saved
`config.yaml`, restores generations and candidates from `run.sqlite`, and continues
in the same directory. Completed candidates are retained; unfinished evaluations
reuse cached episodes, and calls whose responses were lost are sent again.
Keep the full run directory, including `model_calls.jsonl`: previous calls,
including failed or interrupted requests, still consume the original budget.
Configuration and provenance remain unchanged; usage and phase counts accumulate
across resumptions. A moved run directory is supported if its input paths still exist. Completed legacy checkpoints are accepted;
incomplete legacy streaming checkpoints are rejected before model calls. A legacy
schedule transition is appended to `schedule_changes.jsonl`, leaving the original
manifest untouched. Paper AlphaEvolve retains its separate
`examples.alphaevolve --resume` path. Unified resume for Shinka, Lineage, and
AlphaEvolve is unsupported; generation videos remain Elite/Blackjack-only.

A connection failure can surface as Slick's `ProviderError` wrapping an SDK
`APIConnectionError` or an HTTP read error. Connection failures and SDK timeouts
automatically retry up to three times, waiting roughly 1, 2, then 4 seconds with
jitter. Each attempt is logged and consumes its own budget reservation; SDK retries
remain disabled to avoid untracked requests. The generation timeout covers all
attempts and delays together. Authentication errors, HTTP status errors, invalid
requests, and cancellation are not retried. If retries, the deadline, or the budget
are exhausted, the run retains its checkpoint; use `resume` after connectivity
returns. Resuming does not reset budget limits. Custom optimizers own their
recovery and are not supported by this command.

## Evaluate existing policies

```bash
rsikit evaluate --env ocean:g2048 \
  --policy runs/ocean-elitetable-2048-1/winner.py \
  --seeds 2000 2001 --batch-size 32 --max-steps 2000 \
  --workers 4 --output runs/winner-evaluation
```

Accepts multiple files after `--policy`. A file-only evaluation config uses `env`, `output`, `policies`, `environment`, and `evaluation` (and optional `version`/`command`). Model, optimizer, budget, selection, and video sections are rejected for this command. No API credential is required.

## Results and videos

Every run saves resolved `config.yaml`, provenance in `manifest.json`, policy exports and scores in the existing Run database, per-phase `measurements.jsonl`, and `summary.json`. Model call evidence is recorded separately. The backend retains its raw episode or panel artifacts.

Search uses only search seeds. Validation ranks the returned finalists; the selected policy and `selection.json` are saved before testing. Test results never trigger repairs or reselection. With no held-out panel the summary says `not evaluated`. No valid candidate is reported explicitly; the runner does not replace failed search results with a baseline. Errors and cancellation retain partial evidence.

Counts distinguish requested candidates and seed runs from actual backend submissions. Cache hits do not count as fresh execution. Failed executions count as attempts; work lost in an incomplete Ocean panel is unknown rather than zero. Actual tokens and billing are null when the provider does not report them. Scalar transition counts are currently unknown in the aggregate summary; complete episode traces retain the underlying steps.

`--video-top 4 --video-workers 1` currently supports full-episode Blackjack with EliteSearch. It retains the existing generation-video workflow and writes `videos/index.html`. Unsupported video combinations fail before generation. Ocean videos are not implemented.

## Custom components

A Python file passed as an environment or optimizer is imported as executable code. `--print-config` imports these files to discover options, although it does not construct environments or invoke a provider.

An optimizer file exports `Options`, `add_arguments`, and `optimize`:

```python
from pydantic import BaseModel, ConfigDict, Field


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    proposals: int = Field(default=10, ge=1)


def add_arguments(parser):
    parser.add_argument("--proposals", type=int)


async def optimize(*, task, provider, evaluate, run, options, seed):
    # Generate Policy definitions using provider.acall, then:
    # measurements = await evaluate(policies)
    # Return unique successful finalists ranked by SEARCH evidence, best first.
    return []
```

The `options` dictionary contains validated public options plus reserved `generation` and `videos` dictionaries. Do not use those names for public optimizer options. Generation settings include concurrency, timeout, and token limits; the custom optimizer owns scheduling/deadlines for its calls. All paid calls must go through the supplied provider for usage logging and any explicitly configured budget limits. Budgets are optional for custom optimizers too. Evaluation returns `{policy.id: {seed: episode}}`; candidate failures are stored in `episode.error`. The runner persists measurements and policy definitions; custom optimizers own their algorithm-specific checkpoints.

An environment file exports `environment`, an instance of `research.experiment.EnvironmentDefinition`. Its fields are:

- `Options`: Pydantic model with `extra="forbid"`; declare filesystem options as `Path` for path resolution.
- `add_arguments(parser)`: register `--env-*` flags with destinations matching option fields.
- `describe(options, evaluation)`: return policy instructions and the scoring objective.
- `open_evaluator(*, options, evaluation, run)`: async context manager yielding `async evaluate(policies, seeds)`.
- `evaluation_defaults`, `supported_evaluation_fields`, `score_keys`, and `protocol`: describe the workload and supported settings.
- `provenance()`: optional metadata function, called before generation.

The evaluator owns setup, cleanup, and concurrent execution. It returns exactly the requested policy IDs, one Episode per requested seed, including episodes with `error` set for candidate failures. Infrastructure errors propagate. Use `research.experiment.record_execution(policy_id, seed_runs=..., steps=...)` to report actual work; otherwise custom execution counts are unknown. See `research/environments.py` and `research/ocean/environment.py` for working examples. The optimizer API is identical for both.

## Docker paths and migration

The installed `rsikit` command runs directly in your current Python environment. If you use the optional `scripts/run` Docker launcher, it runs in `/app` and packages project files into its image. Keep configs/components inside the project and reference them by project-relative paths, with outputs under `/app/runs` (normally `runs/...`). External host paths require an explicit read-only Docker mount. The unified launcher initially uses the existing Ocean-capable linux/amd64 image for every environment, including Gymnasium. On Apple Silicon this is emulated; native Python is available for local measurements. Do not compare native and emulated timings as equivalent.

| Old example flag | Unified flag |
|---|---|
| Ocean `--arm elite` | `--optimizer elite` |
| Gym `--concurrency` | `--workers` |
| Gym `--episode-timeout`, Ocean `--timeout` | `--timeout-per-seed` |
| `--shoes-per-seed` | `--env-shoes-per-episode` |
| Gym `--heldout-seeds` | `--test-seeds` |
| Ocean implicit validation/test panels | Explicit `selection` config or seed flags |

The Ocean reference/matrix/timing tools remain in `examples.benchmark_ocean`; the unified runner uses the normal upstream batch path. All four built-in selectors use the file contract; custom optimizer files remain supported.
