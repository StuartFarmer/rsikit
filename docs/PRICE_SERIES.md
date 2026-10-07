# Strategies from a single CSV

`PriceSeriesEnv` searches for single-asset allocation strategies using one local
CSV. The minimum format is a `price` header and at least two finite, positive
prices. Optional `timestamp` or `date` columns allow irregular or intraday data:

```csv
timestamp,price
2024-01-05T09:00:00Z,100
2024-01-08T09:00:00Z,102
2024-01-08T09:05:00Z,99
```

Rows are consumed in file order. Timestamps must be ISO dates or datetimes, unique
and increasing in UTC; timezone-free values mean UTC. Weekend gaps and irregular
spacing are allowed. Nothing is sorted, interpolated, or forward-filled.
`timestamp` is detected first, then `date`; without either, only row order matters.
For other headers, use `--env-price-column Close --env-time-column Date`.
Extra columns are ignored. Invalid prices, duplicate headers, malformed rows,
and invalid or unordered timestamps fail before model calls.

## Run

With the existing OpenRouter setup:

```bash
.venv/bin/python -m research.cli run \
  --env PriceSeries --env-data-path prices.csv \
  --optimizer elite --model gpt-oss-120b:nitro \
  --population 100 --generations 25 --elites 20 --max-repairs 3 \
  --workers 16 --generation-concurrency 50 \
  --spend-cap 100 --input-price 0.01 --output-price 0.01 \
  --output "runs/prices-$(date +%Y%m%d-%H%M%S)"
```

The price ceilings follow the existing CLI examples; set them for your provider
as described in [CLI.md](CLI.md). `--print-config` checks configuration without
starting a run; it does not load the CSV.

| Option | Default | Meaning |
| --- | --- | --- |
| `--env-data-path` | Required | CSV path |
| `--env-price-column` | `price` | Price column |
| `--env-time-column` | Detect `timestamp`, then `date` | Optional ISO timestamp column |
| `--env-asset-name` | `asset` | Asset label in policy instructions and rendered charts |
| `--env-fee-rate` | `0.001` | Proportional fee on buys and sells |
| `--env-initial-cash` | `10000` | Starting wealth, in the CSV's quote currency |
| `--env-start-time`, `--env-end-time` | Whole file | Inclusive timestamp bounds |

In YAML, these fields go under `environment` without `env-`, using underscores.
CSV paths in YAML resolve relative to the config; flag paths resolve from the
working directory. No downloader or additional data file is required.

## Trading and observations

Actions are NumPy arrays `[target_asset_fraction]` in `[0, 1]`, measured after
fees. Each step trades at the observed price and advances one row. The final step
liquidates all remaining holdings and pays the sell fee. Total reward equals
final cash minus starting cash.

Strategies must execute at least one nonzero-value trade per episode. Staying
entirely in cash fails evaluation and cannot enter the leaderboard, even when
trading strategies lose money. Buying once and holding qualifies, including with
zero fees. The generated-policy instructions state this requirement. Failures
use the normal policy-error/repair path rather than a non-finite numeric score.

Observation float64 arrays contain `[price, previous_bar_return, asset_fraction,
wealth / initial_cash, bar_index]`. The zero-based bar index restarts for every
selected interval; the first return is zero. Policies maintain their own history.
Observations contain no future prices or full-series normalization statistics.
Episode `info` records wealth, fees, `trade_value`, `liquidation_value`, and the
current UTC timestamp when supplied. Timestamps are not part of policy observations.

Accounting is shared with Bitcoin: fractional holdings, no shorts or leverage,
no cash interest, unlimited liquidity, and idealized observed-price fills.
Spreads, slippage, and corporate actions are not simulated; supply the price
representation appropriate for your experiment.

## Separate periods from the same file

By default, search uses the entire CSV. Add `--env-end-time 2023-01-01` to train
through a boundary, then evaluate the saved winner on the later part of that file:

```bash
.venv/bin/python -m research.cli evaluate \
  --env PriceSeries --env-data-path prices.csv \
  --env-start-time 2023-01-01 \
  --policy runs/YOUR_SEARCH/winner.py \
  --output runs/prices-later-period
```

Use the same column and accounting settings for both runs. Bounds require a
timestamp column and at least two selected prices. Date-only bounds mean midnight
UTC; use full timestamps for intraday boundaries. Runs start with fresh cash and
policy memory and liquidate independently. Sharing one boundary price does not
share a return when it is the last training and first evaluation price.

There is no automatic split. The default seed is `0`; other seeds replay the
same selected prices and only change stochastic policies. For unseen-market
evaluation use different time bounds in a separate run, not `--validation-seeds`
or `--test-seeds` on the training interval. `--max-steps` is rejected: select an
interval to retain final liquidation in the score.

Each run saves `dataset.json` with the CSV SHA-256, resolved path, selected
columns, zero-based row range (`start_row` inclusive, `stop_row` exclusive), and
first/last selected timestamps. Keep the original CSV with the run archive.
The usual configuration, policies, episodes, and summaries are also saved.

## Python

```python
from rsikit.envs import PriceSeriesEnv

env = PriceSeriesEnv("prices.csv", fee_rate=0.001)
observation, info = env.reset()
```

## Render a saved winner

With the `video` extra installed, export the top policy from the final generation
(25 in this example):

```bash
.venv/bin/python -m research.elitesearch.videos runs/YOUR_SEARCH \
  --generation 25 --top 1 --workers 1 --output runs/YOUR_SEARCH/videos
```

Open `videos/index.html` for the MP4 and final PNG chart. Replay loads the saved
CSV columns, time range, asset name, fees, starting cash, and seeds from
`config.yaml`, checks the dataset against `dataset.json`, and verifies the replay
score against the search score. No model calls are needed.

The layout matches Bitcoin: price and fills, equity versus buy-and-hold with the
same fees, allocation, and drawdown. Labels use `asset_name`; amounts are in the
CSV's quote currency. The horizontal axis uses timestamps when present and bar
indices otherwise. Each seed starts with fresh cash and policy state.

For a frame directly from Python, wrap the environment before resetting it:

```python
from rsikit.envs.price_series_render import PriceSeriesRenderer

env = PriceSeriesRenderer(PriceSeriesEnv("prices.csv", asset_name="EUR/USD"))
observation, info = env.reset()
frame = env.render()  # uint8 RGB array, 1280 x 720
```

`BitcoinEnv` retains its bundled data, daily-date validation, date-ordinal
observations, USD info fields, and existing chart/replay support.
