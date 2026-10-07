# Bitcoin allocation

`BitcoinEnv` evolves daily BTC/USD portfolio allocations to maximize **final cash
minus starting cash**, after transaction fees. It uses existing NumPy and
Gymnasium dependencies; there is no network access during construction or steps.

```python
import numpy as np
from rsikit.envs import BitcoinEnv

env = BitcoinEnv(fee_rate=0.001, initial_cash=10_000)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(np.array([0.75]))
```

## Data and split

Downloaded from the [Coin Metrics Community API](https://docs.coinmetrics.io/api/v4/),
using daily `PriceUSD`, on September 19, 2026. The complete dataset covers
September 19, 2016 through September 18, 2026: 3,652 consecutive daily prices.
These are USD reference prices, not exchange-specific executable quotes.

| Panel | Prices | First date | Last date | Daily steps |
| --- | ---: | --- | --- | ---: |
| Training (default) | 2,557 | 2016-09-19 | 2023-09-19 | 2,556 |
| Reserved validation | 1,096 | 2023-09-19 | 2026-09-18 | 1,095 |

The boundary price is shared, but **no daily return overlaps**. A training action
at September 18, 2023 earns the return to September 19 and liquidates there;
the first validation action at September 19 earns the return to September 20.
Both panels start with cash and end with a fee-paying sale of any BTC.

Training lives in `rsikit/envs/data/bitcoin_train.csv`. Validation lives separately
in `data/bitcoin/validation.csv`, outside the installed library package. Both panels are included in the
application image. The downloaded source URL, timestamps, row counts,
and SHA-256 hashes are in [metadata](../data/bitcoin/metadata.json).
Validation is evaluated only when explicitly selected for replay, never by the
automatic per-generation training video export.

Refresh bundled datasets on the host, then let the launcher rebuild the image.
This changes the optimization problem and should use a new run:

```sh
.venv/bin/python -m examples.download_bitcoin --as-of 2026-09-19
```

The downloader checks complete calendar coverage and valid prices before replacing
either panel. It never interpolates, forward-fills, or silently shortens a window.
`--source /path/to/download.csv` processes an already downloaded API CSV offline.
Dates are pinned in the shipped snapshot, not moved on each reset.
Coin Metrics data is attributed to [Coin Metrics](https://coinmetrics.io/) under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), as documented in
its [data archive](https://github.com/coinmetrics/data). The derived CSVs retain
dates and prices and split them into the two panels.

## Trading and objective

An action is a float64 NumPy array of shape `(1,)`, with target BTC fraction in
`[0, 1]`: zero means cash, one means fully invested. The target is measured **after
the trade's fee**. Cash earns no interest; there is no borrowing or shorting.
Invalid actions raise `gymnasium.error.InvalidAction` without advancing time.

At observed price `p[t]`, the simulator rebalances to the requested fraction and
charges `fee_rate * abs(traded_usd)` on buys and sells. It then advances to
`p[t+1]`. If `W` is current wealth, `B` is current BTC value, `a` is the target,
and `f` is the fee rate, the signed trade value is:

```text
x = (a * W - B) / (1 + sign(a * W - B) * f * a)
post_trade_wealth = W - f * abs(x)
cash = (1 - a) * post_trade_wealth
btc_value_next_day = a * post_trade_wealth * p[t+1] / p[t]
```

The final step also sells all remaining BTC and pays the sell fee. Each reward
is `wealth_after_step - wealth_before_step`, so undiscounted episode reward sums
to net USD profit. This objective favors absolute wealth, with no risk penalty or
log-return transformation. `info` contains `wealth`, that step's total `fee`, and
cumulative `total_fees`, signed `trade_usd` at the observed price, and
`liquidation_usd` sold at the final price (zero on nonterminal steps).

Repeating 50% rebalances the allocation after price changes. To hold your current
BTC quantity without trading, return `np.array([observation[2]])`. Staying at 100%
buys once and holds until liquidation; staying at 0% earns exactly zero.

This assumes fractional BTC, unlimited liquidity, and fills at the observed daily
reference price. It does not model spread, slippage, intraday execution, taxes, or
cash yield. The fee knob is configurable; add execution detail when those effects
are part of the experiment.

## Observations and rsikit

Each observation is a fresh float64 array; previous observations remain unchanged.

| Index | Meaning |
| --- | --- |
| 0 | Current BTC price in USD |
| 1 | Most recent simple daily return; zero at reset |
| 2 | Current fraction of wealth in BTC, before the next action |
| 3 | Current wealth divided by initial cash |
| 4 | Python date ordinal; decode with `date.fromordinal(int(obs[4]))` |

No future prices or full-series normalization statistics appear in observations
or `info`. Policies maintain their own history in memory. The environment's
`instructions` describes the complete contract for generated policies.

```python
from copy import deepcopy
from examples.bitcoin import Solution
from rsikit import Evaluator

# Inside an async function:
with BitcoinEnv() as env:
    policy = Solution(deepcopy(env.observation_space), deepcopy(env.action_space))
    try:
        episode = await Evaluator().evaluate(policy, env, seed=0)
    finally:
        await policy.close()
print(episode.total_reward, len(episode))
```

The existing AlphaEvolve, ShinkaEvolve, LineageSearch and EliteSearch CLIs accept
`--env Bitcoin`. Rebuild the application image to install the new environment and
training CSV. For example, after setting the usual OpenRouter credentials:

```sh
./scripts/run examples.elitesearch --env Bitcoin \
  --seeds 0 --heldout-seeds 1 --generations 3 --population 8 --elites 3
```

This command makes paid model calls. Leave `--max-steps` unset to evaluate full
episodes and include terminal liquidation. A TimeLimit measures wealth at its
truncation point without the final sale.

**Different seeds are not different market samples.** Reset always replays the
same chronology. The existing CLI's `--heldout-seeds` changes the policy's seed,
not the data panel; its score is still training-period performance. Use a single
training seed for deterministic policies. Historical data is public and policies
can memorize it, so reserve chronological validation for the final frozen policy.

To validate later, explicitly create a separate environment and separate `Run`:

```python
from functools import partial

make_validation_env = partial(BitcoinEnv, data_path="data/bitcoin/validation.csv")
# Only after freezing the policy:
# Create policy and environment instances; Evaluator resets them.
# For isolated execution, construct this environment on the host and pass it
# to Executor with a separate Run for storage; loaded price tuples serialize.
```

The validation panel starts with fresh portfolio and policy state, without
training-period warmup. The training `BitcoinEnv` instance contains no validation
prices. Keep validation results out of generation/repair feedback.

## Performance charts and videos

With the existing `video` extra installed, export a saved EliteSearch generation:

```sh
./scripts/run examples.bitcoin_videos runs/bitcoin-smoke1 \
  --generation 14 --top 1 --seeds 0 --output runs/bitcoin-training-videos
./scripts/run examples.bitcoin_videos runs/bitcoin-smoke1 \
  --generation 14 --top 1 --split validation --seeds 100 \
  --output runs/bitcoin-validation-videos
```

Open `index.html` in the output directory. Each policy has a 1280×720 MP4 at
30 fps, one day/action per frame, and final PNG and standalone SVG charts for the
last seed. SVG is the source frame: `render_svg()` returns vector geometry/text,
and `render()` rasterizes the same source with resvg for Gymnasium and video.
Install the `video` extra; Latin Modern fonts are bundled, with no TeX requirement.

Each MP4 also has a `.svg.zip` archive containing one SVG per action, a shared
`fonts/` directory, and `timeline.json` with fps and seed boundaries. The archive
preserves the source vectors for later web animation. Extract it intact to preserve
font paths; standalone final SVGs embed their fonts. Rendering remains independent
of training; style-version changes reuse recorded action traces.

The charts show BTC price with buy/sell fills in an aligned strip and a distinct final-liquidation
marker, equity after fees, a buy-and-hold baseline with the same fees, BTC
allocation, and drawdown from the running equity peak. Net profit, cumulative
fees, and maximum drawdown are displayed. `manifest.json` includes per-seed
daily equity, prices, allocations, fees, drawdowns, and signed fill notionals.
Fill `index` refers to the corresponding daily history row. Equity includes
cash and the current market value of BTC; the last point is liquidated cash.

`--split training` uses the saved search seeds and training CSV. `--split validation`
(or `--split holdout`) uses the saved held-out seeds and reserved validation CSV.
`--seeds` overrides the policy seeds without changing the market panel. Policies
are always selected by their **training** rankings; evaluation does not alter the
source run or feed validation scores into optimization. Use `--generation` to
evaluate a frozen generation, or omit it to export all completed generations.
Repeatedly inspecting validation to choose a policy makes it selection data;
reserve another unseen period for the final assessment.

For an explicit section of the selected CSV, add inclusive bounds:

```sh
./scripts/run examples.bitcoin_videos runs/bitcoin-smoke1 \
  --generation 14 --top 1 --split validation \
  --start-date 2024-01-01 --end-date 2024-12-31
```

`--data-path /path/to/prices.csv` accepts another consecutive daily price CSV.
Validation returns must follow the packaged training period; overlapping ranges
are rejected. Both bounds must lie within the selected CSV and include at least
two prices. Every section starts with fresh cash and policy memory, with no
earlier-price warmup. The same `start_date`/`end_date` keyword arguments work on
`BitcoinEnv`. Cache identity includes dates, data contents, split, and seeds so
training traces cannot be reused as validation traces.

Generated policies execute in the application container; rendering replays
their actions locally and checks the resulting score. No model calls are made.
For training videos after every generation, use
`examples.elitesearch --env Bitcoin --video-top 4`; this exports only training
data. Held-out date evaluation remains an explicit, separate replay command.

For a trusted policy's own rendering loop:

```python
from rsikit.envs.bitcoin_render import BitcoinRenderer

with BitcoinRenderer(BitcoinEnv(), policy_name="My policy") as env:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # uint8 RGB, shape (720, 1280, 3)
```

## Verification and speed

```sh
.venv/bin/python -m unittest tests.test_bitcoin
./scripts/run examples.bitcoin --steps 1000000 --repeats 3
```

The [recorded benchmark](bitcoin-benchmark-2026-09-19.json) measured **709,982 daily
steps/second** median on macOS arm64 / Python 3.14.2, across three one-million-step
runs. Each run included 391 episode resets. The benchmark cycles through cash,
partial, and full allocations, exercising buys, sells, fees, and terminal sales.
It exits with an error if median throughput falls below 100,000 steps/second.

Timing includes action selection, real `env.step` calls, new observations,
reward accumulation, and resets. Imports and initial CSV loading are excluded.
This is local environment throughput; the asynchronous runner's copying and
validation, Docker transport, and model inference add costs. No claim is made
that the entire optimization pipeline runs at this rate.
