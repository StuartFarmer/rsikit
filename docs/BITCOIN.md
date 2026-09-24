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
in `data/bitcoin/validation.csv`, outside the installed package and the environment
directory copied into Docker. The downloaded source URL, timestamps, row counts,
and SHA-256 hashes are in [metadata](../data/bitcoin/metadata.json).
No validation strategy evaluation was performed while implementing this environment.

Refresh explicitly; this changes the optimization problem and should use a new run:

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
cumulative `total_fees`.

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
import asyncio
from examples.bitcoin import Solution  # Buy-and-hold baseline
from rsikit.episode import run_episode

result = asyncio.run(run_episode(BitcoinEnv, Solution))
print(result[4]["episode"])  # Net USD profit and daily steps
```

The existing AlphaEvolve, ShinkaEvolve, LineageSearch and EliteSearch CLIs accept
`--env Bitcoin`. Rebuild the Docker worker to install the new environment and
training CSV. For example, after setting the usual OpenRouter credentials:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
.venv/bin/python -m examples.elitesearch --env Bitcoin \
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
# result = await run_episode(make_validation_env, FrozenPolicy)
# For isolated execution, construct this environment on the host and pass it
# to a separate Run; its loaded price tuples serialize with the environment.
```

The validation panel starts with fresh portfolio and policy state, without
training-period warmup. The training `BitcoinEnv` instance contains no validation
prices. Keep validation results out of generation/repair feedback.

## Verification and speed

```sh
.venv/bin/python -m unittest tests.test_bitcoin
.venv/bin/python -m examples.bitcoin --steps 1000000 --repeats 3
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
