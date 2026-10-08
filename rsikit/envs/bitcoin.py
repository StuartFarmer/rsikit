"""Daily, self-financing BTC/USD allocation with proportional trading fees."""

import csv
import math
from datetime import date
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .price_series import PriceSeriesEnv

TRAIN_DATA = Path(__file__).with_name("data") / "bitcoin_train.csv"


def load_prices(path):
    """Load strictly consecutive daily prices; never fill gaps or missing values."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    dates = tuple(date.fromisoformat(row["date"]).toordinal() for row in rows)
    prices = tuple(float(row["price_usd"]) for row in rows)
    if len(prices) < 2 or any(not math.isfinite(p) or p <= 0 for p in prices):
        raise ValueError("Need at least two finite, positive daily prices")
    if any(b - a != 1 for a, b in zip(dates, dates[1:])):
        raise ValueError("Price dates must be consecutive, unique, and increasing")
    return dates, prices


class BitcoinEnv(PriceSeriesEnv):
    """Allocate 0–100% of wealth to BTC; reward sums to terminal cash minus initial cash.

    Default data contains only the seven-year training interval. Validation requires
    explicitly supplying its CSV. Seeds do not change a historical price path.

    Args:
        data_path (str | Path): CSV with `date` and `price_usd` columns, at least two finite
            positive prices, and consecutive increasing ISO dates.
        fee_rate (float): Proportional fee on each trade, finite and in [0, 1).
        initial_cash (float): Positive finite starting USD balance.
        start_date (str | None): Optional inclusive ISO date within the CSV.
        end_date (str | None): Optional inclusive ISO date within the CSV, after start_date.

    Actions are float64 arrays `[target_btc_fraction]` in [0, 1]. Observations
    contain price, last return, current allocation, wealth / initial_cash, and
    date ordinal. Trades execute at the observed price before advancing one day.
    Final holdings are liquidated with fees. Staying entirely in cash is allowed.
    The bundled Coin Metrics data retains CC BY-NC 4.0 terms.

    Raises:
        ValueError: Prices, dates, interval, fee, or starting cash are invalid.
        OSError: The CSV cannot be read.
    """

    _trade_key = "trade_usd"
    _liquidation_key = "liquidation_usd"
    _timestamps = ()
    _require_trade = False  # Preserve the original Bitcoin cash-baseline contract.

    def __init__(
        self,
        data_path=TRAIN_DATA,
        fee_rate=0.001,
        initial_cash=10_000.0,
        *,
        start_date=None,
        end_date=None,
    ):
        self._dates, self._prices = load_prices(data_path)
        start = date.fromisoformat(start_date).toordinal() if start_date else self._dates[0]
        end = date.fromisoformat(end_date).toordinal() if end_date else self._dates[-1]
        if start < self._dates[0] or end > self._dates[-1] or start >= end:
            raise ValueError("Date range must contain at least two prices within the CSV")
        left, right = start - self._dates[0], end - self._dates[0] + 1
        self._dates, self._prices = self._dates[left:right], self._prices[left:right]
        self._initialize(self._prices, fee_rate, initial_cash)
        fee_rate, initial_cash = self.fee_rate, self.initial_cash
        self.observation_space = spaces.Box(
            np.array([0, -1, 0, 0, 1], dtype=np.float64),
            np.array([np.inf, np.inf, 1, np.inf, date.max.toordinal()], dtype=np.float64),
            dtype=np.float64,
        )
        self._done = True
        self.instructions = (
            "Maximize total wealth gained trading BTC against USD cash. Each action is a "
            "float64 array [target_btc_fraction] in [0, 1], measured AFTER trading fees. "
            "The remainder stays in non-interest-bearing cash. No leverage or shorts. "
            f"Start with ${initial_cash:g} cash. Every buy and sell costs {fee_rate:.8g} "
            "times its USD notional. Trade at the currently observed daily price, then "
            "advance to the next day's price. This is an idealized daily-price fill. "
            "Reward is the change in USD wealth, including fees; total reward equals "
            "final cash minus starting cash. Remaining BTC is sold with fees at the last "
            "price. Observation float64[5]: [0] current BTC/USD price, [1] last daily "
            "simple return (zero at reset), [2] current BTC fraction before your trade, "
            "[3] current wealth / initial_cash, [4] date ordinal (date.fromordinal). "
            "Only current and past data are visible. Maintain your own price history in "
            "policy memory. Repeating a target rebalances drift and may incur a fee; "
            "returning the current fraction holds without trading. Reset always starts "
            "the same chronological path, independent of seed."
        )

    def _observation(self):
        return np.array(
            [
                self._prices[self._index],
                self._ratios[self._index - 1] - 1 if self._index else 0,
                self._allocation,
                self._wealth / self.initial_cash,
                self._dates[self._index],
            ],
            dtype=np.float64,
        )
