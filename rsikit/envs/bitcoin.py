"""Daily, self-financing BTC/USD allocation with proportional trading fees."""

import csv
import math
from datetime import date
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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


class BitcoinEnv(gym.Env):
    """Allocate 0–100% of wealth to BTC; reward sums to terminal cash minus initial cash.

    Default data contains only the seven-year training interval. Validation requires
    explicitly supplying its CSV. Seeds do not change a historical price path.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path=TRAIN_DATA,
        fee_rate=0.001,
        initial_cash=10_000.0,
        *,
        start_date=None,
        end_date=None,
    ):
        fee_rate, initial_cash = float(fee_rate), float(initial_cash)
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("fee_rate must be finite and in [0, 1)")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self._dates, self._prices = load_prices(data_path)
        start = date.fromisoformat(start_date).toordinal() if start_date else self._dates[0]
        end = date.fromisoformat(end_date).toordinal() if end_date else self._dates[-1]
        if start < self._dates[0] or end > self._dates[-1] or start >= end:
            raise ValueError("Date range must contain at least two prices within the CSV")
        left, right = start - self._dates[0], end - self._dates[0] + 1
        self._dates, self._prices = self._dates[left:right], self._prices[left:right]
        self._ratios = tuple(b / a for a, b in zip(self._prices, self._prices[1:]))
        if any(not math.isfinite(r) or r <= 0 for r in self._ratios):
            raise ValueError("Daily price ratios must be finite and positive")
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._index = 0
        self._wealth = self.initial_cash
        self._allocation = self._total_fees = 0.0
        self._done = False
        return self._observation(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        if (
            not isinstance(action, np.ndarray)
            or action.shape != (1,)
            or action.dtype.kind not in "fiu"
        ):
            raise gym.error.InvalidAction("Expected a real array [BTC fraction] in [0, 1]")
        target = float(action[0])
        if not 0 <= target <= 1:  # Also rejects NaN and infinities.
            raise gym.error.InvalidAction("BTC fraction must be finite and in [0, 1]")

        previous = self._wealth
        difference = target - self._allocation
        # Solve x = target * (wealth - fee * abs(x)) - current_btc_value.
        # This targets post-fee allocation exactly, including a fully funded 100% buy.
        trade = (
            difference * previous / (1 + self.fee_rate * target * (1 if difference >= 0 else -1))
        )
        fee = self.fee_rate * abs(trade)
        funded = previous - fee
        btc_value = target * funded * self._ratios[self._index]
        cash = (1 - target) * funded
        self._index += 1
        self._done = self._index == len(self._prices) - 1
        liquidation = btc_value if self._done else 0.0
        if self._done:
            fee += self.fee_rate * btc_value
            cash += btc_value * (1 - self.fee_rate)
            btc_value = 0.0
        self._wealth = cash + btc_value
        self._allocation = btc_value / self._wealth if self._wealth else 0.0
        self._total_fees += fee
        return (
            self._observation(),
            self._wealth - previous,
            self._done,
            False,
            {
                "wealth": self._wealth,
                "fee": fee,
                "total_fees": self._total_fees,
                "trade_usd": trade,
                "liquidation_usd": liquidation,
            },
        )
