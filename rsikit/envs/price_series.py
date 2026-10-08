"""Single-asset, self-financing allocation over rows in a local CSV."""

import csv
import hashlib
import io
import math
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _timestamp(value):
    """ISO dates/datetimes; timezone-free input denotes UTC."""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


class PriceSeriesEnv(gym.Env):
    """Allocate 0–100% to one asset; one step advances one CSV row.

    Seeds affect policies, not the historical path. Timestamp spacing may vary.
    The fifth observation is a zero-based bar index within the selected interval.

    Args:
        data_path (str | Path): UTF-8 CSV with at least two finite positive prices.
        fee_rate (float): Proportional fee on each trade, finite and in [0, 1).
        initial_cash (float): Positive finite starting balance in the quote currency.
        price_column (str): CSV price column name.
        time_column (str | None): Timestamp column; None detects `timestamp`, then `date`.
        asset_name (str): Nonempty asset name used in policy instructions.
        start_time (str | None): Optional inclusive ISO timestamp lower bound.
        end_time (str | None): Optional inclusive ISO timestamp upper bound.

    Timestamps must be unique and increasing; naive timestamps mean UTC. Gaps
    are preserved. Observations are float64 arrays containing price, previous
    return, current allocation, wealth / initial_cash, and selected-row index.
    Actions are float64 arrays `[target_asset_fraction]` in [0, 1]. Reward is
    change in wealth after fees; final holdings are liquidated on termination.
    A policy that never trades is rejected with Gymnasium InvalidAction.

    Raises:
        ValueError: CSV data, interval, fee, cash, or asset name are invalid.
        OSError: The CSV cannot be read.
    """

    metadata = {"render_modes": []}
    _trade_key = "trade_value"
    _liquidation_key = "liquidation_value"
    _require_trade = True

    def __init__(
        self,
        data_path,
        fee_rate=0.001,
        initial_cash=10_000.0,
        *,
        price_column="price",
        time_column=None,
        asset_name="asset",
        start_time=None,
        end_time=None,
    ):
        data = Path(data_path).read_bytes()
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        try:
            columns = reader.fieldnames or []
            if len(set(columns)) != len(columns) or not all(columns):
                raise ValueError("CSV columns must be nonempty and unique")
            if price_column not in columns:
                raise ValueError(f"CSV needs a {price_column!r} price column")
            if time_column is None:
                time_column = next((c for c in ("timestamp", "date") if c in columns), None)
            if time_column is not None and time_column not in columns:
                raise ValueError(f"CSV needs a {time_column!r} timestamp column")
            if (start_time is not None or end_time is not None) and time_column is None:
                raise ValueError("Time ranges require a timestamp column")
            prices, timestamps = [], []
            for number, row in enumerate(reader, 2):
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("row does not match the CSV header")
                    price = float(row[price_column])
                    if not math.isfinite(price) or price <= 0:
                        raise ValueError("price must be finite and positive")
                    prices.append(price)
                    if time_column is not None:
                        timestamp = _timestamp(row[time_column])
                        if timestamps and timestamp <= timestamps[-1]:
                            raise ValueError("timestamps must be unique and increasing")
                        timestamps.append(timestamp)
                except ValueError as exc:
                    raise ValueError(f"CSV row {number}: {exc}") from exc
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV near line {reader.line_num}: {exc}") from exc
        left = bisect_left(timestamps, _timestamp(start_time)) if start_time is not None else 0
        right = (
            bisect_right(timestamps, _timestamp(end_time)) if end_time is not None else len(prices)
        )
        self._initialize(tuple(prices[left:right]), fee_rate, initial_cash)
        self._timestamps = tuple(t.isoformat() for t in timestamps[left:right])
        self.dataset = dict(
            path=str(Path(data_path).resolve()),
            sha256=hashlib.sha256(data).hexdigest(),
            price_column=price_column,
            time_column=time_column,
            start_row=left,
            stop_row=right,
            prices=len(self._prices),
            first_timestamp=self._timestamps[0] if self._timestamps else None,
            last_timestamp=self._timestamps[-1] if self._timestamps else None,
        )
        if not isinstance(asset_name, str) or not asset_name.strip():
            raise ValueError("asset_name must be a nonempty string")
        self.asset_name = asset_name
        self.instructions = (
            f"Maximize wealth gained trading {asset_name} against cash in the CSV's quote currency. "
            "Each action is a float64 array [target_asset_fraction] in [0, 1], measured AFTER "
            "trading fees. The remainder stays in non-interest-bearing cash. No leverage or shorts. "
            f"Start with {self.initial_cash:g} cash. Every buy and sell costs {self.fee_rate:.8g} "
            "times its traded value. Trade at the currently observed price, then advance one CSV "
            "row. Rows need not be equally spaced in time. These are idealized observed-price fills. "
            "Reward is the change in wealth including fees; total reward equals final cash minus "
            "starting cash. Remaining holdings are sold with fees at the last selected price. "
            "Observation float64[5]: [0] current price, [1] previous bar's simple return (zero at "
            "reset), [2] current asset fraction before trading, [3] wealth / initial_cash, "
            "[4] zero-based bar index within this episode. Only current and past prices are "
            "observed. Maintain history in policy memory. Repeating a target rebalances drift "
            "and may incur a fee; returning the current fraction holds without trading. "
            "You must execute at least one nonzero-value trade per episode; a strategy that "
            "stays entirely in cash is rejected, not scored. Buying once and holding qualifies. "
            "Every reset replays the same selected rows, independent of seed."
        )

    def _initialize(self, prices, fee_rate, initial_cash):
        """Shared accounting setup for CSV series and the compatible Bitcoin preset."""
        fee_rate, initial_cash = float(fee_rate), float(initial_cash)
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("fee_rate must be finite and in [0, 1)")
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if len(prices) < 2:
            raise ValueError("Selected interval needs at least two prices")
        self._prices = prices
        self._ratios = tuple(b / a for a, b in zip(prices, prices[1:]))
        if any(not math.isfinite(r) or r <= 0 for r in self._ratios):
            raise ValueError("Price ratios must be finite and positive")
        self.fee_rate, self.initial_cash = fee_rate, initial_cash
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            np.array([0, -1, 0, 0, 0], dtype=np.float64),
            np.array([np.inf, np.inf, 1, np.inf, np.inf], dtype=np.float64),
            dtype=np.float64,
        )
        self._done = True

    def _observation(self):
        return np.array(
            [
                self._prices[self._index],
                self._ratios[self._index - 1] - 1 if self._index else 0,
                self._allocation,
                self._wealth / self.initial_cash,
                self._index,
            ],
            dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._index = 0
        self._wealth = self.initial_cash
        self._allocation = self._total_fees = 0.0
        self._has_traded = False
        self._done = False
        return self._observation(), self._time_info()

    def _time_info(self):
        return {"timestamp": self._timestamps[self._index]} if self._timestamps else {}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        if (
            not isinstance(action, np.ndarray)
            or action.shape != (1,)
            or action.dtype.kind not in "fiu"
        ):
            raise gym.error.InvalidAction("Expected a real array [asset fraction] in [0, 1]")
        target = float(action[0])
        if not 0 <= target <= 1:  # Also rejects NaN and infinities.
            raise gym.error.InvalidAction("Asset fraction must be finite and in [0, 1]")

        previous = self._wealth
        difference = target - self._allocation
        # Solve x = target * (wealth - fee * abs(x)) - current_asset_value.
        # This targets post-fee allocation exactly, including a fully funded 100% buy.
        trade = (
            difference * previous / (1 + self.fee_rate * target * (1 if difference >= 0 else -1))
        )
        self._has_traded = self._has_traded or trade != 0.0
        fee = self.fee_rate * abs(trade)
        funded = previous - fee
        asset_value = target * funded * self._ratios[self._index]
        cash = (1 - target) * funded
        self._index += 1
        self._done = self._index == len(self._prices) - 1
        if self._done and self._require_trade and not self._has_traded:
            raise gym.error.InvalidAction(
                "Strategy made no trades: staying entirely in cash is rejected"
            )
        liquidation = asset_value if self._done else 0.0
        if self._done:
            fee += self.fee_rate * asset_value
            cash += asset_value * (1 - self.fee_rate)
            asset_value = 0.0
        self._wealth = cash + asset_value
        self._allocation = asset_value / self._wealth if self._wealth else 0.0
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
                self._trade_key: trade,
                self._liquidation_key: liquidation,
                **self._time_info(),
            },
        )
