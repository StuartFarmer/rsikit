"""SVG performance charts; visual history never enters policy observations."""

from datetime import date

import gymnasium as gym

from .render_theme import BLUE, FPS, GOLD, GREEN, GRID, MUTED, RED, RULE
from .svg_frame import SvgFrame, rasterize


class BitcoinRenderer(gym.Wrapper):
    """Price/fills, fee-adjusted equity, allocation and peak-to-trough drawdown."""

    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}
    title = "Bitcoin"
    price_label = "BTC / USD"
    allocation_label = "BTC"
    currency = "USD"
    currency_symbol = "$"
    step_label = "day"
    value_key = "usd"

    def __init__(self, env, *, policy_name="Baseline agent", split="training", policy_id=""):
        super().__init__(env)
        self.policy_name, self.split = policy_name, split
        self.policy_id = policy_id
        self.history, self.trades = [], []
        self.seed = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.seed = seed
        self.trades = []
        self._peak = self.unwrapped.initial_cash
        self.history = [self._point(observation, 0.0, self._peak)]
        return observation, info

    def _date_label(self, observation):
        return date.fromordinal(int(observation[4])).isoformat()

    def _point(self, observation, fees, buy_hold):
        equity = self.unwrapped._wealth
        self._peak = max(self._peak, equity)
        return dict(
            date=self._date_label(observation),
            price=float(observation[0]),
            equity=equity,
            fees=fees,
            allocation=float(observation[2]),
            buy_hold=buy_hold,
            drawdown=1 - equity / self._peak,
        )

    def step(self, action):
        result = self.env.step(action)
        observation, _, done, _, info = result
        index = len(self.history) - 1
        game = self.unwrapped
        trade, liquidation = info[game._trade_key], info[game._liquidation_key]
        if trade:
            self.trades.append(
                {
                    "index": index,
                    "price": self.history[-1]["price"],
                    self.value_key: trade,
                    "kind": "buy" if trade > 0 else "sell",
                }
            )
        if liquidation:
            self.trades.append(
                {
                    "index": index + 1,
                    "price": float(observation[0]),
                    self.value_key: -liquidation,
                    "kind": "liquidation",
                }
            )
        buy_hold = (
            game.initial_cash / (1 + game.fee_rate) * observation[0] / self.history[0]["price"]
        )
        if done:
            buy_hold *= 1 - game.fee_rate
        self.history.append(self._point(observation, info["total_fees"], float(buy_hold)))
        return result

    def render(self):
        return rasterize(self.render_svg(embed_fonts=False))

    def render_svg(self, *, embed_fonts=True):
        """Return the source SVG, with selectable text and vector curves/fills."""
        if not self.history:
            raise RuntimeError("Call reset() before rendering")
        frame = SvgFrame(f"{self.title} policy performance")
        text = frame.text
        current = self.history[-1]
        initial = self.unwrapped.initial_cash
        profit = current["equity"] - initial
        identity = f"   /   policy {self.policy_id[:12]}" if self.policy_id else ""
        frame.header(
            f"{self.title} / policy performance",
            self.policy_name,
            f"{self.split.upper()}   /   seed {self.seed}   /   {self.step_label} {len(self.history) - 1}"
            f"   /   {self.history[0]['date']} to {current['date']}{identity}",
            f"{self.currency_symbol}{current['equity']:,.2f}",
            f"Net {profit:+,.2f} {self.currency}  /  {profit / initial:+.2%}",
            GREEN if profit >= 0 else RED,
        )
        max_drawdown = max(p["drawdown"] for p in self.history)
        text(32, 136, self.price_label, 20, face="serif", max_width=450)
        text(
            1248,
            140,
            f"Fees {self.currency_symbol}{current['fees']:,.2f}   /   {len(self.trades)} fills   /   Max drawdown {max_drawdown:.2%}",
            16,
            MUTED,
            "rt",
            max_width=735,
        )
        text(32, 348, f"Equity / {self.currency}", 20, face="serif", max_width=440)
        frame.line([(538, 359), (570, 359)], BLUE, 2)
        text(580, 351, "Policy", 16, BLUE)
        frame.line([(676, 359), (708, 359)], GOLD, 2, dash="8 5")
        text(718, 351, "Buy & hold (same fees)", 16, GOLD)
        frame.line([(1010, 359), (1042, 359)], MUTED, 1, dash="3 5")
        text(1052, 351, "Starting cash", 16, MUTED)
        text(32, 528, f"{self.allocation_label} allocation / %", 20, face="serif", max_width=600)
        text(718, 528, "Drawdown / % below peak", 20, face="serif")

        def chart(keys, colors, bounds, *, limits=None, cash=False, date_y=None):
            left, top, right, bottom = bounds
            values = [p[k] for p in self.history for k in keys]
            low, high = limits or (
                min(values + ([initial] if cash else [])),
                max(values + ([initial] if cash else [])),
            )
            if high == low:
                low, high = low - max(1, abs(low) * 0.05), high + max(1, abs(high) * 0.05)

            def xy(index, value):
                return (
                    left + (right - left) * index / max(1, len(self.history) - 1),
                    bottom - (value - low) / (high - low) * (bottom - top),
                )

            for i in range(3):
                value = low + (high - low) * i / 2
                py = xy(0, value)[1]
                frame.line([(left, py), (right, py)], GRID)
                label = f"{value:.0%}" if limits else self._value_label(value)
                text(
                    left - 12,
                    py,
                    label,
                    15,
                    MUTED,
                    "rm",
                    max_width=left - (650 if left > 650 else 12),
                )
            if cash:
                py = xy(0, initial)[1]
                frame.line([(left, py), (right, py)], MUTED, dash="3 5")
            for key, color in zip(keys, colors):
                points = [xy(i, p[key]) for i, p in enumerate(self.history)]
                if len(points) > 1:
                    frame.line(
                        points,
                        color,
                        2,
                        dash="8 5" if key == "buy_hold" else None,
                        id=key.replace("_", "-"),
                    )
                x, y = points[-1]
                frame.circle(x, y, 3, color, id=f"{key}-current")
            indices = dict.fromkeys((0, (len(self.history) - 1) // 2, len(self.history) - 1))
            for i in indices:
                anchor = "lt" if i == 0 else "rt" if i == len(self.history) - 1 else "mt"
                text(
                    xy(i, low)[0],
                    date_y or bottom + 12,
                    self.history[i]["date"],
                    14,
                    MUTED,
                    anchor,
                    max_width=(right - left) / 3 - 12,
                )
            return xy

        with frame.group("price-panel"):
            xy = chart(["price"], [BLUE], (106, 181, 1224, 254), date_y=316)
        # All fills stay aligned with price/time, without obscuring the price curve.
        with frame.group("fills"):
            text(32, 281, "Fills", 15, MUTED)
            frame.line([(106, 293), (1224, 293)], RULE)
            for i, trade in enumerate(self.trades):
                x, _ = xy(trade["index"], trade["price"])
                if trade["kind"] == "liquidation":
                    frame.polygon(
                        [(x, 295), (x + 5, 300), (x, 305), (x - 5, 300)], RED, id=f"fill-{i}"
                    )
                elif trade["kind"] == "buy":
                    frame.polygon([(x, 278), (x - 4, 286), (x + 4, 286)], GREEN, id=f"fill-{i}")
                else:
                    frame.polygon([(x, 305), (x - 4, 297), (x + 4, 297)], RED, id=f"fill-{i}")
        with frame.group("equity-panel"):
            chart(["buy_hold", "equity"], [GOLD, BLUE], (106, 389, 1224, 474), cash=True)
        with frame.group("allocation-panel"):
            chart(["allocation"], [BLUE], (106, 569, 595, 646), limits=(0, 1))
        with frame.group("drawdown-panel"):
            chart(["drawdown"], [RED], (768, 569, 1224, 646), limits=(0, max(0.01, max_drawdown)))
        text(
            32,
            697,
            "Fresh cash per seed; seeds change policy randomness, not the market sample.",
            14,
            MUTED,
        )
        text(1248, 697, "Fills: up = buy / down = sell / diamond = liquidation", 14, MUTED, "rt")
        return frame.svg(embed_fonts=embed_fonts)

    def _value_label(self, value):
        return f"{value:,.0f}"
