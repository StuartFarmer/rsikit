"""Optional Pillow charts; visual history never enters policy observations."""

from datetime import date
from functools import lru_cache

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK, MUTED = "#1f2933", "#52616b"
GREEN, RED, BLUE, GOLD = "#18705b", "#b44949", "#284b63", "#a87014"


@lru_cache(maxsize=12)
def _font(size):
    return ImageFont.load_default(size=size)


class BitcoinRenderer(gym.Wrapper):
    """Price/fills, fee-adjusted equity, allocation and peak-to-trough drawdown."""

    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env, *, policy_name="Baseline agent", split="training"):
        super().__init__(env)
        self.policy_name, self.split = policy_name, split
        self.history, self.trades = [], []
        self.seed = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.seed = seed
        self.trades = []
        self._peak = self.unwrapped.initial_cash
        self.history = [self._point(observation, 0.0, self._peak)]
        return observation, info

    def _point(self, observation, fees, buy_hold):
        equity = self.unwrapped._wealth
        self._peak = max(self._peak, equity)
        return dict(
            date=date.fromordinal(int(observation[4])).isoformat(),
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
        if info["trade_usd"]:
            self.trades.append(
                dict(
                    index=index,
                    price=self.history[-1]["price"],
                    usd=info["trade_usd"],
                    kind="buy" if info["trade_usd"] > 0 else "sell",
                )
            )
        if info["liquidation_usd"]:
            self.trades.append(
                dict(
                    index=index + 1,
                    price=float(observation[0]),
                    usd=-info["liquidation_usd"],
                    kind="liquidation",
                )
            )
        game = self.unwrapped
        buy_hold = (
            game.initial_cash / (1 + game.fee_rate) * observation[0] / self.history[0]["price"]
        )
        if done:
            buy_hold *= 1 - game.fee_rate
        self.history.append(self._point(observation, info["total_fees"], float(buy_hold)))
        return result

    def render(self):
        if not self.history:
            raise RuntimeError("Call reset() before rendering")
        image = Image.new("RGB", (1280, 720), "#fafaf8")
        draw = ImageDraw.Draw(image)
        current = self.history[-1]
        initial = self.unwrapped.initial_cash
        profit = current["equity"] - initial

        def text(x, y, value, size=13, color=INK, anchor="lt"):
            draw.text((x, y), str(value), font=_font(size), fill=color, anchor=anchor)

        text(28, 20, "BITCOIN / POLICY PERFORMANCE", 25)
        text(
            28,
            54,
            f"{self.policy_name[:72]}  /  {self.split.upper()}  /  seed {self.seed}",
            color=MUTED,
        )
        text(1250, 20, f"${current['equity']:,.2f}", 27, GREEN if profit >= 0 else RED, "rt")
        text(
            1250,
            54,
            f"Net {profit:+,.2f} USD  /  {profit / initial:+.2%}",
            color=MUTED,
            anchor="rt",
        )
        draw.line((28, 79, 1252, 79), fill="#cad3d5")
        text(
            28,
            94,
            f"{self.history[0]['date']} to {current['date']}  /  day {len(self.history) - 1}",
        )
        text(500, 94, f"Fees ${current['fees']:,.2f}  /  {len(self.trades)} fills")
        text(900, 94, f"Max drawdown {max(p['drawdown'] for p in self.history):.2%}")
        text(28, 127, "BTC / USD", 15)
        draw.polygon([(555, 126), (550, 137), (560, 137)], fill=GREEN)
        text(569, 127, "BUY", color=GREEN)
        draw.polygon([(665, 138), (660, 127), (670, 127)], fill=RED)
        text(679, 127, "SELL", color=RED)
        draw.polygon([(785, 126), (791, 132), (785, 138), (779, 132)], fill=RED)
        text(799, 127, "FINAL LIQUIDATION", color=RED)
        text(28, 320, "EQUITY / USD", 15)
        text(550, 320, "Policy", color=BLUE)
        text(660, 320, "Buy & hold (same fees)", color=GOLD)
        text(940, 320, "Starting cash", color=MUTED)
        text(28, 517, "BTC ALLOCATION / %", 14, BLUE)
        text(735, 517, "DRAWDOWN / % BELOW PEAK", 14, RED)

        def chart(keys, colors, bounds, *, limits=None, cash=False):
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
                y = xy(0, value)[1]
                draw.line((left, y, right, y), fill="#e1e6e8")
                label = f"{value:.0%}" if limits else f"{value:,.0f}"
                text(left - 10, y, label, 12, MUTED, "rm")
            if cash:
                y = xy(0, initial)[1]
                draw.line((left, y, right, y), fill="#a6b8b1", width=2)
            for key, color in zip(keys, colors):
                points = [xy(i, p[key]) for i, p in enumerate(self.history)]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
                x, y = points[-1]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            for i in dict.fromkeys((0, (len(self.history) - 1) // 2, len(self.history) - 1)):
                text(xy(i, low)[0], bottom + 8, self.history[i]["date"], 12, MUTED, "mt")
            return xy

        xy = chart(["price"], [BLUE], (108, 159, 1224, 280))
        for trade in self.trades:
            x, y = xy(trade["index"], trade["price"])
            if trade["kind"] == "liquidation":
                points = [(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)]
            else:
                direction = 1 if trade["kind"] == "buy" else -1
                points = [
                    (x, y - 6 * direction),
                    (x - 5, y + 5 * direction),
                    (x + 5, y + 5 * direction),
                ]
            draw.polygon(points, fill=GREEN if trade["kind"] == "buy" else RED)
        chart(["buy_hold", "equity"], [GOLD, BLUE], (108, 355, 1224, 476), cash=True)
        chart(["allocation"], [BLUE], (108, 550, 585, 654), limits=(0, 1))
        chart(
            ["drawdown"],
            [RED],
            (750, 550, 1224, 654),
            limits=(0, max(0.01, max(p["drawdown"] for p in self.history))),
        )
        text(
            28,
            695,
            "Fresh cash and policy state per seed. Seeds change policy randomness; dates determine the market sample.",
            12,
            MUTED,
        )
        return np.asarray(image).copy()
