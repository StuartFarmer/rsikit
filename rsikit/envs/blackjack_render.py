"""Optional Pillow table renderer; wrap a BlackjackEnv only when making visuals."""

import math
from copy import deepcopy
from functools import cached_property, lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blackjack import _value

WIDTH, HEIGHT = 1280, 720
ASSETS = Path(__file__).with_name("data") / "blackjack"
INK = "#1F2933"
MUTED = "#52616B"
ACCENT = "#284B63"
GREEN = "#18705B"
RED = "#B44949"
SUITS = ("hearts", "diamonds", "clubs", "spades")
RANKS = ("A", *[f"{n:02}" for n in range(2, 11)], "J", "Q", "K")


class _Card(int):
    """A normal blackjack value carrying a private, exact deck identity for drawing."""

    def __new__(cls, value, identity):
        card = super().__new__(cls, value)
        card.identity = identity
        return card

    def __getnewargs__(self):
        return int(self), self.identity


@lru_cache(maxsize=24)
def _font(size):
    return ImageFont.load_default(size=size)


class BlackjackRenderer(gym.Wrapper):
    """One deterministic RGB frame per action, with a cumulative reward timeline.

    Card identities reproduce the base environment's shuffle with a separate RNG.
    Neither rendering nor the visual history is included in policy observations.
    """

    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env, *, policy_name="Baseline agent"):
        super().__init__(env)
        self.policy_name = policy_name
        self.seed = None
        self.points = [0.0]
        self._dealt = self._settled = self._sat_out = False
        self._action = "READY"
        self._detail = "Place a wager to begin"
        self._reward = 0.0

    def reset(self, *, seed=None, options=None):
        rng = (
            np.random.default_rng(seed) if seed is not None else deepcopy(self.unwrapped.np_random)
        )
        observation, info = self.env.reset(seed=seed, options=options)
        self.seed = seed
        self._identify_shoe(rng)
        self.points = [0.0]
        self._dealt = self._settled = self._sat_out = False
        self._action = "READY"
        self._detail = "Place a wager to begin"
        self._reward = 0.0
        return observation, info

    def _identify_shoe(self, rng):
        game = self.unwrapped
        identities = rng.permutation(len(game._template)) % 52
        game._shoe = [
            _Card(value, int(identity)) for value, identity in zip(game._shoe, identities)
        ]

    def step(self, action):
        game = self.unwrapped
        phase = game._phase
        active = game._active if phase else 0
        wager = game._bets[active] if phase else None
        rng = deepcopy(game.np_random)
        result = self.env.step(action)
        _, reward, _, _, info = result
        if info.get("shoe_shuffled"):
            self._identify_shoe(rng)
        self.points.append(self.points[-1] + reward)
        self._dealt = True
        self._settled = info.get("round_complete", False)
        self._sat_out = info.get("sat_out", False)
        self._reward = reward
        if phase == 0:
            wager = game.bet_sizes[action]
            self._action = "BET" if wager else "SIT OUT"
            self._detail = f"Wager {wager} points" if wager else "Observing the other player"
        else:
            self._action = ("STAND", "HIT", "DOUBLE", "SPLIT")[action]
            self._detail = f"Hand {active + 1}"
            if action == 2:
                self._detail += f"   /   Wager {wager} to {2 * wager} points"
            elif action == 3:
                self._detail += f"   /   {len(game._hands)} hands in play"
        return result

    @cached_property
    def _background(self):
        image = Image.new("RGB", (WIDTH, HEIGHT), "#FAFAF8")
        draw = ImageDraw.Draw(image)
        draw.line((28, 76, 1252, 76), fill="#CAD3D5")
        draw.rounded_rectangle((24, 86, 1256, 504), 118, fill="#EDF3F1", outline="#A6B8B1", width=2)
        draw.rounded_rectangle((24, 550, 1256, 704), 18, fill="#FFFFFF", outline="#CAD3D5")
        return image

    @staticmethod
    @lru_cache(maxsize=106)
    def _card_image(identity, width):
        name = (
            "card_back.png"
            if identity is None
            else f"card_{SUITS[identity // 13]}_{RANKS[identity % 13]}.png"
        )
        with Image.open(ASSETS / name) as source:
            card = source.convert("RGBA")
        card = card.crop(card.getbbox())
        return card.resize((width, round(width * 60 / 42)), Image.Resampling.NEAREST)

    def render(self):
        image = self._background.copy()
        draw = ImageDraw.Draw(image)
        game = self.unwrapped
        step = len(self.points) - 1
        score = self.points[-1]

        def text(x, y, value, size=12, fill=INK, anchor="lt"):
            draw.text(
                (x, y),
                str(value),
                font=_font(max(12, size)),
                fill=fill,
                anchor=anchor,
                stroke_width=0.5,
                stroke_fill=fill,
            )

        def box(bounds, radius=10, fill=None, outline=None, width=1):
            draw.rounded_rectangle(
                bounds,
                radius,
                fill=fill,
                outline=outline,
                width=width,
            )

        def card(x, y, identity, width=84):
            height = round(width * 60 / 42)
            box((x + 2, y + 4, x + width + 2, y + height + 4), 3, fill="#CDD8D3")
            image.paste(
                self._card_image(identity, width),
                (round(x), round(y)),
                self._card_image(identity, width),
            )

        def chip(x, y, wager):
            draw.ellipse(
                (x - 13, y - 13, x + 13, y + 13),
                fill="#FFFFFF",
                outline="#728891",
                width=2,
            )
            draw.ellipse(
                (x - 9, y - 9, x + 9, y + 9),
                outline="#B4C1C6",
                width=1,
            )
            text(x, y, wager, 12, INK, "mm")

        text(30, 22, "BLACKJACK", 26)
        text(
            32,
            54,
            f"{self.policy_name}   /   {game.decks} decks   /   {'H17' if game.hit_soft_17 else 'S17'}",
            12,
            MUTED,
        )
        round_number = max(1, game._rounds + int(game._phase == 1))
        text(770, 26, "ROUND", 10, MUTED)
        text(770, 42, f"{round_number:02}", 20)
        text(986, 26, "SEED / SHOE", 10, MUTED)
        shoe = game._shoe_number - int(self._settled and game._shuffled)
        text(986, 44, f"{self.seed} / {shoe}", 16)
        text(873, 26, "ACTION", 10, MUTED)
        text(873, 42, f"{step:03}", 20)
        text(1250, 23, f"{score:+.1f}", 32, GREEN if score >= 0 else RED, "rt")
        text(1250, 59, "NET POINTS", 10, MUTED, "rt")

        text(640, 115, "DEALER", 11, ACCENT, "mt")
        text(148, 180, "BLACKJACK", 18, ACCENT)
        text(148, 205, "PAYS 3 : 2", 12, MUTED)
        text(
            148,
            242,
            "Dealer stands on soft 17" if not game.hit_soft_17 else "Dealer hits soft 17",
            11,
            MUTED,
        )
        for offset in (8, 4, 0):
            card(1080 + offset, 151 - offset, None, 63)
        text(1116, 252, "SHOE", 10, ACCENT, "mt")

        if self._dealt:
            hands, dealer = game._hands, game._dealer
            reveal = self._settled and any(_value(hand)[0] <= 21 for hand in hands)
            shown = [c.identity for c in dealer] if reveal else [dealer[0].identity, None]
            spacing = min(92, 480 / max(1, len(shown)))
            start = 640 - (84 + spacing * (len(shown) - 1)) / 2
            for i, identity in enumerate(shown):
                card(start + spacing * i, 137, identity)
            total = (
                str(_value(dealer)[0])
                if reveal
                else f"{int(dealer[0]) if dealer[0] != 1 else 'A'} + ?"
            )
            text(640, 267, total, 16, INK, "mt")
            lane = min(350, 1080 / len(hands))
            start = 640 - lane * len(hands) / 2
            for i, hand in enumerate(hands):
                x = start + lane * i
                active = not self._settled and i == game._active
                box(
                    (x + 6, 302, x + lane - 6, 475),
                    17,
                    "#E2ECF2" if active else "#F8FAF9",
                    ACCENT if active else "#C2CFCA",
                    width=2 if active else 1,
                )
                label = "OTHER PLAYER" if self._sat_out else f"HAND {i + 1:02}"
                text(x + 22, 316, label, 11, ACCENT if active else MUTED)
                value, soft = _value(hand)
                total = str(value) + (" SOFT" if soft else "") if len(hand) > 1 else "WAITING"
                text(x + lane - 23, 314, total, 16, INK, "rt")
                width = 63 if len(hands) == 4 else 84
                spacing = min(width + 8, (lane - 55 - width) / max(1, len(hand) - 1))
                for j, item in enumerate(hand):
                    card(x + 23 + j * spacing, 341, item.identity, width)
                chip(x + lane - 32, 449, game._bets[i])
                if active:
                    text(x + lane / 2, 486, "YOUR TURN", 10, ACCENT, "mt")
        else:
            text(640, 220, "A fresh shoe", 26, INK, "mt")
            text(640, 260, "Waiting for the first wager", 14, MUTED, "mt")

        text(34, 521, "LAST ACTION", 10, MUTED)
        text(163, 517, self._action, 18, ACCENT)
        text(280, 521, self._detail, 12, MUTED)
        if self._settled:
            outcome = "SITTING OUT" if self._sat_out else f"ROUND {self._reward:+g}"
            text(1246, 519, outcome, 14, GREEN if self._reward >= 0 else RED, "rt")
        elif self._dealt:
            text(1246, 521, f"Hand {game._active + 1} to act", 12, MUTED, "rt")

        text(44, 567, "NET POINTS", 13)
        text(158, 569, "Cumulative reward  /  every agent action", 11, MUTED)
        text(1234, 565, f"{score:+.1f}", 22, GREEN if score >= 0 else RED, "rt")
        left, right, top, bottom = 77, 1218, 605, 674
        low, high = min(-1, min(self.points)), max(1, max(self.points))
        tick = max(1, math.ceil((high - low) / 4))
        low, high = math.floor(low / tick) * tick, math.ceil(high / tick) * tick

        def y(value):
            return bottom - (value - low) / (high - low) * (bottom - top)

        zero = y(0)
        for value in range(low, high + 1, tick):
            py = y(value)
            draw.line(
                (left, py, right, py),
                fill="#9CAEB1" if value == 0 else "#E1E6E8",
                width=1,
            )
            text(left - 14, py, f"{value:+g}" if value else "0", 10, MUTED, "rm")
        for i in range(5):
            index = round(step * i / 4)
            px = left + (right - left) * i / 4
            text(px, bottom + 10, index, 10, MUTED, "mt")
        for i in range(1, len(self.points)):
            x1 = left + (right - left) * (i - 1) / max(1, step)
            x2 = left + (right - left) * i / max(1, step)
            before, after = self.points[i - 1 : i + 1]
            color = GREEN if before >= 0 else RED
            draw.rectangle(
                (
                    x1,
                    min(zero, y(before)),
                    x2,
                    max(zero, y(before)),
                ),
                fill="#E0EEE8" if before >= 0 else "#F5E4E4",
            )
            draw.line(
                (x1, y(before), x2, y(before)),
                fill=color,
                width=2,
            )
            draw.line(
                (x2, y(before), x2, y(after)),
                fill=GREEN if after >= 0 else RED,
                width=2,
            )
        px = right if step else left
        draw.ellipse(
            (px - 4, y(score) - 4, px + 4, y(score) + 4),
            fill=GREEN if score >= 0 else RED,
        )
        return np.asarray(image).copy()
