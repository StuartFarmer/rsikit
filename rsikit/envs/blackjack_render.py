"""SVG performance frames; wrap a BlackjackEnv only when making visuals."""

import math
from copy import deepcopy

import gymnasium as gym
import numpy as np

from .blackjack import _value
from .render_theme import (
    BLUE,
    FPS,
    GREEN,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    NEGATIVE_FILL,
    POSITIVE_FILL,
    RED,
    RULE,
    SURFACE,
)
from .svg_frame import SvgFrame, rasterize

RANKS = ("A", *[str(n) for n in range(2, 11)], "J", "Q", "K")
# Unit-square suit geometry keeps the canonical card entirely vector.
SUITS = (
    "M .5 .9 L .08 .46 C -.2 .08 .32 -.1 .5 .22 C .68 -.1 1.2 .08 .92 .46 Z",
    "M .5 0 L 1 .5 L .5 1 L 0 .5 Z",
    "M .43 .68 C -.12 1 -.16 .22 .3 .38 C .03 -.13 .97 -.13 .7 .38 "
    "C 1.16 .22 1.12 1 .57 .68 L .7 1 L .3 1 Z",
    "M .5 0 C .36 .22 -.16 .46 .08 .74 C .2 .86 .38 .76 .44 .68 "
    "L .3 1 L .7 1 L .56 .68 C .62 .76 .8 .86 .92 .74 C 1.16 .46 .64 .22 .5 0 Z",
)


class _Card(int):
    """A normal blackjack value carrying a private, exact deck identity for drawing."""

    def __new__(cls, value, identity):
        card = super().__new__(cls, value)
        card.identity = identity
        return card

    def __getnewargs__(self):
        return int(self), self.identity


class BlackjackRenderer(gym.Wrapper):
    """One deterministic RGB frame per action, with a cumulative reward timeline.

    Card identities reproduce the base environment's shuffle with a separate RNG.
    Neither rendering nor the visual history is included in policy observations.
    """

    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, env, *, policy_name="Baseline agent", split="training", policy_id=""):
        super().__init__(env)
        self.policy_name = policy_name
        self.split, self.policy_id = split, policy_id
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

    @staticmethod
    def _draw_card(frame, x, y, identity, width, id):
        height = width * 10 / 7
        with frame.group(id, transform=f"translate({x} {y})"):
            frame.rect((0, 0, width, height), SURFACE, RULE, radius=3)
            if identity is None:
                frame.rect((5, 5, width - 5, height - 5), HIGHLIGHT, BLUE)
                frame.text(width / 2, height / 2, "?", 24, BLUE, "mm", face="serif")
                return
            suit, rank = divmod(identity, 13)
            color = RED if suit < 2 else INK
            frame.text(7, 6, RANKS[rank], 20, color, face="serif")
            frame.text(width - 7, height - 24, RANKS[rank], 16, color, "rt", face="serif")
            size = width * 0.34
            frame.add(
                "path",
                d=SUITS[suit],
                fill=color,
                transform=f"translate({(width - size) / 2} {(height - size) / 2}) scale({size})",
            )

    def render(self):
        return rasterize(self.render_svg(embed_fonts=False))

    def render_svg(self, *, embed_fonts=True):
        """Return a vector frame; embed fonts for standalone viewing by default."""
        frame = SvgFrame("Blackjack policy performance")
        text = frame.text
        game = self.unwrapped
        step, score = len(self.points) - 1, self.points[-1]
        round_number = max(1, game._rounds + int(game._phase == 1))
        shoe = game._shoe_number - int(self._settled and game._shuffled)
        identity = f"   /   policy {self.policy_id[:12]}" if self.policy_id else ""
        frame.header(
            "Blackjack",
            self.policy_name,
            f"{self.split.upper()}   /   seed {self.seed}   /   shoe {shoe}   /   "
            f"round {round_number}   /   action {step}{identity}",
            f"{score:+.1f}",
            "Net points",
            GREEN if score >= 0 else RED,
        )
        text(32, 154, "Table rules", 18, face="serif")
        text(32, 183, f"{game.decks} decks  /  pays 3 : 2", 16, MUTED)
        text(
            32,
            210,
            "Dealer hits soft 17" if game.hit_soft_17 else "Dealer stands on soft 17",
            16,
            MUTED,
        )
        text(1248, 154, "Dealer", 18, BLUE, "rt", face="serif")
        if self._dealt:
            hands, dealer = game._hands, game._dealer
            reveal = self._settled and any(_value(hand)[0] <= 21 for hand in hands)
            shown = [c.identity for c in dealer] if reveal else [dealer[0].identity, None]
            spacing = min(76, 460 / max(1, len(shown)))
            start = 640 - (64 + spacing * (len(shown) - 1)) / 2
            with frame.group("dealer"):
                for i, card in enumerate(shown):
                    self._draw_card(frame, start + spacing * i, 148, card, 64, f"dealer-card-{i}")
                total = (
                    str(_value(dealer)[0])
                    if reveal
                    else f"{int(dealer[0]) if dealer[0] != 1 else 'A'} + ?"
                )
                text(640, 252, total, 20, INK, "mt", face="serif", id="dealer-total")
            lane = min(440, 1216 / len(hands))
            start = 640 - lane * len(hands) / 2
            for i, hand in enumerate(hands):
                x = start + lane * i
                active = not self._settled and i == game._active
                with frame.group(f"hand-{i + 1}"):
                    frame.rect(
                        (x + 6, 296, x + lane - 6, 472), HIGHLIGHT if active else SURFACE, RULE
                    )
                    if active:
                        frame.rect((x + 6, 296, x + lane - 6, 299), BLUE)
                    label = "Other player" if self._sat_out else f"Hand {i + 1}"
                    text(x + 22, 313, label, 18, BLUE if active else MUTED, face="bold")
                    value, soft = _value(hand)
                    total = str(value) + (" soft" if soft else "") if len(hand) > 1 else "Waiting"
                    text(x + lane - 22, 313, total, 20, INK, "rt", face="serif")
                    width = 58 if len(hands) == 4 else 70
                    spacing = min(width + 10, (lane - 48 - width) / max(1, len(hand) - 1))
                    for j, card in enumerate(hand):
                        self._draw_card(
                            frame,
                            x + 22 + j * spacing,
                            344,
                            card.identity,
                            width,
                            f"hand-{i + 1}-card-{j}",
                        )
                    text(x + lane - 22, 448, f"Wager {game._bets[i]}", 16, MUTED, "rt")
                    if active:
                        text(x + 22, 448, "To act", 16, BLUE)
        else:
            text(640, 236, "A fresh shoe", 30, INK, "mt", face="serif")
            text(640, 285, "Waiting for the first wager", 18, MUTED, "mt")

        with frame.group("last-action"):
            text(32, 495, "Last action", 16, MUTED)
            text(146, 491, self._action, 22, BLUE, face="bold")
            text(288, 495, self._detail, 16, MUTED, max_width=675)
            if self._settled:
                outcome = "Sitting out" if self._sat_out else f"Round {self._reward:+g}"
                text(1248, 493, outcome, 20, GREEN if self._reward >= 0 else RED, "rt")
            elif self._dealt:
                text(1248, 495, f"Hand {game._active + 1} to act", 16, MUTED, "rt")
        frame.line([(32, 528), (1248, 528)], RULE)
        text(32, 543, "Cumulative reward", 20, face="serif")
        text(1248, 547, "Net points / agent actions", 16, MUTED, "rt")
        left, right, top, bottom = 86, 1224, 586, 660
        low, high = min(-1, min(self.points)), max(1, max(self.points))
        tick = max(1, math.ceil((high - low) / 4))
        low, high = math.floor(low / tick) * tick, math.ceil(high / tick) * tick

        def y(value):
            return bottom - (value - low) / (high - low) * (bottom - top)

        with frame.group("reward"):
            zero = y(0)
            for value in range(low, high + 1, tick):
                py = y(value)
                frame.line([(left, py), (right, py)], MUTED if value == 0 else GRID)
                text(left - 14, py, f"{value:+g}" if value else "0", 15, MUTED, "rm")
            for index in dict.fromkeys(round(step * i / 4) for i in range(5)):
                px = left + (right - left) * index / max(1, step)
                text(px, bottom + 12, index, 15, MUTED, "mt")
            for i in range(1, len(self.points)):
                x1 = left + (right - left) * (i - 1) / max(1, step)
                x2 = left + (right - left) * i / max(1, step)
                before, after = self.points[i - 1 : i + 1]
                frame.rect(
                    (x1, min(zero, y(before)), x2, max(zero, y(before))),
                    POSITIVE_FILL if before >= 0 else NEGATIVE_FILL,
                )
                frame.line([(x1, y(before)), (x2, y(before))], GREEN if before >= 0 else RED, 2)
                frame.line([(x2, y(before)), (x2, y(after))], GREEN if after >= 0 else RED, 2)
            frame.circle(
                right if step else left,
                y(score),
                4,
                GREEN if score >= 0 else RED,
                id="reward-current",
            )
        return frame.svg(embed_fonts=embed_fonts)
