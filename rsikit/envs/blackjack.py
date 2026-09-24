"""Finite-shoe blackjack with public card events and multiple shoes per episode."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _value(cards):
    total = sum(cards)
    soft = int(total <= 11 and 1 in cards)
    return total + 10 * soft, soft


class BlackjackEnv(gym.Env):
    """Single-player blackjack with memory across rounds and no privileged observations.

    Actions: during betting, choose an index into bet_sizes; during play,
    0=stand, 1=hit, 2=double, 3=split. Only masked actions are legal.
    See docs/BLACKJACK.md for the observation layout and precise rules.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decks: int = 6,
        penetration: float = 0.75,
        bet_sizes: tuple[int, ...] = (1, 2, 4, 8, 0),
        hit_soft_17: bool = False,
        max_split_hands: int = 4,
        double_after_split: bool = True,
        shoes_per_episode: int = 24,
    ):
        if type(shoes_per_episode) is not int or shoes_per_episode < 1:
            raise ValueError("shoes_per_episode must be a positive integer")
        if type(decks) is not int or not 1 <= decks <= 8:
            raise ValueError("decks must be an integer from 1 to 8")
        if not isinstance(penetration, (int, float)) or not 0 < penetration <= 1:
            raise ValueError("penetration must be in (0, 1]")
        bet_sizes = tuple(bet_sizes)
        if not bet_sizes or any(type(b) is not int or not 0 <= b <= 10000 for b in bet_sizes):
            raise ValueError("bet_sizes must contain integer wagers from 0 to 10000")
        if type(max_split_hands) is not int or not 1 <= max_split_hands <= 4:
            raise ValueError("max_split_hands must be an integer from 1 to 4")
        if type(hit_soft_17) is not bool or type(double_after_split) is not bool:
            raise ValueError("hit_soft_17 and double_after_split must be booleans")
        self.decks = decks
        self.penetration = float(penetration)
        self.bet_sizes = bet_sizes
        self.hit_soft_17 = hit_soft_17
        self.max_split_hands = max_split_hands
        self.double_after_split = double_after_split
        self.shoes_per_episode = shoes_per_episode
        self._template = np.tile(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]), 4 * decks)
        # Every completed player hand has raw sum <=30; dealer <=26. Even the
        # lowest-valued cards in the entire shoe cannot exceed this many draws.
        # ponytail: a conservative fixed reserve reduces small-shoe penetration;
        # tighten the bound if deeper single-deck play becomes necessary.
        # The fixed reserve avoids mid-round shuffles and composition leaks.
        reserve = int(
            np.searchsorted(
                np.cumsum(np.sort(self._template)), 30 * max_split_hands + 26, side="right"
            )
        )
        self.cut_card = max(1, min(int(52 * decks * penetration), 52 * decks - reserve + 1))
        self.action_space = spaces.Discrete(max(4, len(bet_sizes)))
        high = np.array(
            [1, 31, 1, 10, 22, 10, 3, 4, 2 * max(bet_sizes), 32]
            + [10] * 32
            + [1] * self.action_space.n,
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(np.zeros_like(high), high, dtype=np.int32)
        self._done = True
        self.instructions = (
            f"Play single-player blackjack from a shuffled {decks}-deck shoe. "
            f"Aces are 1 or 11; tens and face cards are 10. Dealer "
            f"{'hits' if hit_soft_17 else 'stands on'} soft 17 and hits below 17. "
            "Dealer peeks for blackjack before player decisions. Natural blackjack pays 3:2; "
            "other wins pay 1:1, ties return the stake, losses cost the stake. "
            f"Split equal values into at most {max_split_hands} hands; split aces receive "
            "one card each and cannot be resplit. Split 21 pays 1:1. "
            "Double on any initial two cards, taking exactly one more card. "
            f"Doubling after splitting is {'allowed' if double_after_split else 'not allowed'}. "
            "No insurance or surrender. Credit is unlimited. Maximize total net profit. "
            "The int32 observation is: [0] phase (0=bet, 1=play), [1] current hand total, "
            "[2] usable ace during play or fresh-shoe flag during betting, "
            "[3] dealer upcard, [4] current hand length, [5] pair value "
            "(0 unless equal two-card values), [6] hand index, [7] number of player hands, "
            "[8] current wager, [9] number of newly exposed cards. "
            "[10:10+obs[9]] contains only cards newly exposed since the previous observation; "
            "the rest through [41] is zero padding. [42:] is the legal-action mask. "
            "Table fields [1:9] are zero during betting except [2]: in multi-shoe episodes, "
            "obs[2]=1 announces a freshly shuffled shoe before the next wager. "
            "Process any newly exposed cards from the completed round first, then clear "
            "your card-counting memory if phase=0 and obs[2]=1. "
            f"During betting, action i places bet_sizes[i] units; bet_sizes={bet_sizes}. "
            "A zero wager sits out one round: observe a stand-in player who hits below 17 "
            "and stands otherwise (no splitting or doubling), followed by normal dealer play. "
            "The entire round advances in one step with zero reward and returns to betting. "
            "The bet is placed before any cards for that round are revealed. During play, "
            "0=stand, 1=hit, 2=double, 3=split. Illegal actions raise InvalidAction. "
            "A completed round returns net profit as reward and the next betting observation. "
            "Keep your own memory between rounds. Dealer hole cards are exposed only at "
            "showdown or blackjack, and stay hidden when all player hands bust. "
            f"Each shoe ends after a completed round reaches the cut card ({self.cut_card} "
            "cards dealt); a safety reserve may reduce requested penetration. "
            f"An episode plays {shoes_per_episode} shoes, automatically shuffling between "
            "them and summing net profit. Reset starts a new episode."
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rounds = self._phase = self._shoe_number = 0
        self._shuffle()
        self._done = False
        self._exposed = []
        return self._observation(), {}

    def _shuffle(self):
        self._shoe = self.np_random.permutation(self._template).tolist()
        self._position = 0
        self._shoe_number += 1
        self._shuffled = self.shoes_per_episode > 1

    def _draw(self, *, visible=True):
        card = self._shoe[self._position]
        self._position += 1
        if visible:
            self._exposed.append(card)
        return card

    def _observation(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.int32)
        obs[0] = self._phase
        if self._phase:
            hand = self._hands[self._active]
            total, soft = _value(hand)
            pair = hand[0] if len(hand) == 2 and hand[0] == hand[1] else 0
            obs[1:9] = (
                total,
                soft,
                self._dealer[0],
                len(hand),
                pair,
                self._active,
                len(self._hands),
                self._bets[self._active],
            )
            double = len(hand) == 2 and (len(self._hands) == 1 or self.double_after_split)
            split = bool(pair) and len(self._hands) < self.max_split_hands
            self._legal = (True, True, double, split) + (False,) * (self.action_space.n - 4)
        else:
            obs[2] = self._shuffled
            self._legal = tuple(
                not self._done and i < len(self.bet_sizes) for i in range(self.action_space.n)
            )
        obs[9] = len(self._exposed)
        obs[10 : 10 + len(self._exposed)] = self._exposed
        obs[42:] = self._legal
        return obs

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a new shoe")
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or not 0 <= action < len(self._legal)
            or not self._legal[action]
        ):
            legal = [i for i, allowed in enumerate(self._legal) if allowed]
            raise gym.error.InvalidAction(
                f"Action is outside the legal-action mask: phase={self._phase}, legal actions={legal}"
            )
        self._exposed = []
        self._shuffled = False
        reward, info = 0.0, {}
        if self._phase == 0:
            self._hands = [[self._draw()]]
            self._dealer = [self._draw()]
            self._hands[0].append(self._draw())
            self._dealer.append(self._draw(visible=False))
            self._bets = [self.bet_sizes[action]]
            self._active = 0
            self._phase = 1
            self._natural = _value(self._hands[0])[0] == 21
            if self._natural or _value(self._dealer)[0] == 21:
                reward, info = self._finish()
            elif self._bets[0] == 0:
                while _value(self._hands[0])[0] < 17:
                    self._hands[0].append(self._draw())
                reward, info = self._finish()
        else:
            hand = self._hands[self._active]
            if action == 3:
                card = hand.pop()
                self._hands.insert(self._active + 1, [card])
                self._bets.insert(self._active + 1, self._bets[self._active])
            else:
                if action in (1, 2):
                    hand.append(self._draw())
                if action == 2:
                    self._bets[self._active] *= 2
                if action != 1 or _value(hand)[0] >= 21:
                    self._active += 1
            # Deal the second card to each split hand only when it becomes active.
            while self._active < len(self._hands):
                hand = self._hands[self._active]
                if len(hand) == 1:
                    hand.append(self._draw())
                    if hand[0] == 1 or _value(hand)[0] == 21:
                        self._active += 1
                        continue
                break
            if self._active == len(self._hands):
                reward, info = self._finish()
        return self._observation(), reward, self._done, False, info

    def _finish(self):
        dealer, soft = _value(self._dealer)
        dealer_natural = dealer == 21 and len(self._dealer) == 2
        totals = [_value(hand)[0] for hand in self._hands]
        if dealer_natural or any(total <= 21 for total in totals):
            self._exposed.append(self._dealer[1])
            if not self._natural:
                while dealer < 17 or (dealer == 17 and soft and self.hit_soft_17):
                    self._dealer.append(self._draw())
                    dealer, soft = _value(self._dealer)
        reward = 0.0
        for total, bet in zip(totals, self._bets):
            if dealer_natural:
                reward += 0 if self._natural else -bet
            elif self._natural:
                reward += 1.5 * bet
            elif total > 21:
                reward -= bet
            elif dealer > 21 or total > dealer:
                reward += bet
            elif total < dealer:
                reward -= bet
        self._rounds += 1
        self._phase = 0
        if self._position >= self.cut_card:
            if self._shoe_number == self.shoes_per_episode:
                self._done = True
            else:
                self._shuffle()
        return reward, {
            "round_complete": True,
            "rounds": self._rounds,
            "player_hands": len(self._hands),
            "sat_out": self._bets[0] == 0,
            "shoe_shuffled": self._shuffled,
        }
