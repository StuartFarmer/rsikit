"""Trusted Hold'em rules and duplicate blocks; candidate execution is injected."""

import math
import random
from collections import Counter
from hashlib import sha256
from numbers import Integral

from pokerkit import Automation, Deck, NoLimitTexasHoldem

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
CARDS = tuple(Deck.STANDARD)


class CandidateFailure(Exception):
    def __init__(self, seat, message):
        super().__init__(message)
        self.seat = seat


def derived_seed(seed, *parts):
    return int.from_bytes(sha256(repr((seed, *parts)).encode()).digest()[:16], "big")


def schedule(count, seed, encounters=None):
    """Equal appearances, no self matches; cyclic wrap handles partial tables.

    For 60 players this is 10 tables; for 50 it is 25 tables (three appearances
    each). Evaluate whole cycles so everyone has exactly the same hand budget.
    """
    if count < 2:
        raise ValueError("Self-play requires at least two policies")
    size = min(count, 6)
    rng = random.Random(seed)
    encounters = encounters or Counter()
    best = None
    # ponytail: sample 16 schedules; use a block-design solver only if imbalance matters.
    for _ in range(16):
        order = list(range(count))
        rng.shuffle(order)
        groups = [
            tuple(order[(start + j) % count] for j in range(size))
            for start in range(0, math.lcm(count, size), size)
        ]
        pairs = Counter()
        for group in groups:
            for a in group:
                for b in group:
                    if a < b:
                        pairs[a, b] += 1
        cost = sum((encounters[pair] + n) ** 2 - encounters[pair] ** 2 for pair, n in pairs.items())
        if best is None or cost < best[0]:
            best = cost, groups
    return best[1]


def play_hand(count, seed, act, *, stack=200, on_action=None):
    """Return net chips, with blinds 1/2 and stacks reset for each hand.

    Actions: -1 raises all-in, 0 checks/folds, 1 checks/calls, integers >= 2
    raise TO that many chips. -1 also represents a legal one-chip all-in bet.
    """
    state = NoLimitTexasHoldem.create_state(AUTOMATIONS, True, 0, (1, 2), 2, stack, count)
    deck = list(CARDS)
    random.Random(seed).shuffle(deck)
    for seat in range(count):
        state.deal_hole(deck[2 * seat : 2 * seat + 2], player_index=seat)
    cursor = 2 * count
    history = []
    while state.status:
        if state.actor_index is None:
            if state.card_burning_status:
                state.burn_card(deck[cursor])
                cursor += 1
            else:
                amount = state.board_dealing_count
                if not amount:
                    raise RuntimeError("Poker engine stalled outside betting/dealing")
                state.deal_board(deck[cursor : cursor + amount])
                cursor += amount
            continue
        seat = state.actor_index
        can_raise = state.can_complete_bet_or_raise_to()
        observation = {
            "seat": seat,
            "button": count - 1,
            "street": state.street_index,
            "hole_cards": [repr(c) for c in state.hole_cards[seat]],
            "board": [repr(c) for c in state.get_board_cards(0)],
            "stacks": list(state.stacks),
            "bets": list(state.bets),
            "active": list(state.statuses),
            "pot": state.total_pot_amount,
            "to_call": state.checking_or_calling_amount,
            "min_raise_to": state.min_completion_betting_or_raising_to_amount if can_raise else 0,
            "max_raise_to": state.max_completion_betting_or_raising_to_amount if can_raise else 0,
            "big_blind": 2,
            "history": list(history),
        }
        action = act(seat, observation)
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise CandidateFailure(seat, "Action must be an integer (not bool)")
        action = int(action)
        try:
            if action == -1 and can_raise:
                state.complete_bet_or_raise_to(observation["max_raise_to"])
            elif action == 0 and observation["to_call"]:
                state.fold()
            elif action in (0, 1):
                state.check_or_call()
            elif action >= 2:
                state.complete_bet_or_raise_to(action)
            else:
                raise ValueError("Negative action")
        except ValueError as exc:
            raise CandidateFailure(seat, f"Illegal action {action}: {exc}") from exc
        history.append([observation["street"], seat, action])
        if on_action is not None:
            on_action(seat, observation, action)
    result = list(state.payoffs)
    if sum(result) != 0:
        raise RuntimeError("Poker engine did not conserve chips")
    return result


def run_block(
    count, deals, seed, act, *, stack=200, start_rotation=None, reset=None, progress=None
):
    """Play every deal in every cyclic seat assignment; results follow original IDs.

    act receives physical seat indices. start_rotation maps seats to original
    competitor indices and must clear all policy memory between duplicate replays.
    """
    chips = [0] * count
    position_chips = [[0] * count for _ in range(count)]
    actions = [Counter() for _ in range(count)]
    for rotation in range(count):
        occupants = [(seat + rotation) % count for seat in range(count)]
        if start_rotation is not None:
            start_rotation(occupants)

        def observe(seat, observation, action):
            label = (
                "fold"
                if action == 0 and observation["to_call"]
                else (
                    "raise"
                    if action == -1 or action >= 2
                    else "call"
                    if observation["to_call"]
                    else "check"
                )
            )
            actions[occupants[seat]][label] += 1

        for hand in range(deals):
            if reset is not None:
                reset(derived_seed(seed, "policy", rotation, hand))
            payoffs = play_hand(
                count, derived_seed(seed, "deal", hand), act, stack=stack, on_action=observe
            )
            for seat, payoff in enumerate(payoffs):
                player = occupants[seat]
                chips[player] += payoff
                position_chips[player][seat] += payoff
            if progress is not None:
                progress(rotation * deals + hand + 1, count * deals)
    return {
        "chips": chips,
        "hands": count * deals,
        "position_chips": position_chips,
        "actions": [dict(a) for a in actions],
    }
