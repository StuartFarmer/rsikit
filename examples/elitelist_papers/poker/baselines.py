"""Fixed, deliberately simple opponents for plumbing checks and held-out reports."""

from rsikit.policy import _policy_class

SOURCE = """from rsikit import Policy
class Solution(Policy):
    async def act(self, o):
        style = {style}
        if style == 0:
            return 1
        if style == 1:
            return 0
        ranks = ["23456789TJQKA".index(card[0]) + 2 for card in o["hole_cards"]]
        strong = ranks[0] == ranks[1] or min(ranks) >= 10
        if style == 2:
            return 1 if strong or o["to_call"] <= 2 else 0
        if style == 3:
            return -1 if strong and o["max_raise_to"] else 1
        if style == 4:
            if o["min_raise_to"] and self.rng.random() < 0.3:
                return -1 if o["min_raise_to"] == 1 else o["min_raise_to"]
            return 1
        return 0 if o["to_call"] > o["pot"] / 3 and not strong else 1
"""


def policies(count=6):
    return [
        _policy_class(
            f"Reference {i}",
            SOURCE.format(style=i % 6),
            "Fixed simple reference; not a strong poker benchmark",
        )
        for i in range(count)
    ]
