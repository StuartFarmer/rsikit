"""A deterministic ten-circle baseline scoring 1.0."""

import numpy as np

from rsikit.policy import Policy


class Solution(Policy):
    async def act(self, observation):
        return np.array(
            [(x, y, 0.1) for y in (0.25, 0.75) for x in (0.1, 0.3, 0.5, 0.7, 0.9)],
            dtype=np.float64,
        )
