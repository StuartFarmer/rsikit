"""One-submission circle packing in the unit square."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CirclePackingEnv(gym.Env):
    """Submit a circle packing in the unit square in one step.

    Args:
        count: Number of circles to pack.

    The observation is a dictionary with square bounds and circle count. The
    action is a float64 array of shape `(count, 3)`, with `(x, y, radius)` rows.
    Positive radii, containment and non-overlap are checked with tolerance 1e-9.
    A feasible packing earns the sum of radii; infeasible geometry earns zero.
    Every submission terminates the episode. Info contains `feasible`,
    `violations`, `sum_radii`, and `min_margin`.
    """

    metadata = {"render_modes": []}
    instructions = (
        "Pack the requested count of circles into the unit square. Observation bounds are "
        "[xmin, ymin, xmax, ymax]; count is the number of circles. Return a float64 array "
        "with one (x, y, radius) row per circle. Radii must be positive. Circles must lie "
        "inside the square without overlapping, with a geometric tolerance of 1e-9. "
        "Maximize the sum of radii. A feasible packing earns that sum; infeasible geometry "
        "earns zero. There is one submission per episode."
    )

    def __init__(self, count: int = 10):
        self.count = count
        self.observation_space = spaces.Dict(
            {
                "bounds": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float64),
                "count": spaces.Discrete(count + 1),
            }
        )
        self.action_space = spaces.Box(
            np.zeros((count, 3)),
            np.tile([1.0, 1.0, 0.5], (count, 1)),
            dtype=np.float64,
        )

    def _observation(self):
        return {"bounds": np.array([0.0, 0.0, 1.0, 1.0]), "count": self.count}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        violations, margins = [], []
        for i, (x, y, radius) in enumerate(action):
            if radius <= 0:
                violations.append(f"Circle {i}: radius must be positive")
            margin = min(x - radius, y - radius, 1 - x - radius, 1 - y - radius)
            margins.append(float(margin))
            if margin < -1e-9:
                violations.append(f"Circle {i} crosses the square boundary")
            for j in range(i):
                ox, oy, other_radius = action[j]
                gap = math.hypot(x - ox, y - oy) - radius - other_radius
                margins.append(float(gap))
                if gap < -1e-9:
                    violations.append(f"Circles {j} and {i} overlap")
        sum_radii = math.fsum(float(row[2]) for row in action)
        feasible = not violations
        info = {
            "feasible": feasible,
            "violations": violations,
            "sum_radii": sum_radii,
            "min_margin": min(margins, default=0.0),
        }
        return self._observation(), sum_radii if feasible else 0.0, True, False, info
