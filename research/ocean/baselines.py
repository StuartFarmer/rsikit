"""Eight reviewed, deterministic policies; none are claimed to be LLM-generated."""

from rsikit import PolicyDefinition

SOURCE = """import numpy as np
from rsikit import Policy

def slide(boards, direction):
    rows = boards.reshape(-1, 4, 4)
    if direction < 2:
        rows = rows.transpose(0, 2, 1)
    if direction in (1, 3):
        rows = rows[:, :, ::-1]
    rows = rows.reshape(-1, 4).copy()
    order = np.argsort(np.where(rows != 0, np.arange(4), 4), axis=1, kind="stable")
    rows = np.take_along_axis(rows, order, axis=1)
    gains = np.zeros(len(rows))
    for col in range(3):
        merge = (rows[:, col] != 0) & (rows[:, col] == rows[:, col + 1])
        rows[merge, col] += 1
        rows[merge, col + 1] = 0
        gains[merge] += 2.0 ** rows[merge, col]
    order = np.argsort(np.where(rows != 0, np.arange(4), 4), axis=1, kind="stable")
    rows = np.take_along_axis(rows, order, axis=1).reshape(-1, 4, 4)
    if direction in (1, 3):
        rows = rows[:, :, ::-1]
    if direction < 2:
        rows = rows.transpose(0, 2, 1)
    return rows.reshape(-1, 16), gains.reshape(-1, 4).sum(axis=1)

def value(boards, gains):
    grid = boards.reshape(-1, 4, 4)
    empty = (boards == 0).sum(axis=1)
    rough = np.abs(np.diff(grid, axis=1)).sum(axis=(1, 2))
    rough += np.abs(np.diff(grid, axis=2)).sum(axis=(1, 2))
    corner = boards[:, 0]
    weights = WEIGHTS
    return weights[0] * gains + weights[1] * empty + weights[2] * corner - weights[3] * rough

class Solution(Policy):
    async def act(self, observation):
        # Upstream 3.0: empty flags, then 16 one-hot tile features per cell.
        occupied = observation[:, 16:32] == 0
        tiles = observation[:, 32:288].reshape(-1, 16, 16)
        observation = np.where(occupied, tiles.argmax(axis=2) + 1, 0)
        scores = []
        for direction in range(4):
            board, gains = slide(observation, direction)
            legal = np.any(board != observation, axis=1)
            score = value(board, gains) + PRIORITY[direction]
            if LOOKAHEAD:
                futures = []
                for second in range(4):
                    next_board, next_gains = slide(board, second)
                    futures.append(value(next_board, next_gains))
                score += 0.25 * np.max(futures, axis=0)
            scores.append(np.where(legal, score, -np.inf))
        return np.argmax(scores, axis=0).astype(np.int64)
"""


def policies(env_name="g2048"):
    """Return frozen source variants, from legal priority to bounded two-ply search."""
    if env_name != "g2048":
        result = [
            PolicyDefinition.from_text(
                """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.zeros(len(observation), dtype=np.int64)
""",
                name="noop",
                description="Upstream discrete-action fallback",
            )
        ]
        if env_name in ("breakout", "maze"):
            result.append(
                PolicyDefinition.from_text(
                    BREAKOUT_REFERENCE if env_name == "breakout" else MAZE_REFERENCE,
                    name="ball-tracker" if env_name == "breakout" else "depth-first-explorer",
                    description="Frozen deterministic reference; independent state per row",
                )
            )
        return result
    variants = [
        ("legal-priority", (0, 0, 0, 0), False, (4, 1, 3, 2)),
        ("merge-greedy", (1, 0, 0, 0), False, (0, 0, 0, 0)),
        ("empty-cells", (0, 1, 0, 0), False, (0, 0, 0, 0)),
        ("corner", (0.1, 8, 4, 0), False, (0, 0, 0, 0)),
        ("smooth-board", (0.1, 8, 0, 1), False, (0, 0, 0, 0)),
        ("corner-smooth", (0.2, 10, 5, 1), False, (0, 0, 0, 0)),
        ("merge-two-ply", (1, 0, 0, 0), True, (0, 0, 0, 0)),
        ("board-two-ply", (0.2, 10, 5, 1), True, (0, 0, 0, 0)),
    ]
    return [
        PolicyDefinition.from_text(
            SOURCE.replace("WEIGHTS", repr(weights))
            .replace("PRIORITY", repr(priority))
            .replace("LOOKAHEAD", repr(lookahead)),
            name=name,
            description="Reviewed benchmark policy; deterministic, row-independent; no spawn lookahead.",
        )
        for name, weights, lookahead, priority in variants
    ]


BREAKOUT_REFERENCE = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        # Default arena: x is top-left, paddle width is relative to 62 pixels.
        paddle = observation[:, 0] + observation[:, 9] * 31 / 576
        ball = observation[:, 2] + 16 / 576
        delta = ball - paddle
        return np.where(delta < -0.02, 1, np.where(delta > 0.02, 2, 0)).astype(np.int64)
"""

MAZE_REFERENCE = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def reset(self, *, seed=None):
        self.positions = None

    async def act(self, observation):
        if self.positions is None:
            self.positions = [(0, 0) for _ in observation]
            self.visited = [{(0, 0)} for _ in observation]
            self.paths = [[] for _ in observation]
        directions = [(1, 0, 1), (0, 1, 4), (-1, 0, 3), (0, -1, 2)]
        actions = np.zeros(len(observation), dtype=np.int64)
        for i, grid in enumerate(observation.reshape(-1, 11, 11)):
            x, y = self.positions[i]
            for dx, dy, action in directions:
                target = (x + dx, y + dy)
                if grid[5 + dy, 5 + dx] != 1 and target not in self.visited[i]:
                    self.paths[i].append((x, y))
                    self.visited[i].add(target)
                    self.positions[i] = target
                    actions[i] = action
                    break
            else:
                if self.paths[i]:
                    target = self.paths[i].pop()
                    actions[i] = next(a for dx, dy, a in directions
                                      if (x + dx, y + dy) == target)
                    self.positions[i] = target
        return actions
"""
