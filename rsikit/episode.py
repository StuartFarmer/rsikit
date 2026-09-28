"""Episode trajectories returned by Evaluator."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Episode:
    """Copied trajectory: T actions/rewards/flags, T+1 observations/infos.

    Transition t is observations[t], actions[t], rewards[t], observations[t+1],
    terminations[t], truncations[t], infos[t+1]. Index 0 of infos is reset info.
    """

    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminations: list[bool] = field(default_factory=list)
    truncations: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def final_step(self) -> tuple:
        """The last Gymnasium step tuple, including its per-step reward."""
        return (
            self.observations[-1],
            self.rewards[-1],
            self.terminations[-1],
            self.truncations[-1],
            self.infos[-1],
        )
