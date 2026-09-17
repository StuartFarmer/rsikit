"""The executable policy contract; training is not part of this lifecycle."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from gymnasium import Space

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Policy(ABC, Generic[Observation, Action]):
    """Choose actions using public task instructions and episode-local state."""

    def __init__(
        self,
        observation_space: Space[Observation],
        action_space: Space[Action],
        *,
        instructions: str,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.instructions = instructions

    async def reset(self, *, seed: int | None = None) -> None:
        """Reset randomness; subclasses also clear their own episode memory."""
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)

    @abstractmethod
    async def act(self, observation: Observation) -> Action:
        """Return an action without access to the environment."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release policy resources."""
