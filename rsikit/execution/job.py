"""One evaluation request and its eventual episode result."""

from dataclasses import dataclass, field

import gymnasium as gym

from ..episode import Episode
from ..evaluation import Evaluator, _validate_seed
from ..policy import Policy, PolicyDefinition


@dataclass(eq=False)
class Job:
    """One episode on worker copies of the inputs; submit each job only once.

    Inputs are snapshotted when enqueued. PolicyDefinition inputs are loaded inside
    the worker deadline. task_id identifies the submitted Huey task.
    """

    policy: Policy | PolicyDefinition
    environment: gym.Env
    seed: int | None = None
    max_steps: int | None = field(default=None, kw_only=True)
    instructions: str | None = field(default=None, kw_only=True)
    result: Episode | None = field(default=None, init=False)
    task_id: str | None = field(default=None, init=False)
    _submitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.policy, (Policy, PolicyDefinition)):
            raise TypeError("policy must be a Policy instance or PolicyDefinition")
        if not isinstance(self.environment, gym.Env):
            raise TypeError("environment must be a Gymnasium environment")
        _validate_seed(self.seed)
        Evaluator(max_steps=self.max_steps)
        if self.instructions is not None:
            if not isinstance(self.instructions, str):
                raise TypeError("instructions must be text or None")
            if isinstance(self.policy, Policy):
                raise ValueError("Set instructions on the policy instance before creating a Job")

    @property
    def done(self) -> bool:
        """Whether an episode result is available, including policy failures."""
        return self.result is not None
