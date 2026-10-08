"""The proposal/feedback boundary shared by policy optimizers."""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from .episode import Episode
from .policy import PolicyDefinition


class Optimizer(Protocol):
    """Proposal/feedback contract implemented by policy search strategies.

    Each round proposes unique policy IDs and consumes one complete mapping of
    those IDs to seed-keyed episodes. Include failed and rejected candidates in
    feedback. Optimizers decide how to score evidence and select the best policy;
    callers own generation clients, environments, executors, and persistence.
    """

    @property
    def done(self) -> bool:
        """Whether search is complete and no further proposal rounds are needed."""
        ...

    @property
    def best(self) -> PolicyDefinition | None:
        """Best accepted definition so far, or ``None`` before any success."""
        ...

    async def propose(self) -> list[PolicyDefinition]:
        """Create the next complete round of policy definitions.

        Returns:
            Policies with unique IDs, or an empty list only when ``done`` is true.
                Execution creates fresh runtime instances for each episode.
        """
        ...

    def update(self, results: Mapping[str, Mapping[int, Episode]]) -> None:
        """Consume exactly one complete round of evaluation feedback.

        Args:
            results: Outstanding policy IDs mapped to seed IDs and episodes,
                including rejected and failed candidates. Successful panels must
                use the same seeds so their scores can be compared.
        """
        ...


def validate_results(results, pending, *, seed_panel=None):
    """Validate complete optimizer feedback without mutating optimizer state.

    Args:
        results (Mapping[str, Mapping[int, Episode]]): Policy IDs mapped to seed-keyed episodes.
        pending (Collection[str]): Nonempty outstanding policy IDs to match exactly.
        seed_panel (set[int] | None): Established seed IDs, or ``None`` to infer
            it from the first nonempty, wholly successful candidate panel.

    Returns:
        seed_panel (set[int] | None): The established seed set, or ``None`` if no successful panel establishes
            one. Empty or partly failed panels do not establish or constrain seeds.

    Raises:
        ValueError: Policy IDs, seed types, episodes, or successful seed panels
            do not satisfy the round contract.
    """
    if not pending or not isinstance(results, Mapping) or set(results) != set(pending):
        raise ValueError("Feedback must contain exactly the outstanding policy IDs")
    panel = seed_panel
    for episodes in results.values():
        if not isinstance(episodes, Mapping):
            raise ValueError("Evaluator must return episodes keyed by seed")
        for seed, episode in episodes.items():
            if type(seed) is not int or not isinstance(episode, Episode):
                raise ValueError("Expected integer seeds and Episode values")
            episode.validate_complete()
        if episodes and all(episode.error is None for episode in episodes.values()):
            if panel is not None and set(episodes) != panel:
                raise ValueError("All candidates must use the same seed panel")
            panel = set(episodes)
    return panel


async def search(
    optimizer: Optimizer,
    evaluate: Callable[
        [Sequence[PolicyDefinition]], Awaitable[Mapping[str, Mapping[int, Episode]]]
    ],
    *,
    on_checkpoint: Callable[[Optimizer], None] | None = None,
) -> PolicyDefinition | None:
    """Drive complete proposal/evaluation/feedback rounds until the optimizer finishes.

    Args:
        optimizer: Search strategy implementing the proposal/feedback contract.
        evaluate: Async callable receiving the proposed definitions and returning
            policy IDs mapped to seed-keyed episodes, including failed episodes.
        on_checkpoint: Optional synchronous callback invoked after each proposal,
            after each update, and when search fails or is cancelled. Persistence
            is the callback's responsibility.

    Returns:
        The optimizer's best definition, possibly ``None`` if none was accepted.

    Raises:
        RuntimeError: A proposal is empty while the optimizer is not done.
        ValueError: A proposal contains duplicate IDs or feedback is invalid.

    Note:
        Exceptions from the optimizer, evaluation, and checkpoints propagate.
        During error handling, a secondary checkpoint failure is logged without
        replacing the original exception. The caller owns all resource cleanup.
    """

    def checkpoint():
        if on_checkpoint is not None:
            on_checkpoint(optimizer)

    try:
        while not optimizer.done:
            policies = await optimizer.propose()
            checkpoint()
            if not policies:
                if not optimizer.done:
                    raise RuntimeError("Empty proposal round without optimizer completion")
                break
            ids = [p.id for p in policies]
            if len(set(ids)) != len(ids):
                raise ValueError("Proposal round contains duplicate policy IDs")
            results = await evaluate(policies)
            validate_results(results, ids)
            optimizer.update(results)
            checkpoint()
        return optimizer.best
    except BaseException:
        try:
            checkpoint()
        except BaseException:
            logging.getLogger(__name__).exception("Checkpoint failed while handling search error")
        raise


class SequentialOptimizer(Optimizer):
    """A feedback-dependent search with one policy per complete round.

    An interrupted generator cannot resume its local state. Keep the failure visible
    until explicitly closed instead of treating its exhausted iterator as completion.
    """

    def __init__(self):
        self._proposal_stream = None
        self._pending_policy = None
        self._closed = False
        self._exhausted = False
        self._proposal_failure = None
        self._seed_panel = None
        self.evidence: list[dict[str, dict[int, Episode]]] = []

    @property
    def done(self) -> bool:
        return self._closed or self._exhausted

    async def propose(self) -> list[PolicyDefinition]:
        if self.done:
            return []
        if self._pending_policy is not None:
            raise RuntimeError("Update the pending policy before proposing again")
        if self._proposal_failure is not None:
            raise self._proposal_failure
        if self._proposal_stream is None:
            self._proposal_stream = self._proposals()
        try:
            policy = await self._proposal_stream.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            return []
        except BaseException as exc:
            self._proposal_failure = exc
            raise
        self._pending_policy = policy
        return [policy]

    def update(self, results: Mapping[str, Mapping[int, Episode]]) -> None:
        from .evaluation import episode_error, episode_scores

        pending = [self._pending_policy.id] if self._pending_policy is not None else []
        panel = validate_results(results, pending, seed_panel=self._seed_panel)
        episodes = results[self._pending_policy.id]
        scores = episode_scores(episodes)
        error = episode_error(episodes) or (None if scores else "No accepted episodes")
        self._accept(list(scores.values()), seed_scores=scores, error=error)
        self.evidence.append({self._pending_policy.id: dict(episodes)})
        self._seed_panel = panel
        self._pending_policy = None

    async def _run_evaluated(self):
        if self.evaluate is None:
            raise ValueError("run requires an evaluator; otherwise use propose/update")
        await search(self, self.evaluate)

    async def aclose(self):
        self._closed = True
        if self._proposal_stream is not None:
            await self._proposal_stream.aclose()
        self._pending_policy = None
