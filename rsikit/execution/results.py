"""Translate returned Huey outcomes and record them in the caller's context."""

import logging

from huey.exceptions import TaskException

from ..episode import Episode
from ..evaluation import InfrastructureError
from ..policy import PolicyDefinition

logger = logging.getLogger("rsikit.execution")


def decode_result(data, timeout: float) -> Episode:
    """Validate an episode payload or translate Huey's task failure."""
    try:
        if isinstance(data, TaskException):
            if data.metadata.get("error", "").startswith("TaskTimeout("):
                return Episode(error=f"Episode exceeded {timeout:g}s")
            raise InfrastructureError(data.metadata.get("traceback") or str(data))
        return Episode.from_data(data)
    except (InfrastructureError, ValueError, TypeError) as exc:
        raise InfrastructureError(f"Invalid episode result: {exc}") from exc


def attach_result(job, data, timeout: float):
    """Account for a returned task, attach its episode, and report policy failure."""
    policy_id = (
        job.policy.id if isinstance(job.policy, PolicyDefinition) else type(job.policy).__qualname__
    )
    # Count returned tasks even when their payload fails validation.
    logger.debug(
        "Episode %s seed=%s finished",
        policy_id[:12],
        job.seed,
        extra={"event": "episode_timing", "policy_id": policy_id, "seed": job.seed},
    )
    job.result = decode_result(data, timeout)
    if job.result.error:
        logger.error(
            "Policy %s failed (seed=%s): %s",
            policy_id[:12],
            job.seed,
            job.result.error,
            extra={"event": "evaluation_failed", "policy_id": policy_id, "seed": job.seed},
        )
