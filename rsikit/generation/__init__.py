"""Generate a Policy definition through Slick."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field
from slick import prompt

from rsikit.policy import Policy

WORKER_LIBRARIES = (Path(__file__).parent / "prompts" / "libraries.txt").read_text(encoding="utf-8")


class _Response(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


@prompt(template="generate_policy.j2", output_type=_Response)
async def _generate(
    task: str, *, libraries: str = WORKER_LIBRARIES, generated: _Response
) -> type[Policy]:
    """Return a Policy definition; the caller chooses where its implementation executes."""
    policy = Policy.from_text(
        generated.implementation, name=generated.name, description=generated.description
    )
    logging.getLogger(__name__).info(
        "Generated %s — %s",
        policy.name,
        policy.description,
        extra={"event": "policy_generated", "policy_id": policy.id},
    )
    return policy


async def generate(task: str, *, libraries: str = WORKER_LIBRARIES, **kwargs) -> type[Policy]:
    """Generate a policy and report its attempt through ordinary Run logging."""
    logger = logging.getLogger(__name__)
    attempt = uuid4().hex
    logger.info(
        "Generating policy",
        extra={
            "progress": dict(
                kind="batch_started",
                batch_id="generic",
                label="Batch 1",
                total_candidates=None,
                optimizer="Policy generation",
                columns={},
            )
        },
    )
    record = dict(
        kind="candidate",
        batch_id="generic",
        attempt_id=attempt,
        revision=0,
        status="generating",
        proposal_done=False,
    )
    logger.info("Requesting policy", extra={"progress": record})
    try:
        policy = await _generate(task, libraries=libraries, **kwargs)
    except BaseException as exc:
        logger.error(
            "Generation failed: %s",
            exc,
            exc_info=not isinstance(exc, asyncio.CancelledError),
            extra={
                "progress": {
                    **record,
                    "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                }
            },
        )
        raise
    policy._progress_attempt = attempt
    logger.info(
        "Generated %s — %s",
        policy.name,
        policy.description,
        extra={
            "progress": dict(
                record,
                status="generated",
                proposal_done=True,
                policy_id=policy.id,
                name=policy.name,
                description=policy.description,
            )
        },
    )
    return policy
