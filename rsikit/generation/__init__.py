"""Generate a Policy definition through Slick."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field
from slick import prompt

from rsikit.policy import PolicyDefinition

WORKER_LIBRARIES = (Path(__file__).parent / "prompts" / "libraries.txt").read_text(encoding="utf-8")


class _Response(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


@prompt(template="generate_policy.j2", output_type=_Response)
async def _generate(
    task: str, *, libraries: str = WORKER_LIBRARIES, generated: _Response
) -> PolicyDefinition:
    """Return a Policy definition; the caller chooses where its implementation executes."""
    policy = PolicyDefinition.from_text(
        generated.implementation, name=generated.name, description=generated.description
    )
    logging.getLogger(__name__).info(
        "Generated %s — %s",
        policy.name,
        policy.description,
        extra={"event": "policy_generated", "policy_id": policy.id},
    )
    return policy


async def generate(task: str, *, libraries: str = WORKER_LIBRARIES, **kwargs) -> PolicyDefinition:
    """Ask the configured Slick model to generate a policy source definition.

    Args:
        task: Task description, including the observation/action contract and
            any requirements the generated policy must satisfy.
        libraries: Text describing libraries available in the execution worker;
            defaults to the bundled worker-library description.
        **kwargs (Any): Additional keyword arguments passed to the Slick prompt call.

    Returns:
        A definition containing generated source, name, and description. Source
            is neither executed nor statically validated here; call ``validate`` or
            evaluate it before accepting it.

    Note:
        Configure Slick's model/provider before calling. Provider, parsing, and
        cancellation errors propagate after progress logging records the failure.
        Within a ``Run`` context, normal logging also updates run progress.
    """
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
