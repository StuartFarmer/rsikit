"""Generate a Policy definition through Slick."""

import logging
from pathlib import Path

from pydantic import BaseModel, Field
from slick import prompt

from rsikit.policy import Policy

WORKER_LIBRARIES = (Path(__file__).parent / "prompts" / "libraries.txt").read_text(encoding="utf-8")


class _Response(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


@prompt(template="generate_policy.j2", output_type=_Response)
async def generate(
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
