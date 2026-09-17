"""Generate a Policy definition through Slick."""

from pydantic import BaseModel, Field
from slick import prompt

from rsikit.policy import Policy, _policy_class


class _Response(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


@prompt(template="generate_policy.j2", output_type=_Response)
async def generate(task: str, *, generated: _Response) -> type[Policy]:
    """Return a Policy definition; Run chooses where its implementation executes."""
    return _policy_class(generated.name, generated.implementation)
