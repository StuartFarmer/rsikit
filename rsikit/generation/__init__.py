"""Generate a named policy value through Slick; no host execution or files."""

from slick import prompt

from rsikit.policy import Policy, _PolicyResponse


@prompt(template="generate_policy.j2", output_type=_PolicyResponse)
async def generate(task: str, *, generated: _PolicyResponse) -> Policy:
    """Return the generated policy, including its model-chosen name."""
    return generated.policy()
