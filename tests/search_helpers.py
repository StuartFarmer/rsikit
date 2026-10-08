"""Policy proposals and deterministic measurements for search contract tests."""

import json
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from rsikit import PolicyDefinition
from tests.helpers import episodes


def proposal(value):
    return json.dumps(
        dict(
            name=f"Policy {value}",
            description=f"Candidate {value}",
            implementation="from rsikit import Policy\nclass Solution(Policy):\n"
            "    async def act(self, observation): return 0\n" + f"# fitness {value}\n",
        )
    )


def policy(value):
    data = json.loads(proposal(value))
    return PolicyDefinition.from_text(data.pop("implementation"), **data)


async def measure(policies, *, seeds=(0,)):
    return {
        p.id: episodes({seed: float(p.description.split()[-1]) for seed in seeds}) for p in policies
    }


def templates(module):
    return patch.object(prompts, "TEMPLATE_ROOT", Path(module.__file__).parent / "prompts")
