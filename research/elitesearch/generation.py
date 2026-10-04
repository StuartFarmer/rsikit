"""elitesearch generation contracts and candidate mutation rules."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from rsikit.policy import InvalidPolicy, Policy


class _PolicyResponse(BaseModel, extra="forbid"):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    implementation: str = Field(min_length=1)

    def to_policy(self) -> type[Policy]:
        return Policy.from_text(self.implementation, name=self.name, description=self.description)


class InvalidCandidate(InvalidPolicy):
    """A generated candidate failed this optimizer's acceptance rules."""


class Edit(BaseModel, extra="forbid"):
    search: str = Field(min_length=1)
    replacement: str


class Mutation(BaseModel, extra="forbid"):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    edits: list[Edit] = Field(min_length=1)


def check_rewrite(parent: str, child: str) -> str:
    """Reject blank or unchanged source; the whole organism is editable."""
    if not child.strip() or child == parent:
        raise InvalidCandidate("Candidate is blank or unchanged")
    return child


def apply_edits(parent: str, edits: list[Edit]) -> str:
    """Apply exact, sequential edits atomically across the whole organism."""
    child = parent
    for edit in edits:
        search, replacement = edit.search, edit.replacement
        start = child.find(search)
        if not search or start < 0 or child.find(search, start + 1) >= 0:
            raise InvalidCandidate("Search must match exactly one nonempty source segment")
        end = start + len(search)
        child = child[:start] + replacement + child[end:]
    return check_rewrite(parent, child)
