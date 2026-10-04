"""alphaevolve generation contracts and candidate mutation rules."""

import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from rsikit.policy import InvalidPolicy, Policy

MARKER = re.compile(r"^.*?EVOLVE-BLOCK-(START|END)[^\r\n]*(?:\r?\n|$)", re.MULTILINE)


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


def evolution_regions(content: str) -> tuple[list[tuple[int, int]], tuple[str, ...]]:
    """Return editable spans and immutable pieces, including marker lines.

    Marker tokens are reserved, language-independent lexical delimiters. A file
    without markers is entirely editable. Nested or unbalanced markers fail.
    """
    regions, skeleton = [], []
    start, previous = None, 0
    for marker in MARKER.finditer(content):
        if marker[1] == "START" and start is None:
            skeleton.append(content[previous : marker.end()])
            start = marker.end()
        elif marker[1] == "END" and start is not None:
            regions.append((start, marker.start()))
            previous, start = marker.start(), None
        else:
            raise InvalidCandidate("Evolution markers must be balanced and non-nested")
    if start is not None:
        raise InvalidCandidate("Unclosed evolution block")
    if not regions:
        return [(0, len(content))], ()
    skeleton.append(content[previous:])
    return regions, tuple(skeleton)


def check_rewrite(parent: str, child: str) -> str:
    """Reject empty, unchanged, or out-of-bounds replacements without normalizing code."""
    if not child.strip() or child == parent:
        raise InvalidCandidate("Candidate is blank or unchanged")
    if evolution_regions(parent)[1] != evolution_regions(child)[1]:
        raise InvalidCandidate("Candidate changed the immutable skeleton or markers")
    return child


def apply_edits(parent: str, edits: list[Edit]) -> str:
    """Apply exact, sequential edits atomically within the evolution boundaries."""
    child = parent
    for edit in edits:
        search, replacement = edit.search, edit.replacement
        start = child.find(search)
        if not search or start < 0 or child.find(search, start + 1) >= 0:
            raise InvalidCandidate("Search must match exactly one nonempty source segment")
        end = start + len(search)
        if not any(left <= start and end <= right for left, right in evolution_regions(child)[0]):
            raise InvalidCandidate("Search crosses an immutable boundary")
        child = child[:start] + replacement + child[end:]
    return check_rewrite(parent, child)
