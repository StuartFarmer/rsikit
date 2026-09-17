"""Apply exact text edits while preserving marked evolution boundaries."""

import ast
import re

from pydantic import BaseModel, Field

MARKER = re.compile(r"^.*?EVOLVE-BLOCK-(START|END)[^\r\n]*(?:\r?\n|$)", re.MULTILINE)


class Edit(BaseModel, extra="forbid"):
    search: str = Field(min_length=1)
    replacement: str


class Mutation(BaseModel, extra="forbid"):
    edits: list[Edit] = Field(min_length=1)


class Program(BaseModel, extra="forbid"):
    source: str = Field(min_length=1)


class InvalidCandidate(ValueError):
    """A generated candidate failed an edit or evaluation requirement."""


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


def check_program(source: str) -> None:
    """Check syntax and the exported class without importing candidate code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise InvalidCandidate(f"Invalid Python: {exc}") from exc
    if not any(isinstance(node, ast.ClassDef) and node.name == "Solution" for node in tree.body):
        raise InvalidCandidate("Program must define a top-level Solution class")
