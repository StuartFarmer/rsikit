"""Validate source replacements without executing generated code."""

import re

# Same lexical edit convention as slick-bits/alphaevolve, kept local for portability.
MARKER = re.compile(r"^.*?EVOLVE-BLOCK-(START|END)[^\r\n]*(?:\r?\n|$)", re.MULTILINE)
DIFF = re.compile(
    r"^<<<<<<< SEARCH\r?\n(.*?)^={5,7}\r?\n(.*?)^>>>>>>> REPLACE(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)
DELIMITER = re.compile(r"^(?:<{6,} SEARCH|>{6,} REPLACE|={5,7})\s*$", re.MULTILINE)


class InvalidCandidate(ValueError):
    """A proposal is empty, unchanged, or outside its permitted edit regions."""


class ProposalRejected(InvalidCandidate):
    """A generator rejected its final source before returning a candidate."""

    def __init__(self, source: str, feedback: str):
        super().__init__(feedback)
        self.source = source


def _skeleton(source: str) -> tuple[str, ...]:
    pieces, previous, opened = [], 0, False
    for marker in MARKER.finditer(source):
        if marker[1] == "START" and not opened:
            pieces.append(source[previous : marker.end()])
            opened = True
        elif marker[1] == "END" and opened:
            previous, opened = marker.start(), False
        else:
            raise InvalidCandidate("Evolution markers must be balanced and non-nested")
    if opened:
        raise InvalidCandidate("Unclosed evolution block")
    return (*pieces, source[previous:]) if pieces else ()


def validate_source(source: str, parent: str | None = None) -> None:
    """Preserve marked regions, ignoring one optional final LF/CRLF in comparisons.

    Model responses often omit the file's final line ending. Compare without it;
    do not strip other whitespace or change the source that will be evaluated.
    """
    source = re.sub(r"\r?\n\Z", "", source)
    if parent is not None:
        parent = re.sub(r"\r?\n\Z", "", parent)
    if not source.strip() or source == parent:
        raise InvalidCandidate("Source is blank or unchanged")
    skeleton = _skeleton(source)
    if parent is not None and skeleton != _skeleton(parent):
        raise InvalidCandidate("Candidate changed immutable source or evolution markers")


def mutable_regions(source: str) -> list[tuple[int, int]]:
    """Return validated editable spans, excluding marker lines."""
    _skeleton(source)
    markers = list(MARKER.finditer(source))
    return (
        [(start.end(), end.start()) for start, end in zip(markers[::2], markers[1::2])]
        if markers
        else [(0, len(source))]
    )


def apply_diff(parent: str, response: str) -> str:
    """Apply exact, unique SEARCH/REPLACE edits contained in mutable regions."""
    matches = list(DIFF.finditer(response))
    if not matches or DELIMITER.search(DIFF.sub("", response)):
        raise InvalidCandidate("Missing or malformed SEARCH/REPLACE blocks")
    source = parent
    for match in matches:
        search, replacement = (re.sub(r"\r?\n$", "", part) for part in match.groups())
        start = source.find(search)
        if not search or start < 0 or source.find(search, start + 1) >= 0:
            raise InvalidCandidate("SEARCH must match exactly one nonempty source segment")
        end = start + len(search)
        if not any(left <= start and end <= right for left, right in mutable_regions(source)):
            raise InvalidCandidate("SEARCH crosses an immutable boundary")
        source = source[:start] + replacement + source[end:]
    validate_source(source, parent)
    return source
