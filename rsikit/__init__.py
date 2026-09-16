"""Recursive Self Improvement Kit: measured search over executable artifacts."""

from .edits import InvalidCandidate, ProposalRejected, validate_source
from .evaluation import Evaluation, EvaluationError, LocalEvaluator
from .population import AlphaEvolve, DGMArchive, EoH
from .prompt_search import PromptSearch, PromptTrial
from .proposer import (
    Draft,
    PromptProposer,
    Proposer,
    RepairingProposer,
    SlickProposer,
    recent_context,
)
from .reflection import ReflectionMemory
from .repair import RepairAttempt, RepairExhausted, RepairResult, repair_until_valid
from .selection import better, lineage_weights, rank_parents, top_candidates
from .shinka import ShinkaEvolve, ShinkaProposer
from .strategies import Candidate, HillClimb, SequentialStrategy

__all__ = [
    "AlphaEvolve",
    "DGMArchive",
    "Draft",
    "EoH",
    "Candidate",
    "Evaluation",
    "EvaluationError",
    "InvalidCandidate",
    "HillClimb",
    "LocalEvaluator",
    "ProposalRejected",
    "Proposer",
    "PromptProposer",
    "PromptSearch",
    "PromptTrial",
    "ReflectionMemory",
    "RepairAttempt",
    "RepairExhausted",
    "RepairingProposer",
    "RepairResult",
    "SlickProposer",
    "ShinkaEvolve",
    "ShinkaProposer",
    "SequentialStrategy",
    "repair_until_valid",
    "recent_context",
    "better",
    "lineage_weights",
    "rank_parents",
    "top_candidates",
    "validate_source",
]
