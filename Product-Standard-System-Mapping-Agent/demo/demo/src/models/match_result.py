from __future__ import annotations
from dataclasses import dataclass, field

from src.models.enums import EngineType, MatchStatus


@dataclass
class CandidateInfo:
    category_id: str
    category_name: str
    coarse_score: float = 0.0
    llm_score: float = 0.0
    final_confidence: float = 0.0
    path_depth: int = 0
    path_total: int = 0
    is_match_target: bool = False


@dataclass
class CandidateNode:
    category_id: str
    category_name: str
    syn_list: list[str] = field(default_factory=list)
    vector_similarity: float = 0.0
    trgm_similarity: float = 0.0
    coarse_score: float = 0.0


@dataclass
class ScoredCandidate:
    category_id: str
    category_name: str
    llm_score: float = 0.0
    final_confidence: float = 0.0


@dataclass
class MatchResult:
    product_name: str
    matched_category_id: str | None = None
    confidence: float = 0.0
    match_status: MatchStatus = MatchStatus.NO_MATCH
    candidates: list[CandidateInfo] = field(default_factory=list)
    engine_type: EngineType = EngineType.RAG_VECTOR
    llm_participated: bool = True
