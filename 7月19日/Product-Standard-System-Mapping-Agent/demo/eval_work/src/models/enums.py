from __future__ import annotations
from enum import Enum


class MatchStatus(str, Enum):
    MATCHED = "MATCHED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_MATCH = "NO_MATCH"
    PENDING_REVIEW = "PENDING_REVIEW"


class EngineType(str, Enum):
    RAG_VECTOR = "RAG_VECTOR"
    RAG_RERANK = "RAG_RERANK"
    PAGE_INDEX = "PAGE_INDEX"


class EvolveActionType(str, Enum):
    SYNONYM_UPDATE = "SYNONYM_UPDATE"
    TAXONOMY_EXPANSION = "TAXONOMY_EXPANSION"
    NONE = "NONE"


class ExpansionStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
