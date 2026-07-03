from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    category_id: str
    category_name: str
    similarity: float


@dataclass
class TrgmSearchResult:
    category_id: str
    category_name: str
    similarity: float
