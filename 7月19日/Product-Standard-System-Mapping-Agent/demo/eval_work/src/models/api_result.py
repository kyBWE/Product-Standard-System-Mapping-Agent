from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LoadDataResult:
    total_categories: int = 0
    total_products: int = 0
    vector_index_status: bool = False
    trgm_index_status: bool = False
    page_index_status: bool = False
    skipped_rows: int = 0
    errors: list[str] = field(default_factory=list)
