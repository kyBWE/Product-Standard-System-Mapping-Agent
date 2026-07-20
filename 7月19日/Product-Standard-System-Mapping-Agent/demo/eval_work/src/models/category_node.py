from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CategoryNode:
    category_id: str
    category_name: str
    category_pids: list[str] = field(default_factory=list)
    category_group_name: str = ""
    syn_list: list[str] = field(default_factory=list)
