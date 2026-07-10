from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    category_id: str
    category_name: str
    syn_list: list[str] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = None
    depth: int = 0
