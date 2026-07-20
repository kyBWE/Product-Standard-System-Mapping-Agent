"""Excel 产品标准体系：树路径解析与字段拼装共用工具。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


HEADERS = [
    "category_id",
    "category_name",
    "category_group_id",
    "category_pids",
    "category_group_name",
    "syn_list",
]


@dataclass
class CategoryRow:
    category_id: str
    category_name: str
    category_group_id: Any
    category_pids: Any
    category_group_name: Any
    syn_list: Any
    source_order: int = 0
    children: list[CategoryRow] = field(default_factory=list, repr=False)

    def as_tuple(self) -> tuple:
        return (
            _maybe_int_id(self.category_id),
            self.category_name,
            self.category_group_id,
            self.category_pids,
            self.category_group_name,
            self.syn_list if self.syn_list is not None else "[]",
        )


def _maybe_int_id(category_id: str):
    """写回 Excel 时尽量保持整数 id（与原表一致）。"""
    try:
        return int(category_id)
    except (TypeError, ValueError):
        return category_id


def parse_pids(value: object) -> list[str]:
    """解析 category_pids，格式如 '[-1],[2],[3],[4]'，过滤虚拟根 -1。"""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    brackets = re.findall(r"\[([^\]]*)\]", text)
    result = [b.strip() for b in brackets if b.strip() and b.strip() != "-1"]
    if not result and "," in text:
        parts = [p.strip().strip("[]") for p in text.split(",")]
        result = [p for p in parts if p and p != "-1"]
    return result


def format_pids(ancestor_ids: list[str]) -> str:
    """写回与原表一致的 pids 字符串。根节点为 '[-1],'。"""
    if not ancestor_ids:
        return "[-1],"
    parts = ["[-1]"] + [f"[{aid}]" for aid in ancestor_ids]
    return ",".join(parts)


def format_group_id(ancestor_ids: list[str]) -> str | None:
    if not ancestor_ids:
        return None
    return ",".join(ancestor_ids)


def format_group_name(ancestor_names: list[str]) -> str | None:
    if not ancestor_names:
        return None
    return ",".join(ancestor_names)


def build_child_fields(parent: CategoryRow, new_name: str, syn_list: str = "[]") -> dict:
    """根据父节点生成新子节点的路径相关字段（不含 category_id）。"""
    parent_pids = parse_pids(parent.category_pids)
    # 新节点的祖先 = 父的祖先 + 父自己
    ancestors = parent_pids + [parent.category_id]

    parent_group_names: list[str] = []
    raw_gn = parent.category_group_name
    if raw_gn is not None and str(raw_gn).strip():
        parent_group_names = [x.strip() for x in str(raw_gn).split(",") if x.strip()]
    ancestor_names = parent_group_names + [parent.category_name]

    return {
        "category_name": new_name,
        "category_group_id": format_group_id(ancestors),
        "category_pids": format_pids(ancestors),
        "category_group_name": format_group_name(ancestor_names),
        "syn_list": syn_list,
    }


def is_descendant_of(row: CategoryRow, ancestor_id: str) -> bool:
    """row 的 ancestor 链中包含 ancestor_id（不含自身）。"""
    return ancestor_id in parse_pids(row.category_pids)


def find_subtree_end_index(rows: list[CategoryRow], parent_id: str) -> int:
    """
    返回应插入新子节点的下标：插在父节点整棵子树最后一个后代之后。
    若父节点尚无后代，则插在父节点下一行。

    说明：若表已是树序，父后的后代是连续块；若表乱序，取全表中
    该父所有后代的最大下标，仍保证新行至少落在「某个后代」之后。
    乱序时建议先跑 sort_taxonomy_excel.py 再插入。
    """
    parent_idx = next(i for i, r in enumerate(rows) if r.category_id == parent_id)
    last_desc = parent_idx
    for i, r in enumerate(rows):
        if is_descendant_of(r, parent_id):
            last_desc = max(last_desc, i)
    return last_desc + 1

def build_forest(rows: list[CategoryRow]) -> list[CategoryRow]:
    """按 pids[-1] 挂父子，兄弟保持原表出现顺序。"""
    node_map = {r.category_id: r for r in rows}
    for r in rows:
        r.children = []

    roots: list[CategoryRow] = []
    attached: set[str] = set()

    for r in rows:
        pids = parse_pids(r.category_pids)
        if not pids:
            roots.append(r)
            attached.add(r.category_id)
            continue
        parent_id = pids[-1]
        parent = node_map.get(parent_id)
        if parent is None:
            roots.append(r)
            attached.add(r.category_id)
        else:
            parent.children.append(r)
            attached.add(r.category_id)

    # 孤儿兜底（理论上上面已覆盖）
    for r in rows:
        if r.category_id not in attached:
            roots.append(r)

    return roots


def dfs_order(roots: list[CategoryRow]) -> list[CategoryRow]:
    ordered: list[CategoryRow] = []

    def walk(node: CategoryRow) -> None:
        ordered.append(node)
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)
    return ordered


def next_category_id(rows: list[CategoryRow]) -> str:
    max_id = 0
    for r in rows:
        try:
            max_id = max(max_id, int(r.category_id))
        except ValueError:
            continue
    return str(max_id + 1)
