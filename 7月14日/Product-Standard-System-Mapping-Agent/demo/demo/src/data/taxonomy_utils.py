from __future__ import annotations

from src.index.page_index_tree import PageIndexTree
from src.infrastructure.db_manager import DBConnectionManager


def parse_numeric_category_id(category_id: str) -> int | None:
    raw = str(category_id).strip().lstrip("#")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def allocate_next_category_id(db: DBConnectionManager) -> str:
    """取 category_texts 与 category_vectors 中数值 id 的最大值 + 1。"""
    rows = db.execute(
        """
        SELECT category_id FROM category_texts
        UNION
        SELECT category_id FROM category_vectors
        """
    )
    max_id = 0
    for row in rows:
        numeric_id = parse_numeric_category_id(row["category_id"])
        if numeric_id is not None:
            max_id = max(max_id, numeric_id)
    if max_id <= 0:
        raise ValueError("无法分配新 category_id：库中不存在有效数值 id")
    return str(max_id + 1)


def build_category_path_fields(page_tree: PageIndexTree, parent_id: str) -> tuple[list[str], str]:
    """新节点的 category_pids / category_group_name：从根到父节点的完整路径。"""
    path = page_tree.get_path_to_root(parent_id)
    if not path:
        return [parent_id], parent_id
    pids = [node.category_id for node in path]
    group_name = ",".join(node.category_name for node in path)
    return pids, group_name


def format_category_path(page_tree: PageIndexTree, node_id: str) -> str:
    path = page_tree.get_path_to_root(node_id)
    if not path:
        return f"#{node_id}"
    return " > ".join(f"{node.category_name}(#{node.category_id})" for node in path)


def resolve_candidate_id(
    analysis: dict,
    candidates: list[tuple[str, str]],
) -> str:
    picked_id = str(analysis.get("root_category_id", "")).strip().lstrip("#")
    if picked_id:
        for cid, _ in candidates:
            if cid == picked_id or cid.lstrip("#") == picked_id:
                return cid

    picked_name = str(analysis.get("root_category_name", "")).strip()
    if picked_name:
        for cid, cname in candidates:
            if cname == picked_name:
                return cid
    return ""


def locate_expansion_parent(
    llm,
    page_tree: PageIndexTree,
    product_name: str,
    *,
    max_depth: int = 10,
    max_candidates: int = 30,
) -> tuple[str, dict, str]:
    """沿树逐层下钻，直到叶子或达到深度上限，返回最深父节点。"""
    candidates = [(r.category_id, r.category_name) for r in page_tree.get_root_nodes()]
    analysis: dict = {}
    parent_id = ""
    path_str = ""

    for depth in range(max_depth):
        if not candidates:
            break

        level_hint = "一级分类" if depth == 0 else f"第{depth + 1}层子分类"
        analysis = llm.detailed_category_analysis(product_name, candidates, level_hint=level_hint)
        picked_id = resolve_candidate_id(analysis, candidates)
        if not picked_id:
            break

        parent_id = picked_id
        path_str = format_category_path(page_tree, picked_id)
        node = page_tree.get_node(picked_id)
        if node is None or not node.children:
            break

        candidates = [
            (child.category_id, child.category_name)
            for child in node.children[:max_candidates]
        ]

    return parent_id, analysis, path_str
