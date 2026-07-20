# -*- coding: utf-8 -*-
"""评测命中判定：严格 ID + 宽松（同枝近邻 / 父子一层）。"""
from __future__ import annotations


def split_path(path_or_group: str) -> list[str]:
    if not path_or_group:
        return []
    if " > " in path_or_group:
        return [x.strip() for x in path_or_group.split(" > ") if x.strip()]
    return [x.strip() for x in path_or_group.split(",") if x.strip()]


def lca_depth(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def path_of(cid: str | None, id2name: dict[str, str], id2group: dict[str, str],
            fallback_path: str = "") -> list[str]:
    if not cid:
        return split_path(fallback_path)
    group = id2group.get(cid, "")
    name = id2name.get(cid, "")
    parts = split_path(group)
    if name and (not parts or parts[-1] != name):
        parts = parts + [name]
    if parts:
        return parts
    return split_path(fallback_path)


def soft_match(
    pred: str | None,
    gt: str,
    id2name: dict[str, str],
    id2group: dict[str, str],
    gt_path_fallback: str = "",
) -> dict:
    """返回 strict / near / soft 判定。

    near: 同父兄弟，或一方为另一方路径前缀（父子，允许差 1~2 层）
    soft: near，或 LCA>=3 且双方都在较深路径上（同子树）
    """
    if pred is None:
        return {
            "strict": False,
            "near": False,
            "soft": False,
            "relation": "no_match",
            "lca_depth": 0,
        }
    pred, gt = str(pred), str(gt)
    if pred == gt:
        return {
            "strict": True,
            "near": True,
            "soft": True,
            "relation": "exact",
            "lca_depth": -1,
        }

    gp = path_of(gt, id2name, id2group, gt_path_fallback)
    pp = path_of(pred, id2name, id2group)
    depth = lca_depth(gp, pp)
    relation = "unrelated"

    # 父子：一方路径是另一方前缀
    if gp and pp and (gp == pp[: len(gp)] or pp == gp[: len(pp)]):
        relation = "ancestor_descendant"
        near = abs(len(gp) - len(pp)) <= 2
        return {
            "strict": False,
            "near": near,
            "soft": True,
            "relation": relation,
            "lca_depth": depth,
        }

    # 同父兄弟：LCA = len-1 for both (same parent)
    if gp and pp and len(gp) >= 2 and len(pp) >= 2 and depth == len(gp) - 1 and depth == len(pp) - 1:
        relation = "sibling"
        return {
            "strict": False,
            "near": True,
            "soft": True,
            "relation": relation,
            "lca_depth": depth,
        }

    # 同子树（至少共享 3 级）
    if depth >= 3:
        relation = "same_subtree"
        return {
            "strict": False,
            "near": False,
            "soft": True,
            "relation": relation,
            "lca_depth": depth,
        }

    if depth >= 2:
        relation = "same_broad"
    elif depth >= 1:
        relation = "same_top"
    else:
        relation = "cross_domain"

    return {
        "strict": False,
        "near": False,
        "soft": False,
        "relation": relation,
        "lca_depth": depth,
    }


def format_path(cid: str | None, id2name: dict[str, str], id2group: dict[str, str],
                fallback: str = "") -> str:
    parts = path_of(cid, id2name, id2group, fallback)
    return " > ".join(parts)
