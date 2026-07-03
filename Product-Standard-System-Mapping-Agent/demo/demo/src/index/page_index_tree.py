from __future__ import annotations
import logging
import re
from dataclasses import dataclass

from src.models.category_node import CategoryNode
from src.models.treenode import TreeNode


logger = logging.getLogger("PageIndexTree")


@dataclass
class IndexHit:
    """PageIndex 推理索引命中：指向树中某条路径，而非扁平检索结果。"""
    node: TreeNode
    root: TreeNode
    path: list[TreeNode]
    match_type: str
    score: float


class PageIndexTree:
    BEAM_WIDTH = 3

    def __init__(self) -> None:
        self._root_nodes: list[TreeNode] = []
        self._node_map: dict[str, TreeNode] = {}
        self._term_index: dict[str, set[str]] = {}

    def build_tree(self, nodes: list[CategoryNode]) -> None:
        self._root_nodes = []
        self._node_map = {}
        self._term_index = {}

        for node in nodes:
            tree_node = TreeNode(
                category_id=node.category_id,
                category_name=node.category_name,
                syn_list=list(node.syn_list),
                children=[],
                parent=None,
                depth=0,
            )
            self._node_map[node.category_id] = tree_node

        for node in nodes:
            tree_node = self._node_map[node.category_id]
            pids = node.category_pids

            if not pids:
                self._root_nodes.append(tree_node)
                tree_node.depth = 0
            else:
                parent_id = pids[-1]
                parent_node = self._node_map.get(parent_id)
                if parent_node is not None:
                    parent_node.children.append(tree_node)
                    tree_node.parent = parent_node
                    tree_node.depth = parent_node.depth + 1
                else:
                    logger.warning(
                        f"父节点不存在: category_id={node.category_id}, "
                        f"parent_id={parent_id}, 将作为根节点处理"
                    )
                    self._root_nodes.append(tree_node)
                    tree_node.depth = 0

        self._build_reasoning_index()
        logger.info(
            f"PageIndex树构建完成: 根节点数={len(self._root_nodes)}, "
            f"总节点数={len(self._node_map)}, 索引词条数={len(self._term_index)}"
        )

    def _build_reasoning_index(self) -> None:
        """构建树形推理索引：词条 -> 节点 ID（类似目录/索引页，不是向量库）。"""
        for node in self._node_map.values():
            terms = {node.category_name.strip().lower()}
            terms.update(s.strip().lower() for s in node.syn_list if s)
            for kw in self._extract_index_terms(node.category_name):
                terms.add(kw)
            for term in terms:
                if term:
                    self._term_index.setdefault(term, set()).add(node.category_id)

    @staticmethod
    def _extract_index_terms(name: str) -> list[str]:
        cleaned = name
        for sw in ["类产品", "产品", "及其", "其他"]:
            cleaned = cleaned.replace(sw, "")
        parts = re.split(r"[、，,|/]", cleaned)
        terms: list[str] = []
        for part in parts:
            part = part.strip().lower()
            if not part:
                continue
            terms.append(part)
            if len(part) >= 2:
                for i in range(len(part) - 1):
                    terms.append(part[i : i + 2])
        return terms

    def lookup_index(self, query: str, top_k: int = 5) -> list[IndexHit]:
        """索引查表：返回树路径线索，供后续逐层推理使用。"""
        q = query.strip().lower()
        if not q:
            return []

        scored: dict[str, IndexHit] = {}
        for node in self._node_map.values():
            score, match_type = self._score_index_match(q, node)
            if score <= 0:
                continue
            path = self._path_to_root(node)
            if not path:
                continue
            hit = IndexHit(
                node=node,
                root=path[0],
                path=path,
                match_type=match_type,
                score=score,
            )
            prev = scored.get(node.category_id)
            if prev is None or hit.score > prev.score:
                scored[node.category_id] = hit

        hits = sorted(scored.values(), key=lambda h: (h.score, h.node.depth), reverse=True)
        return hits[:top_k]

    @staticmethod
    def _score_index_match(query: str, node: TreeNode) -> tuple[float, str]:
        name = node.category_name.strip().lower()
        if query == name:
            return 1.0, "exact"
        for syn in node.syn_list:
            if syn and query == syn.strip().lower():
                return 0.98, "synonym"
        if query in name or name in query:
            return 0.75, "partial"
        for syn in node.syn_list:
            if syn and (query in syn.lower() or syn.lower() in query):
                return 0.72, "synonym_partial"
        return 0.0, "none"

    def _path_to_root(self, node: TreeNode) -> list[TreeNode]:
        path: list[TreeNode] = []
        current: TreeNode | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def get_root_nodes(self) -> list[TreeNode]:
        return self._root_nodes

    def get_children(self, node_id: str) -> list[TreeNode]:
        node = self._node_map.get(node_id)
        if node is None:
            return []
        return node.children

    def get_node(self, node_id: str) -> TreeNode | None:
        return self._node_map.get(node_id)

    def get_leaf_nodes(self) -> list[TreeNode]:
        leaves: list[TreeNode] = []
        for node in self._node_map.values():
            if not node.children:
                leaves.append(node)
        return leaves

    def iter_all_nodes(self):
        def _dfs(nodes: list[TreeNode]):
            for node in nodes:
                yield node
                yield from _dfs(node.children)

        yield from _dfs(self._root_nodes)
