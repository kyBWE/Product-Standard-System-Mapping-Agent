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
                    fallback_parent = None
                    for pid in reversed(pids[:-1]):
                        candidate = self._node_map.get(pid)
                        if candidate is not None:
                            fallback_parent = candidate
                            break
                    if fallback_parent is not None:
                        fallback_parent.children.append(tree_node)
                        tree_node.parent = fallback_parent
                        tree_node.depth = fallback_parent.depth + 1
                        logger.info(
                            f"父节点回退: category_id={node.category_id}, "
                            f"缺失parent_id={parent_id}, "
                            f"回退至ancestor={fallback_parent.category_id}({fallback_parent.category_name})"
                        )
                    else:
                        logger.warning(
                            f"父节点不存在且无可用祖先: category_id={node.category_id}, "
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

        candidate_ids = self._collect_candidates(q)

        scored: dict[str, IndexHit] = {}
        for cid in candidate_ids:
            node = self._node_map.get(cid)
            if node is None:
                continue
            score, match_type = self._score_index_match(q, node)
            if score <= 0:
                sem_score, sem_type = self._semantic_bigram_score(q, cid)
                if sem_score >= 0.35:
                    score = sem_score
                    match_type = sem_type
                else:
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

    def _collect_candidates(self, q: str) -> set[str]:
        candidate_ids: set[str] = set()

        if q in self._term_index:
            candidate_ids.update(self._term_index[q])

        for term in self._extract_index_terms(q):
            if term in self._term_index:
                candidate_ids.update(self._term_index[term])

        if not candidate_ids:
            for node in self._node_map.values():
                if self._quick_match(q, node):
                    candidate_ids.add(node.category_id)

        return candidate_ids

    def _bigram_hit_count(self, q: str, category_id: str) -> int:
        terms = self._extract_index_terms(q)
        count = 0
        for term in terms:
            if len(term) >= 2 and category_id in self._term_index.get(term, set()):
                count += 1
        return count

    @staticmethod
    def _char_bigrams(text: str) -> set[str]:
        return {text[i:i+2] for i in range(len(text) - 1)} if len(text) >= 2 else set()

    @staticmethod
    def _char_unigrams(text: str) -> set[str]:
        return set(text) if text else set()

    @staticmethod
    def _bag_of_words_match_score(query: str, target: str) -> float:
        q_uni = PageIndexTree._char_unigrams(query)
        t_uni = PageIndexTree._char_unigrams(target)
        uni_score = 0.0
        if q_uni and t_uni:
            uni_inter = q_uni & t_uni
            uni_jaccard = len(uni_inter) / len(q_uni | t_uni)
            uni_coverage = len(uni_inter) / len(q_uni)
            uni_score = uni_jaccard * 0.3 + uni_coverage * 0.7
        q_bi = PageIndexTree._char_bigrams(query)
        t_bi = PageIndexTree._char_bigrams(target)
        bi_score = 0.0
        if q_bi and t_bi:
            bi_inter = q_bi & t_bi
            bi_jaccard = len(bi_inter) / len(q_bi | t_bi)
            bi_coverage = len(bi_inter) / len(q_bi)
            bi_score = bi_jaccard * 0.3 + bi_coverage * 0.7
        return max(uni_score, bi_score)

    def _semantic_bigram_score(self, q: str, category_id: str) -> tuple[float, str]:
        node = self._node_map.get(category_id)
        if node is None:
            return 0.0, "none"
        best_score = 0.0
        best_type = "none"
        q_segs = self._extract_chinese_segments(q)
        long_segs = [s for s in q_segs if len(s) >= 4]
        short_segs = [s for s in q_segs if len(s) == 3]
        q_parts = long_segs if long_segs else short_segs
        if not q_parts:
            q_parts = [q]
        for qp in q_parts:
            if len(qp) < 3:
                continue
            min_bow = 0.50 if len(qp) >= 4 else 0.65
            s = self._bag_of_words_match_score(qp, node.category_name.strip().lower())
            if s >= min_bow and s > best_score:
                best_score = s
                best_type = "bag_of_words"
            for syn in node.syn_list:
                if not syn:
                    continue
                sl = syn.strip().lower()
                if not sl:
                    continue
                s_syn = self._bag_of_words_match_score(qp, sl) * 0.95
                if s_syn >= min_bow and s_syn > best_score:
                    best_score = s_syn
                    best_type = "synonym_bag_of_words"
        if best_score < 0.40:
            return 0.0, "none"
        return best_score, best_type

    @staticmethod
    def _quick_match(query: str, node: TreeNode) -> bool:
        name = node.category_name.strip().lower()
        if query in name or name in query:
            return True
        for syn in node.syn_list:
            if syn and (query in syn.lower() or syn.lower() in query):
                return True
        return False

    @staticmethod
    def _negation_factor(query: str, matched: str) -> float:
        if not matched or len(matched) >= len(query):
            return 1.0
        q = query.lower()
        m = matched.lower()
        idx = q.find(m)
        if idx > 0 and q[idx - 1] in "非无未反":
            return 0.15
        return 1.0

    @staticmethod
    def _score_index_match(query: str, node: TreeNode) -> tuple[float, str]:
        name = node.category_name.strip().lower()
        if query == name:
            return 1.0, "exact"
        for syn in node.syn_list:
            if syn and query == syn.strip().lower():
                if len(syn.strip()) <= 2:
                    return 0.70, "synonym_short"
                return 0.98, "synonym"
        if query in name:
            if len(query) <= 2:
                is_core = False
                for suffix in ("设备", "产品", "装置", "系统", "器具", "仪器", "机械", "机器", "材料", "部件", "零件", "组件"):
                    if name == query + suffix:
                        is_core = True
                        break
                if not is_core and (name.startswith(query) or name.endswith(query)):
                    rest = name[len(query):] if name.startswith(query) else name[:-len(query)]
                    if len(rest) <= 3 and all('\u4e00' <= c <= '\u9fff' for c in rest):
                        is_core = True
                if is_core:
                    return 0.70, "partial_short_core"
                return 0.55, "partial_short"
            return 0.75, "partial"
        if name in query:
            neg = PageIndexTree._negation_factor(query, name)
            if len(name) <= 2:
                ratio = len(name) / len(query) if query else 0
                if ratio >= 0.5:
                    return 0.55 * neg, "partial_contained_short_core"
                return 0.35 * neg, "partial_contained_short"
            return 0.55 * neg, "partial_contained"
        for syn in node.syn_list:
            if not syn:
                continue
            sl = syn.strip().lower()
            if not sl:
                continue
            if query in sl:
                return 0.72, "synonym_partial"
            if sl in query:
                neg = PageIndexTree._negation_factor(query, sl)
                if len(sl) <= 2:
                    ratio = len(sl) / len(query) if query else 0
                    if ratio >= 0.5:
                        return 0.60 * neg, "synonym_contained_short_core"
                    return 0.35 * neg, "synonym_contained_short"
                return 0.70 * neg, "synonym_contained"
        cns = PageIndexTree._extract_chinese_segments(query)
        for seg in cns:
            if len(seg) >= 3 and seg in name:
                q_uni = PageIndexTree._char_unigrams(query)
                n_uni = PageIndexTree._char_unigrams(name)
                overlap = q_uni & n_uni
                coverage = len(overlap) / len(q_uni) if q_uni else 0
                if coverage >= 0.9:
                    return 0.65, "segment_match"
                elif coverage >= 0.6:
                    return 0.45, "segment_match_partial"
                continue
        for syn in node.syn_list:
            if not syn:
                continue
            sl = syn.strip().lower()
            if not sl:
                continue
            for seg in cns:
                if len(seg) >= 3 and seg in sl:
                    q_uni = PageIndexTree._char_unigrams(query)
                    s_uni = PageIndexTree._char_unigrams(sl)
                    overlap = q_uni & s_uni
                    coverage = len(overlap) / len(q_uni) if q_uni else 0
                    if coverage >= 0.9:
                        return 0.60, "synonym_segment_match"
                    elif coverage >= 0.6:
                        return 0.40, "synonym_segment_match_partial"
                    continue
        bow_score = PageIndexTree._bag_of_words_match_score(query, name)
        bow_threshold = 0.70 if len(query) <= 3 else 0.45
        if bow_score >= bow_threshold:
            return bow_score, "bag_of_words"
        for syn in node.syn_list:
            if not syn:
                continue
            sl = syn.strip().lower()
            if not sl:
                continue
            bow_score_syn = PageIndexTree._bag_of_words_match_score(query, sl)
            if bow_score_syn >= bow_threshold:
                return bow_score_syn * 0.95, "synonym_bag_of_words"
        cns = PageIndexTree._extract_chinese_segments(query)
        for seg in cns:
            if len(seg) >= 4:
                seg_bow = PageIndexTree._bag_of_words_match_score(seg, name)
                if seg_bow >= 0.70:
                    return seg_bow * 0.85, "segment_bag_of_words"
                for syn2 in node.syn_list:
                    if not syn2:
                        continue
                    sl2 = syn2.strip().lower()
                    if not sl2:
                        continue
                    seg_bow_syn = PageIndexTree._bag_of_words_match_score(seg, sl2)
                    if seg_bow_syn >= 0.70:
                        return seg_bow_syn * 0.80, "synonym_segment_bag_of_words"
        return 0.0, "none"

    @staticmethod
    def _extract_chinese_segments(text: str) -> list[str]:
        raw_segments: list[str] = []
        current: list[str] = []
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                current.append(ch)
            else:
                if len(current) >= 2:
                    raw_segments.append(''.join(current))
                current = []
        if len(current) >= 2:
            raw_segments.append(''.join(current))
        result: list[str] = []
        for seg in raw_segments:
            if len(seg) <= 4:
                result.append(seg)
            else:
                for i in range(len(seg)):
                    for j in range(i + 3, len(seg) + 1):
                        result.append(seg[i:j])
        return result

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

    def get_path_to_root(self, node_id: str) -> list[TreeNode]:
        node = self._node_map.get(node_id)
        if node is None:
            return []
        return self._path_to_root(node)

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

    def add_node(self, category_id: str, category_name: str, parent_id: str, syn_list: list[str] | None = None) -> TreeNode | None:
        parent = self._node_map.get(parent_id)
        if parent is None:
            logger.warning(f"热更新: 父节点不存在 parent_id={parent_id}")
            return None
        if category_id in self._node_map:
            logger.info(f"热更新: 节点已存在 category_id={category_id}")
            return self._node_map[category_id]
        new_node = TreeNode(
            category_id=category_id,
            category_name=category_name,
            syn_list=syn_list or [],
            children=[],
            parent=parent,
            depth=parent.depth + 1,
        )
        parent.children.append(new_node)
        self._node_map[category_id] = new_node
        self._index_node(new_node)
        logger.info(f"热更新: 已添加节点 category_id={category_id}, name={category_name}, parent={parent_id}")
        return new_node

    def _index_node(self, node: TreeNode) -> None:
        terms = {node.category_name.strip().lower()}
        terms.update(s.strip().lower() for s in node.syn_list if s)
        for kw in self._extract_index_terms(node.category_name):
            terms.add(kw)
        for term in terms:
            if term:
                self._term_index.setdefault(term, set()).add(node.category_id)
