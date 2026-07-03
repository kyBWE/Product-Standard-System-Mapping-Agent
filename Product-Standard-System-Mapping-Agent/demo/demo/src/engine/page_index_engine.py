from __future__ import annotations
import logging
from dataclasses import dataclass, field

from src.engine.llm_adapter import LLMAdapter
from src.index.page_index_tree import IndexHit, PageIndexTree
from src.models.enums import EngineType, MatchStatus
from src.models.match_result import CandidateInfo, MatchResult
from src.models.treenode import TreeNode


logger = logging.getLogger("PageIndexEngine")

_FORCED_GUIDE_TYPES = frozenset({"exact", "synonym"})


@dataclass
class PathState:
    node: TreeNode
    path: list[TreeNode] = field(default_factory=list)
    rule_hits: int = 0
    index_guided_steps: int = 0
    llm_steps: int = 0

    @property
    def score(self) -> float:
        depth_score = min(len(self.path) / 10.0, 1.0) * 0.25
        hit_score = min(self.rule_hits / max(len(self.path), 1), 1.0) * 0.40
        guide_score = min(self.index_guided_steps / max(len(self.path), 1), 1.0) * 0.20
        llm_penalty = max(0.0, 1.0 - self.llm_steps * 0.12) * 0.15
        return min(depth_score + hit_score + guide_score + llm_penalty, 1.0)


class PageIndexEngine:
    """无向量、基于树形索引 + 逐层推理的 PageIndex 匹配引擎。"""

    def __init__(
        self,
        tree: PageIndexTree,
        llm: LLMAdapter | None = None,
        force_llm_each_layer: bool = False,
    ):
        self._tree = tree
        self._llm = llm
        self._force_llm_each_layer = force_llm_each_layer

    def match(self, product_name: str) -> MatchResult:
        mode = "逐层LLM" if self._force_llm_each_layer else "索引引导"
        logger.info(f"PageIndex匹配开始: product_name={product_name}, mode={mode}")

        roots = self._tree.get_root_nodes()
        if not roots:
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=EngineType.PAGE_INDEX,
                llm_participated=False,
            )

        if self._force_llm_each_layer:
            return self._match_layer_llm_only(product_name, roots)
        return self._match_index_guided(product_name, roots)

    def _match_index_guided(self, product_name: str, roots: list[TreeNode]) -> MatchResult:
        index_hits = self._tree.lookup_index(product_name)
        if index_hits and index_hits[0].match_type in _FORCED_GUIDE_TYPES:
            guide_path = index_hits[0].path
            state = self._state_from_guide_path(product_name, guide_path)
            confidence = 0.95 if index_hits[0].match_type == "exact" else 0.9
            logger.info(
                f"PageIndex索引强制路径: type={index_hits[0].match_type}, "
                f"target={guide_path[-1].category_name}({guide_path[-1].category_id})"
            )
            return self._build_result(product_name, state, confidence, llm_used=False)

        guide_path = index_hits[0].path if index_hits else None
        force_guide = bool(index_hits)
        entry_roots, root_llm_used = self._select_entry_roots(
            product_name, roots, index_hits
        )

        if guide_path:
            logger.info(
                f"PageIndex索引约束路径: type={index_hits[0].match_type}, "
                f"guide={' > '.join(n.category_name for n in guide_path)}"
            )

        best_state = self._beam_navigate(
            product_name,
            entry_roots,
            guide_path=guide_path,
            force_guide=force_guide,
            initial_llm_steps=int(root_llm_used),
        )
        return self._finalize_match(product_name, best_state, index_hits)

    def _match_layer_llm_only(self, product_name: str, roots: list[TreeNode]) -> MatchResult:
        """逐层 LLM：完全不使用索引，每层由 LLM 推理选路。"""
        logger.info("PageIndex逐层LLM: 不使用索引查表")
        entry_roots, root_llm_used = self._select_entry_roots(
            product_name, roots, index_hits=None
        )
        best_state = self._beam_navigate(
            product_name,
            entry_roots,
            guide_path=None,
            force_guide=False,
            initial_llm_steps=int(root_llm_used),
        )
        return self._finalize_match(product_name, best_state, index_hits=[])

    def _finalize_match(
        self,
        product_name: str,
        best_state: PathState,
        index_hits: list[IndexHit],
    ) -> MatchResult:
        llm_used = best_state.llm_steps > 0
        if best_state.node is None:
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=EngineType.PAGE_INDEX,
                llm_participated=llm_used,
            )
        confidence = self._compute_confidence(best_state, index_hits)
        return self._build_result(product_name, best_state, confidence, llm_used)

    def _select_entry_roots(
        self,
        product_name: str,
        roots: list[TreeNode],
        index_hits: list[IndexHit] | None,
    ) -> tuple[list[TreeNode], bool]:
        """确定入口根：索引引导模式用 Top-1 锁定；逐层 LLM 模式 LLM 从全部根中选。"""
        if index_hits:
            top_root = index_hits[0].root
            logger.info(
                f"PageIndex根节点锁定(索引Top-1): {top_root.category_name}"
            )
            return [top_root], False

        if len(roots) == 1:
            return roots, False

        picked, llm_used = self._llm_pick_root(product_name, roots)
        if picked is not None:
            mode = "逐层LLM" if self._force_llm_each_layer else "索引引导"
            logger.info(
                f"PageIndex根节点消歧({mode}): {len(roots)}选1 -> {picked.category_name}"
            )
            return [picked], llm_used
        return [roots[0]], llm_used

    def _llm_pick_root(
        self, product_name: str, roots: list[TreeNode]
    ) -> tuple[TreeNode | None, bool]:
        return self._llm_pick_child(product_name, roots, [])

    def _state_from_guide_path(
        self, product_name: str, guide_path: list[TreeNode]
    ) -> PathState:
        rule_hits = sum(int(self._rule_hit(product_name, node)) for node in guide_path)
        return PathState(
            node=guide_path[-1],
            path=list(guide_path),
            rule_hits=rule_hits,
            index_guided_steps=max(len(guide_path) - 1, 0),
            llm_steps=0,
        )

    def _beam_navigate(
        self,
        product_name: str,
        entry_roots: list[TreeNode],
        guide_path: list[TreeNode] | None,
        force_guide: bool = False,
        initial_llm_steps: int = 0,
    ) -> PathState:
        beam = [
            PathState(
                node=root,
                path=[root],
                rule_hits=int(self._rule_hit(product_name, root)),
                llm_steps=initial_llm_steps,
            )
            for root in entry_roots
        ]

        max_steps = 12
        for _step in range(max_steps):
            next_beam: list[PathState] = []

            for state in beam:
                if not state.node.children:
                    next_beam.append(state)
                    continue

                children = state.node.children
                guide_child = self._guide_child(state.node, guide_path)

                if force_guide and guide_path:
                    if guide_child is not None:
                        next_beam.append(
                            PathState(
                                node=guide_child,
                                path=state.path + [guide_child],
                                rule_hits=state.rule_hits + int(
                                    self._rule_hit(product_name, guide_child)
                                ),
                                index_guided_steps=state.index_guided_steps + 1,
                                llm_steps=state.llm_steps,
                            )
                        )
                    else:
                        next_beam.append(state)
                    continue

                if self._force_llm_each_layer:
                    picked, used_llm = self._llm_pick_child(
                        product_name, children, state.path
                    )
                    if picked is None:
                        next_beam.append(state)
                        continue
                    rule_hit = self._rule_hit(product_name, picked)
                    next_beam.append(
                        PathState(
                            node=picked,
                            path=state.path + [picked],
                            rule_hits=state.rule_hits + int(rule_hit),
                            index_guided_steps=state.index_guided_steps,
                            llm_steps=state.llm_steps + int(used_llm),
                        )
                    )
                    continue

                rule_matched = self._rule_match(product_name, children)
                expansions: list[TreeNode] = []

                if rule_matched:
                    expansions.extend(rule_matched)
                elif guide_child is not None:
                    expansions.append(guide_child)
                else:
                    picked, used_llm = self._llm_pick_child(
                        product_name, children, state.path
                    )
                    if picked is None:
                        next_beam.append(state)
                        continue
                    expansions.append(picked)
                    for child in self._unique_nodes(expansions):
                        rule_hit = self._rule_hit(product_name, child)
                        guided = (
                            guide_child is not None
                            and child.category_id == guide_child.category_id
                        )
                        next_beam.append(
                            PathState(
                                node=child,
                                path=state.path + [child],
                                rule_hits=state.rule_hits + int(rule_hit),
                                index_guided_steps=state.index_guided_steps + int(guided),
                                llm_steps=state.llm_steps + int(used_llm),
                            )
                        )
                    continue

                for child in self._unique_nodes(expansions):
                    rule_hit = self._rule_hit(product_name, child)
                    guided = (
                        guide_child is not None
                        and child.category_id == guide_child.category_id
                    )
                    next_beam.append(
                        PathState(
                            node=child,
                            path=state.path + [child],
                            rule_hits=state.rule_hits + int(rule_hit),
                            index_guided_steps=state.index_guided_steps + int(guided),
                            llm_steps=state.llm_steps,
                        )
                    )

            if not next_beam:
                break

            next_beam.sort(key=lambda s: s.score, reverse=True)
            beam = next_beam[: PageIndexTree.BEAM_WIDTH]

            if all(not s.node.children for s in beam):
                break

        beam.sort(key=lambda s: s.score, reverse=True)
        return beam[0]

    def _guide_child(
        self, current: TreeNode, guide_path: list[TreeNode] | None
    ) -> TreeNode | None:
        if not guide_path:
            return None
        for i, node in enumerate(guide_path):
            if node.category_id == current.category_id and i + 1 < len(guide_path):
                return guide_path[i + 1]
        return None

    def _llm_pick_child(
        self, product_name: str, children: list[TreeNode], ancestry: list[TreeNode]
    ) -> tuple[TreeNode | None, bool]:
        if self._llm is None or not children:
            return (children[0] if children else None), False
        try:
            names = [c.category_name for c in children]
            parent_path = " > ".join(n.category_name for n in ancestry)
            path_hints = [f"{parent_path} > {name}" if parent_path else name for name in names]
            selected = self._llm.layer_disambiguation(product_name, names, path_hints)
            if selected:
                for child in children:
                    if child.category_name == selected:
                        return child, True
        except Exception as e:
            logger.warning(f"PageIndex逐层推理失败: {e}")
        return children[0], True

    def _compute_confidence(
        self, state: PathState, index_hits: list[IndexHit]
    ) -> float:
        base = state.score
        if index_hits and state.node.category_id == index_hits[0].node.category_id:
            if index_hits[0].match_type == "exact":
                return max(base, 0.95)
            if index_hits[0].match_type == "synonym":
                return max(base, 0.9)
            return max(base, 0.8)
        if self._force_llm_each_layer:
            llm_ratio = state.llm_steps / max(len(state.path) - 1, 1)
            uncertainty = llm_ratio * 0.15
            return max(base - uncertainty, 0.1)
        return base

    def _build_result(
        self,
        product_name: str,
        state: PathState,
        confidence: float,
        llm_used: bool,
    ) -> MatchResult:
        status = self._determine_status(confidence)
        total = len(state.path)
        candidates = [
            CandidateInfo(
                category_id=node.category_id,
                category_name=node.category_name,
                final_confidence=round(confidence, 4) if i == total - 1 else 0.0,
                path_depth=i + 1,
                path_total=total,
                is_match_target=(i == total - 1),
            )
            for i, node in enumerate(state.path)
        ]
        logger.info(
            f"PageIndex匹配完成: {product_name} -> {state.node.category_id}, "
            f"path={' > '.join(n.category_name for n in state.path)}, "
            f"conf={confidence:.4f}"
        )
        return MatchResult(
            product_name=product_name,
            matched_category_id=state.node.category_id,
            confidence=round(confidence, 4),
            match_status=status,
            candidates=candidates,
            engine_type=EngineType.PAGE_INDEX,
            llm_participated=llm_used,
        )

    @staticmethod
    def _unique_nodes(nodes: list[TreeNode]) -> list[TreeNode]:
        seen: set[str] = set()
        result: list[TreeNode] = []
        for node in nodes:
            if node.category_id not in seen:
                seen.add(node.category_id)
                result.append(node)
        return result

    def _rule_match(self, product_name: str, candidates: list[TreeNode]) -> list[TreeNode]:
        return [n for n in candidates if self._rule_hit(product_name, n)]

    @staticmethod
    def _rule_hit(product_name: str, node: TreeNode) -> bool:
        pn = product_name.lower()
        cat = node.category_name.lower()
        if cat in pn or pn in cat:
            return True
        for syn in node.syn_list:
            if syn and syn.lower() in pn:
                return True
        return False

    def _determine_status(self, confidence: float) -> MatchStatus:
        if confidence >= 0.5:
            return MatchStatus.MATCHED
        if confidence >= 0.3:
            return MatchStatus.LOW_CONFIDENCE
        return MatchStatus.NO_MATCH
