from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.engine.llm_adapter import LLMAdapter
from src.engine.query_preprocessor import preprocess_query
from src.index.page_index_tree import IndexHit, PageIndexTree
from src.models.enums import EngineType, MatchStatus
from src.models.match_result import CandidateInfo, MatchResult
from src.models.treenode import TreeNode

if TYPE_CHECKING:
    from src.index.vector_index_manager import VectorIndexManager
    from src.index.trgm_index_manager import TrgmIndexManager
    from src.engine.rerank_adapter import RerankAdapter


logger = logging.getLogger("PageIndexEngine")

_FORCED_GUIDE_TYPES = frozenset({"exact", "synonym", "bag_of_words", "segment_bag_of_words", "partial_short_core"})
_WEAK_GUIDE_TYPES = frozenset({"synonym_short", "partial_short", "synonym_contained_short_core", "partial_contained_short_core"})


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
        vec_mgr: VectorIndexManager | None = None,
        rerank: RerankAdapter | None = None,
        trgm_mgr: TrgmIndexManager | None = None,
    ):
        self._tree = tree
        self._llm = llm
        self._force_llm_each_layer = force_llm_each_layer
        self._vec_mgr = vec_mgr
        self._rerank = rerank
        self._trgm_mgr = trgm_mgr
        self._vec_path_ids: set[str] = set()

    def match(self, product_name: str) -> MatchResult:
        original = product_name
        cleaned = preprocess_query(product_name)
        if not cleaned:
            return MatchResult(
                product_name=original,
                match_status=MatchStatus.NO_MATCH,
                engine_type=EngineType.PAGE_INDEX,
                llm_participated=False,
            )

        mode = "逐层LLM" if self._force_llm_each_layer else "索引引导"
        logger.info(f"PageIndex匹配开始: product_name={cleaned}(原始={original}), mode={mode}")

        roots = self._tree.get_root_nodes()
        if not roots:
            return MatchResult(
                product_name=original,
                match_status=MatchStatus.NO_MATCH,
                engine_type=EngineType.PAGE_INDEX,
                llm_participated=False,
            )

        match_fn = self._match_layer_llm_only if self._force_llm_each_layer else self._match_index_guided
        result = match_fn(cleaned, roots)
        result.product_name = original

        if result.match_status == MatchStatus.NO_MATCH and cleaned != original.strip():
            logger.info(f"PageIndex预处理后未匹配,回退原始查询: {original}")
            result_fallback = match_fn(original.strip(), roots)
            result_fallback.product_name = original
            if result_fallback.match_status != MatchStatus.NO_MATCH:
                return result_fallback

        return result

    _INDEX_SCORE_THRESHOLD = 0.35
    _STRONG_GUIDE_THRESHOLD = 0.55

    def _match_index_guided(self, product_name: str, roots: list[TreeNode]) -> MatchResult:
        index_hits = self._tree.lookup_index(product_name)

        vec_hits = self._vector_guide_lookup(product_name)
        trgm_hits = self._trgm_guide_lookup(product_name)
        fused_hits = self._fuse_vec_trgm_hits(vec_hits, trgm_hits)

        self._vec_path_ids = set()
        if fused_hits:
            for fh in fused_hits:
                for node in fh.path:
                    self._vec_path_ids.add(node.category_id)

        idx_is_exact_or_syn = (
            index_hits
            and index_hits[0].score >= self._STRONG_GUIDE_THRESHOLD
            and index_hits[0].match_type in ("exact", "synonym")
        )

        if idx_is_exact_or_syn and not fused_hits:
            guide_path = index_hits[0].path
            state = self._state_from_guide_path(product_name, guide_path)
            mt = index_hits[0].match_type
            confidence = 0.95 if mt == "exact" else 0.9
            logger.info(
                f"PageIndex索引强制路径(无融合): type={mt}, "
                f"target={guide_path[-1].category_name}({guide_path[-1].category_id})"
            )
            return self._build_result(product_name, state, confidence, llm_used=False)

        if idx_is_exact_or_syn and fused_hits:
            idx_id = index_hits[0].node.category_id
            fused_id = fused_hits[0].node.category_id
            if idx_id == fused_id:
                guide_path = index_hits[0].path
                state = self._state_from_guide_path(product_name, guide_path)
                mt = index_hits[0].match_type
                confidence = 0.95 if mt == "exact" else 0.9
                logger.info(
                    f"PageIndex索引+融合一致: type={mt}, "
                    f"target={guide_path[-1].category_name}({guide_path[-1].category_id})"
                )
                return self._build_result(product_name, state, confidence, llm_used=False)
            logger.info(
                f"PageIndex索引融合冲突: 索引={index_hits[0].node.category_name}({index_hits[0].score:.3f}), "
                f"融合={fused_hits[0].node.category_name}({fused_hits[0].score:.3f}), 采用融合"
            )

        if fused_hits:
            best_hit = fused_hits[0]
            trgm_exact_hit = None
            if trgm_hits and trgm_hits[0].score >= 0.95:
                for fh in fused_hits[:5]:
                    if fh.node.category_id == trgm_hits[0].node.category_id:
                        trgm_exact_hit = fh
                        break
            if trgm_exact_hit is not None:
                best_hit = trgm_exact_hit
            else:
                for fh in fused_hits[:5]:
                    if index_hits and fh.node.category_id == index_hits[0].node.category_id:
                        best_hit = fh
                        break

            need_llm_review = self._should_llm_review(
                product_name, fused_hits, best_hit, trgm_exact_hit
            )
            if need_llm_review and self._llm is not None:
                llm_best = self._llm_final_select(product_name, fused_hits[:5])
                if llm_best is not None:
                    best_hit = llm_best
                    logger.info(
                        f"PageIndex LLM终审改判: target={best_hit.node.category_name}({best_hit.node.category_id})"
                    )

            guide_path = best_hit.path
            state = self._state_from_guide_path(product_name, guide_path)
            sim = best_hit.score
            if sim >= 0.75:
                confidence = min(sim * 0.95, 0.90)
            elif sim >= 0.5:
                confidence = sim * 0.85
            else:
                confidence = sim * 0.7
            idx_agree = (
                index_hits
                and index_hits[0].node.category_id == best_hit.node.category_id
            )
            if idx_agree:
                confidence = min(confidence + 0.05, 0.95)
            llm_used = need_llm_review and self._llm is not None
            logger.info(
                f"PageIndex融合直选: target={guide_path[-1].category_name}({guide_path[-1].category_id}), "
                f"score={sim:.3f}, conf={confidence:.3f}, idx_agree={idx_agree}, "
                f"type={best_hit.match_type}, llm={llm_used}"
            )
            return self._build_result(product_name, state, confidence, llm_used=llm_used)

        if not index_hits or index_hits[0].score < self._INDEX_SCORE_THRESHOLD:
            top_score = index_hits[0].score if index_hits else 0
            logger.info(
                f"PageIndex索引无有效命中或Top-1分数过低: "
                f"hits={len(index_hits) if index_hits else 0}, "
                f"top_score={top_score:.3f}"
            )
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=EngineType.PAGE_INDEX,
                llm_participated=False,
            )

        top_score = index_hits[0].score
        is_strong_hit = top_score >= self._STRONG_GUIDE_THRESHOLD

        if is_strong_hit and index_hits[0].match_type in _FORCED_GUIDE_TYPES:
            guide_path = index_hits[0].path
            state = self._state_from_guide_path(product_name, guide_path)
            mt = index_hits[0].match_type
            if mt == "exact":
                confidence = 0.95
            elif mt == "synonym":
                confidence = 0.9
            else:
                confidence = min(index_hits[0].score * 0.9, 0.85)
            logger.info(
                f"PageIndex索引强制路径: type={index_hits[0].match_type}, "
                f"target={guide_path[-1].category_name}({guide_path[-1].category_id})"
            )
            return self._build_result(product_name, state, confidence, llm_used=False)

        if is_strong_hit and index_hits[0].match_type in _WEAK_GUIDE_TYPES:
            logger.info(
                f"PageIndex弱匹配走beam搜索: type={index_hits[0].match_type}, "
                f"置信度上限0.60"
            )
            guide_path = index_hits[0].path if index_hits else None
            force_guide = True
            entry_roots, root_llm_used = self._select_entry_roots(
                product_name, roots, index_hits
            )
            best_state = self._beam_navigate(
                product_name,
                entry_roots,
                guide_path=guide_path,
                force_guide=force_guide,
                initial_llm_steps=int(root_llm_used),
            )
            raw_conf = self._compute_confidence(best_state, index_hits)
            confidence = min(raw_conf, 0.60)
            llm_used = best_state.llm_steps > 0
            return self._build_result(product_name, best_state, confidence, llm_used)

        guide_path = index_hits[0].path if index_hits else None
        force_guide = is_strong_hit
        entry_roots, root_llm_used = self._select_entry_roots(
            product_name, roots, index_hits
        )

        if guide_path:
            guide_mode = "强制引导" if force_guide else "软引导(加分)"
            logger.info(
                f"PageIndex索引{guide_mode}路径: type={index_hits[0].match_type}, "
                f"score={top_score:.3f}, "
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
                            (guide_child is not None and child.category_id == guide_child.category_id)
                            or child.category_id in self._vec_path_ids
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
                        (guide_child is not None and child.category_id == guide_child.category_id)
                        or child.category_id in self._vec_path_ids
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

    def _trgm_guide_lookup(self, product_name: str, top_k: int = 10) -> list[IndexHit] | None:
        if self._trgm_mgr is None:
            return None
        try:
            trgm_results = self._trgm_mgr.search_by_trgm(
                product_name, threshold=0.3, limit=top_k
            )
            if not trgm_results:
                trgm_results = self._trgm_mgr.search_by_substring(
                    product_name, limit=top_k
                )
            if not trgm_results:
                return None
            hits: list[IndexHit] = []
            for tr in trgm_results:
                node = self._tree.get_node(tr.category_id)
                if node is None:
                    continue
                path = self._tree.get_path_to_root(tr.category_id)
                if not path:
                    continue
                hits.append(IndexHit(
                    node=node,
                    root=path[0],
                    path=path,
                    match_type="trigram_similarity",
                    score=tr.similarity,
                ))
            return hits if hits else None
        except Exception as e:
            logger.warning(f"Trigram辅助检索失败: {e}")
            return None

    def _should_llm_review(
        self,
        product_name: str,
        fused_hits: list[IndexHit],
        best_hit: IndexHit,
        trgm_exact_hit: IndexHit | None,
    ) -> bool:
        if self._llm is None:
            return False
        if trgm_exact_hit is not None:
            return False
        if best_hit.score >= 0.80:
            return False
        if len(fused_hits) < 2:
            return False
        gap = fused_hits[0].score - fused_hits[1].score
        if gap >= 0.15:
            return False
        return True

    def _llm_final_select(
        self, product_name: str, candidates: list[IndexHit]
    ) -> IndexHit | None:
        if not candidates or self._llm is None:
            return None
        try:
            names = [c.node.category_name for c in candidates]
            syn_lists = [c.node.syn_list for c in candidates]
            path_hints = []
            for c in candidates:
                path_str = " > ".join(n.category_name for n in c.path)
                path_hints.append(path_str)

            lines = []
            for i, (name, syns, path) in enumerate(zip(names, syn_lists, path_hints)):
                syn_str = "、".join(syns[:5]) if syns else "无"
                lines.append(f"{i + 1}. {name}（同义词: {syn_str}）路径: {path}")
            candidates_text = "\n".join(lines)

            prompt = f"""请将以下产品名称映射到最合适的一项标准分类。

产品名称：{product_name}

候选标准分类：
{candidates_text}

要求：
1. 综合产品名称、分类名称、同义词与路径判断语义是否匹配；
2. 只选择一个最匹配的候选；若均不合适，selected_index 填 0；
3. confidence 为确信程度（0到1）。

请以JSON格式返回：
{{"selected_index": 1, "confidence": 0.85, "reason": "简要说明"}}"""

            response = self._llm._call_llm(
                prompt,
                system_prompt="你是产品标准分类映射专家，擅长根据产品名称选择最准确的标准分类。",
                method="page_index_final_select",
            )
            result = self._llm._parse_json_response(response)
            idx_raw = int(result.get("selected_index", 0))
            if idx_raw <= 0 or idx_raw > len(candidates):
                return None
            return candidates[idx_raw - 1]
        except Exception as e:
            logger.warning(f"PageIndex LLM终审失败: {e}")
            return None

    def _fuse_vec_trgm_hits(
        self,
        vec_hits: list[IndexHit] | None,
        trgm_hits: list[IndexHit] | None,
        vec_weight: float = 0.6,
        trgm_weight: float = 0.4,
    ) -> list[IndexHit] | None:
        if not vec_hits and not trgm_hits:
            return None
        if vec_hits and not trgm_hits:
            return vec_hits
        if trgm_hits and not vec_hits:
            return trgm_hits

        trgm_exact = (
            trgm_hits
            and trgm_hits[0].score >= 0.95
        )

        if trgm_exact:
            vec_weight = 0.3
            trgm_weight = 0.7

        score_map: dict[str, float] = {}
        hit_map: dict[str, IndexHit] = {}

        for h in vec_hits:
            cid = h.node.category_id
            score_map[cid] = score_map.get(cid, 0.0) + h.score * vec_weight
            if cid not in hit_map:
                hit_map[cid] = h

        for h in trgm_hits:
            cid = h.node.category_id
            score_map[cid] = score_map.get(cid, 0.0) + h.score * trgm_weight
            if cid not in hit_map:
                hit_map[cid] = h

        fused = []
        for cid, combined_score in score_map.items():
            hit = hit_map[cid]
            fused.append(IndexHit(
                node=hit.node,
                root=hit.root,
                path=hit.path,
                match_type="vec_trgm_fused",
                score=combined_score,
            ))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused

    def _vector_guide_lookup(self, product_name: str, top_k: int = 5) -> list[IndexHit] | None:
        if self._vec_mgr is None:
            return None
        try:
            query_vec = self._vec_mgr.embed_query(product_name)
            vec_results = self._vec_mgr.search_by_vector(query_vec, top_k=top_k)
            hits: list[IndexHit] = []
            for vr in vec_results:
                node = self._tree.get_node(vr.category_id)
                if node is None:
                    continue
                path = self._tree.get_path_to_root(vr.category_id)
                if not path:
                    continue
                hits.append(IndexHit(
                    node=node,
                    root=path[0],
                    path=path,
                    match_type="vector_similarity",
                    score=vr.similarity,
                ))

            if hits and self._rerank is not None and hits[0].score < 0.75:
                hits = self._rerank_vector_hits(product_name, hits)

            return hits if hits else None
        except Exception as e:
            logger.warning(f"向量辅助检索失败: {e}")
            return None

    def _rerank_vector_hits(self, product_name: str, hits: list[IndexHit]) -> list[IndexHit]:
        if not self._rerank or len(hits) < 2:
            return hits
        try:
            documents = []
            for h in hits:
                syn_str = "、".join(h.node.syn_list) if h.node.syn_list else ""
                doc = h.node.category_name
                if syn_str:
                    doc += f"（{syn_str}）"
                documents.append(doc)
            scores = self._rerank.rerank_scores(product_name, documents)
            if not scores or max(scores) == 0.0:
                return hits
            reranked = []
            for i, hit in enumerate(hits):
                rs = scores[i] if i < len(scores) else 0.0
                combined = hit.score * 0.4 + rs * 0.6
                reranked.append(IndexHit(
                    node=hit.node,
                    root=hit.root,
                    path=hit.path,
                    match_type="vector_reranked" if rs > 0 else hit.match_type,
                    score=combined,
                ))
            reranked.sort(key=lambda h: h.score, reverse=True)
            logger.info(
                f"PageIndex Rerank精排: top1={reranked[0].node.category_name}({reranked[0].node.category_id}), "
                f"score={reranked[0].score:.3f} (vec={hits[0].score:.3f} -> rerank={scores[0]:.3f})"
            )
            return reranked
        except Exception as e:
            logger.warning(f"PageIndex Rerank精排失败: {e}")
            return hits

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
            if self._vec_path_ids and children:
                for child in children:
                    if child.category_id in self._vec_path_ids:
                        return child, False
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
            mt = index_hits[0].match_type
            if mt == "exact":
                return max(base, 0.95)
            if mt == "synonym":
                return max(base, 0.9)
            if mt == "synonym_short":
                return max(base, 0.70)
            if mt in ("partial", "synonym_partial"):
                return max(base, 0.8)
            if mt == "partial_short":
                return max(base, 0.50)
            if mt == "partial_short_core":
                return max(base, 0.70)
            if mt in ("synonym_contained", "segment_match"):
                return max(base, 0.75)
            if mt in ("bigram", "synonym_segment_match", "bag_of_words"):
                return max(base, 0.70)
            if mt in ("partial_contained", "synonym_contained_short"):
                return max(base, 0.55)
            if mt == "vector_similarity":
                return max(base, min(index_hits[0].score * 0.9, 0.85))
            if mt == "vector_reranked":
                return max(base, min(index_hits[0].score * 0.95, 0.90))
            if mt in ("trigram_similarity", "vec_trgm_fused"):
                return max(base, min(index_hits[0].score * 0.9, 0.85))
            if mt in ("synonym_contained_short_core", "partial_contained_short_core"):
                return max(base, 0.60)
            if mt in ("segment_match_partial", "synonym_segment_match_partial",
                       "synonym_bag_of_words", "segment_bag_of_words",
                       "synonym_segment_bag_of_words"):
                return max(base, 0.45)
            return max(base, 0.8)
        if self._force_llm_each_layer:
            llm_ratio = state.llm_steps / max(len(state.path) - 1, 1)
            uncertainty = llm_ratio * 0.15
            return max(base - uncertainty, 0.1)
        if self._vec_path_ids and state.node.category_id in self._vec_path_ids:
            return max(base, base * 0.8 + 0.15)
        return base * 0.6

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
        if confidence >= 0.65:
            return MatchStatus.MATCHED
        if confidence >= 0.40:
            return MatchStatus.LOW_CONFIDENCE
        return MatchStatus.NO_MATCH
