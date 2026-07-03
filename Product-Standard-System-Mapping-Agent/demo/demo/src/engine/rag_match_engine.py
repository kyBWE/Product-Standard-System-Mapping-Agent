from __future__ import annotations
import logging

from src.engine.llm_adapter import LLMAdapter
from src.engine.rerank_adapter import RerankAdapter
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.models.enums import EngineType, MatchStatus
from src.models.match_result import CandidateInfo, CandidateNode, MatchResult, ScoredCandidate
from src.models.config_models import MatchConfig


logger = logging.getLogger("RAGMatchEngine")


class RAGMatchEngine:
    def __init__(
        self,
        vec_mgr: VectorIndexManager,
        trgm_mgr: TrgmIndexManager,
        llm: LLMAdapter,
        match_config: MatchConfig,
        enable_llm: bool = False,
        rerank: RerankAdapter | None = None,
        fine_match_mode: str = "llm",
        engine_type: EngineType = EngineType.RAG_VECTOR,
    ):
        self._vec_mgr = vec_mgr
        self._trgm_mgr = trgm_mgr
        self._llm = llm
        self._rerank = rerank
        self._config = match_config
        self._fine_match_mode = fine_match_mode
        self._engine_type = engine_type
        self._enable_llm = enable_llm
        self._query_vec_cache: dict[str, list[float]] = {}

    def _get_query_vector(self, text: str) -> list[float]:
        if text in self._query_vec_cache:
            return self._query_vec_cache[text]
        vec = self._vec_mgr.embed_query(text)
        if len(self._query_vec_cache) < 10000:
            self._query_vec_cache[text] = vec
        return vec

    def match(self, product_name: str) -> MatchResult:
        logger.info(f"RAG匹配开始: product_name={product_name}")

        try:
            candidates = self._coarse_recall(product_name)
        except Exception as e:
            logger.error(f"粗召回异常: {e}")
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=self._engine_type,
                llm_participated=False,
            )

        if not candidates:
            logger.info(f"粗召回结果为空, 标记无匹配: {product_name}")
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=self._engine_type,
                llm_participated=False,
            )

        best = candidates[0]
        skip_llm = not self._enable_llm
        product_in_name = product_name in best.category_name
        name_in_product = best.category_name in product_name

        # 体系内已有完全同名节点：粗召回即可，无需精排
        if not skip_llm and product_name == best.category_name:
            skip_llm = True

        ambiguous_top = False if skip_llm else self._is_ambiguous_top(product_name, candidates)

        if not skip_llm and not ambiguous_top:
            if best.coarse_score >= 0.85:
                skip_llm = True
            elif best.trgm_similarity >= 0.8 and best.coarse_score >= 0.7 and best.vector_similarity >= 0.5:
                skip_llm = True
            elif best.coarse_score >= 0.75 and best.vector_similarity >= 0.7:
                skip_llm = True
            elif best.trgm_similarity >= 0.6 and best.coarse_score >= 0.65 and len(candidates) == 1:
                skip_llm = True
            elif product_in_name and best.vector_similarity >= 0.6:
                skip_llm = True
            elif name_in_product and best.vector_similarity >= 0.6:
                skip_llm = True
            elif best.coarse_score >= 0.5 and best.vector_similarity >= 0.6 and len(candidates) >= 2 and (candidates[0].coarse_score - candidates[1].coarse_score) >= 0.05:
                skip_llm = True

        if skip_llm:
            llm_bonus = self._skip_fine_bonus(product_name, best)
            confidence = self._compute_final_confidence(best.coarse_score, llm_bonus)
            status = self._determine_status(confidence)
            logger.info(
                f"跳过精匹配(粗召回高置信): {product_name} -> {best.category_name} "
                f"mode={self._fine_match_mode} coarse={best.coarse_score:.4f} trgm={best.trgm_similarity:.4f}"
            )
            return MatchResult(
                product_name=product_name,
                matched_category_id=best.category_id,
                confidence=round(confidence, 4),
                match_status=status,
                candidates=[
                    CandidateInfo(
                        category_id=c.category_id,
                        category_name=c.category_name,
                        coarse_score=round(c.coarse_score, 4),
                        final_confidence=round(
                            self._compute_final_confidence(
                                c.coarse_score,
                                llm_bonus if c.category_id == best.category_id else 0.0,
                            ),
                            4,
                        ),
                    )
                    for c in candidates[:5]
                ],
                engine_type=self._engine_type,
                llm_participated=False,
            )

        try:
            scored = self._fine_match(product_name, candidates)
        except Exception as e:
            logger.error(f"精匹配异常: {e}, 以粗召回结果为准")
            best = candidates[0]
            confidence = self._compute_final_confidence(best.coarse_score, 0)
            status = self._determine_status(confidence)
            return MatchResult(
                product_name=product_name,
                matched_category_id=best.category_id,
                confidence=round(confidence, 4),
                match_status=status,
                candidates=[
                    CandidateInfo(
                        category_id=c.category_id,
                        category_name=c.category_name,
                        coarse_score=round(c.coarse_score, 4),
                        final_confidence=round(
                            self._compute_final_confidence(c.coarse_score, 0), 4
                        ),
                    )
                    for c in candidates[:5]
                ],
                engine_type=self._engine_type,
                llm_participated=False,
            )

        if not scored:
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=self._engine_type,
            )

        best = scored[0]
        status = self._determine_status(best.final_confidence)

        candidate_infos = [
            CandidateInfo(
                category_id=sc.category_id,
                category_name=sc.category_name,
                coarse_score=round(
                    next(
                        (c.coarse_score for c in candidates if c.category_id == sc.category_id),
                        0,
                    ),
                    4,
                ),
                llm_score=round(sc.llm_score, 4),
                final_confidence=round(sc.final_confidence, 4),
            )
            for sc in scored[:5]
        ]

        logger.info(
            f"RAG匹配完成: product_name={product_name}, "
            f"matched_id={best.category_id}, confidence={best.final_confidence:.4f}, "
            f"status={status.value}"
        )

        return MatchResult(
            product_name=product_name,
            matched_category_id=best.category_id,
            confidence=round(best.final_confidence, 4),
            match_status=status,
            candidates=candidate_infos,
            engine_type=self._engine_type,
            llm_participated=True,
        )

    def _coarse_recall(self, product_name: str) -> list[CandidateNode]:
        trgm_results = self._trgm_mgr.search_by_trgm(
            product_name, threshold=self._config.trgm_threshold, limit=self._config.coarse_top_k
        )

        vec_results = []
        query_list = self._get_query_vector(product_name)
        if query_list:
            vec_results = self._vec_mgr.search_by_vector(query_list, top_k=self._config.coarse_top_k)

        candidate_map = {}

        for vr in vec_results:
            candidate_map[vr.category_id] = CandidateNode(
                category_id=vr.category_id,
                category_name=vr.category_name,
                vector_similarity=vr.similarity,
                trgm_similarity=0.0,
                coarse_score=0.0,
            )

        for tr in trgm_results:
            if tr.category_id in candidate_map:
                candidate_map[tr.category_id].trgm_similarity = tr.similarity
            else:
                candidate_map[tr.category_id] = CandidateNode(
                    category_id=tr.category_id,
                    category_name=tr.category_name,
                    vector_similarity=0.0,
                    trgm_similarity=tr.similarity,
                    coarse_score=0.0,
                )

        if not candidate_map:
            substr_results = self._trgm_mgr.search_by_substring(
                product_name, limit=self._config.coarse_top_k
            )
            for sr in substr_results:
                candidate_map[sr.category_id] = CandidateNode(
                    category_id=sr.category_id,
                    category_name=sr.category_name,
                    vector_similarity=0.0,
                    trgm_similarity=sr.similarity,
                    coarse_score=0.0,
                )

        for c in candidate_map.values():
            if product_name == c.category_name:
                c.trgm_similarity = max(c.trgm_similarity, 1.0)
            if c.trgm_similarity >= 0.8:
                c.coarse_score = 0.3 * c.vector_similarity + 0.7 * c.trgm_similarity
            else:
                c.coarse_score = (
                    self._config.vector_weight * c.vector_similarity
                    + self._config.trgm_weight * c.trgm_similarity
                )
            if product_name == c.category_name:
                c.coarse_score = max(c.coarse_score, 0.95)

        candidates = sorted(candidate_map.values(), key=lambda c: self._candidate_rank_key(product_name, c), reverse=True)
        return candidates[: self._config.coarse_top_k]

    @staticmethod
    def _candidate_rank_key(product_name: str, candidate: CandidateNode) -> tuple:
        exact = 1 if product_name == candidate.category_name else 0
        return (exact, candidate.coarse_score, candidate.vector_similarity, candidate.trgm_similarity)

    def _is_ambiguous_top(self, product_name: str, candidates: list[CandidateNode]) -> bool:
        if len(candidates) < 2:
            return False
        if candidates[0].category_name == product_name:
            return False
        top_score = candidates[0].coarse_score
        tied = sum(1 for c in candidates[:5] if c.coarse_score >= top_score - 1e-6)
        if tied >= 2:
            return True
        exact = next((c for c in candidates if c.category_name == product_name), None)
        if exact and exact.category_id != candidates[0].category_id:
            return True
        return False

    def _skip_fine_bonus(self, product_name: str, best: CandidateNode) -> float:
        if product_name == best.category_name:
            return 1.0
        if best.trgm_similarity >= 0.8:
            return 1.0
        if product_name in best.category_name or best.category_name in product_name:
            return 0.7
        if best.coarse_score >= 0.5 and best.vector_similarity >= 0.6:
            return 0.4
        return 0.3

    def _fine_match(self, product_name: str, candidates: list[CandidateNode]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        if self._fine_match_mode == "rerank" and self._rerank is not None:
            return self._fine_match_rerank(product_name, candidates)
        return self._fine_match_llm(product_name, candidates)

    def _fine_match_llm(self, product_name: str, candidates: list[CandidateNode]) -> list[ScoredCandidate]:
        syn_map = self._load_syn_list(candidates)
        llm_payload = [
            (c.category_id, c.category_name, syn_map.get(c.category_id, []))
            for c in candidates
        ]
        coarse_hints = [c.coarse_score for c in candidates]

        try:
            llm_scores = self._llm.multi_candidate_semantic_scoring(
                product_name, llm_payload, coarse_hints=coarse_hints
            )
        except Exception as e:
            logger.warning(f"多候选语义打分失败: {e}")
            llm_scores = [0.0] * len(candidates)

        scored: list[ScoredCandidate] = []
        for c, llm_score in zip(candidates, llm_scores):
            scored.append(ScoredCandidate(
                category_id=c.category_id,
                category_name=c.category_name,
                llm_score=llm_score,
                final_confidence=self._compute_final_confidence(c.coarse_score, llm_score),
            ))

        scored.sort(key=lambda x: x.final_confidence, reverse=True)
        logger.info(
            f"LLM多候选精匹配: {product_name} -> "
            f"{scored[0].category_name}(llm={scored[0].llm_score:.3f}, final={scored[0].final_confidence:.3f})"
            if scored else f"LLM多候选精匹配无结果: {product_name}"
        )
        return scored

    def _fine_match_rerank(self, product_name: str, candidates: list[CandidateNode]) -> list[ScoredCandidate]:
        syn_map = self._load_syn_list(candidates)
        documents = [
            self._build_rerank_document(c.category_name, syn_map.get(c.category_id, []))
            for c in candidates
        ]

        try:
            rerank_scores = self._rerank.rerank_scores(product_name, documents)
        except Exception as e:
            logger.warning(f"Rerank精匹配失败: {e}")
            rerank_scores = [0.0] * len(candidates)

        if len(rerank_scores) < len(candidates):
            rerank_scores = rerank_scores + [0.0] * (len(candidates) - len(rerank_scores))

        scored: list[ScoredCandidate] = []
        for c, rerank_score in zip(candidates, rerank_scores):
            scored.append(ScoredCandidate(
                category_id=c.category_id,
                category_name=c.category_name,
                llm_score=rerank_score,
                final_confidence=self._compute_final_confidence(c.coarse_score, rerank_score),
            ))

        scored.sort(key=lambda x: x.final_confidence, reverse=True)
        logger.info(
            f"Rerank精匹配: {product_name} -> "
            f"{scored[0].category_name}(rerank={scored[0].llm_score:.3f}, final={scored[0].final_confidence:.3f})"
            if scored else f"Rerank精匹配无结果: {product_name}"
        )
        return scored

    @staticmethod
    def _build_rerank_document(category_name: str, syn_list: list[str]) -> str:
        syn_text = "、".join(syn_list[:8]) if syn_list else "无"
        return f"标准分类：{category_name}；同义词：{syn_text}"

    def _load_syn_list(self, candidates: list[CandidateNode]) -> dict[str, list[str]]:
        try:
            cat_ids = [c.category_id for c in candidates]
            if not cat_ids:
                return {}
            placeholders = ",".join(["%s"] * len(cat_ids))
            rows = self._vec_mgr._db.execute(
                f"SELECT category_id, syn_list FROM category_vectors WHERE category_id IN ({placeholders})",
                tuple(cat_ids),
            )
            result = {}
            for r in rows:
                syn = r["syn_list"]
                if syn:
                    result[r["category_id"]] = list(syn) if isinstance(syn, list) else []
            return result
        except Exception:
            return {}

    def _fine_match_weight(self) -> float:
        if self._fine_match_mode == "rerank":
            return self._config.rerank_weight
        return self._config.llm_weight

    def _compute_final_confidence(self, coarse_score: float, fine_score: float) -> float:
        return self._config.coarse_weight * coarse_score + self._fine_match_weight() * fine_score

    def _determine_status(self, confidence: float) -> MatchStatus:
        if confidence >= self._config.low_confidence_threshold:
            return MatchStatus.MATCHED
        elif confidence >= self._config.expand_confidence_threshold:
            return MatchStatus.LOW_CONFIDENCE
        else:
            return MatchStatus.NO_MATCH
