from __future__ import annotations
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.engine.llm_adapter import LLMAdapter
from src.engine.page_index_engine import PageIndexEngine
from src.engine.rag_match_engine import RAGMatchEngine
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.models.enums import EngineType, MatchStatus
from src.models.match_result import MatchResult
from src.orchestration.result_exporter import ResultExporter
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler


logger = logging.getLogger("MatchOrchestrator")

MATCH_CACHE_SIZE = 5000


class LRUMatchCache:
    def __init__(self, maxsize: int = MATCH_CACHE_SIZE):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, MatchResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> MatchResult | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: MatchResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        rate = self._hits / total * 100 if total > 0 else 0
        return f"hits={self._hits}, misses={self._misses}, rate={rate:.1f}%"


class MatchOrchestrator:
    def __init__(
        self,
        rag_engine: RAGMatchEngine,
        page_engine: PageIndexEngine,
        scheduler: SelfEvolveScheduler,
        trgm_mgr: TrgmIndexManager,
        exporter: ResultExporter,
        output_dir: str = "./output",
        max_workers: int = 4,
    ):
        self._rag_engine = rag_engine
        self._page_engine = page_engine
        self._scheduler = scheduler
        self._trgm_mgr = trgm_mgr
        self._exporter = exporter
        self._output_dir = output_dir
        self._max_workers = max_workers
        self._match_cache = LRUMatchCache()

    def run_single(self, product_name: str, engine_type: EngineType = EngineType.RAG_VECTOR) -> MatchResult:
        cached = self._match_cache.get(product_name)
        if cached is not None:
            result = MatchResult(
                product_name=product_name,
                matched_category_id=cached.matched_category_id,
                confidence=cached.confidence,
                match_status=cached.match_status,
                candidates=cached.candidates,
                engine_type=cached.engine_type,
                llm_participated=cached.llm_participated,
            )
            return result

        engine = self._rag_engine if engine_type == EngineType.RAG_VECTOR else self._page_engine
        result = engine.match(product_name)
        self._match_cache.put(product_name, result)
        self._post_process(result)
        return result

    def run_batch(self, product_names: list[str], engine_type: EngineType = EngineType.RAG_VECTOR) -> list[MatchResult]:
        unique_names = list(dict.fromkeys(product_names))
        dedup_count = len(product_names) - len(unique_names)
        if dedup_count > 0:
            logger.info(f"批量匹配去重: {len(product_names)} -> {len(unique_names)} (去除{dedup_count}个重复)")

        total = len(unique_names)
        logger.info(f"批量匹配开始: 共{total}条(去重后), 引擎={engine_type.value}, 并发={self._max_workers}")

        unique_results: dict[str, MatchResult] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_map = {}
            for name in unique_names:
                future = executor.submit(self._match_safe, name, engine_type)
                future_map[future] = name

            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    result = future.result()
                    unique_results[name] = result
                except Exception as e:
                    logger.error(f"匹配异常: product_name={name}, error={e}")
                    unique_results[name] = MatchResult(
                        product_name=name,
                        match_status=MatchStatus.NO_MATCH,
                        engine_type=engine_type,
                        llm_participated=False,
                    )
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"批量匹配进度: {completed}/{total}")

        results = []
        for name in product_names:
            results.append(unique_results.get(name, MatchResult(
                product_name=name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=engine_type,
                llm_participated=False,
            )))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_label = "rag" if engine_type == EngineType.RAG_VECTOR else "page_index"
        file_name = f"match_results_{engine_label}_{timestamp}.csv"
        self._exporter.export_csv(results, file_name)

        logger.info(f"批量匹配完成: 共{len(results)}条, 缓存统计: {self._match_cache.stats}")
        return results

    def _match_safe(self, product_name: str, engine_type: EngineType) -> MatchResult:
        try:
            return self.run_single(product_name, engine_type)
        except Exception as e:
            return MatchResult(
                product_name=product_name,
                match_status=MatchStatus.NO_MATCH,
                engine_type=engine_type,
                llm_participated=False,
            )

    def _post_process(self, result: MatchResult) -> None:
        try:
            trgm_sim = 0.0
            if result.matched_category_id:
                category_name = ""
                try:
                    cat_result = self._trgm_mgr._db.execute_one(
                        "SELECT category_name FROM category_texts WHERE category_id = %s",
                        (result.matched_category_id,),
                    )
                    category_name = cat_result["category_name"] if cat_result else ""
                except Exception:
                    pass
                if category_name:
                    trgm_sim = self._trgm_mgr.get_trgm_similarity(result.product_name, category_name)

            self._scheduler.process_match_result(result, trgm_sim)
        except Exception as e:
            logger.warning(f"自进化后处理异常: {e}")
