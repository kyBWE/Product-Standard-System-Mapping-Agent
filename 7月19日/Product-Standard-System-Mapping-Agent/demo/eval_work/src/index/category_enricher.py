from __future__ import annotations
import json
import logging
import os
import time

from src.engine.llm_adapter import LLMAdapter
from src.index.vector_index_manager import VectorIndexManager
from src.index.trgm_index_manager import TrgmIndexManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.category_node import CategoryNode


logger = logging.getLogger("CategoryEnricher")

SYN_CACHE_FILE = "synonym_cache.json"


class CategoryEnricher:
    def __init__(self, db: DBConnectionManager, llm: LLMAdapter,
                 vec_mgr: VectorIndexManager | None = None,
                 trgm_mgr: TrgmIndexManager | None = None):
        self._db = db
        self._llm = llm
        self._vec_mgr = vec_mgr
        self._trgm_mgr = trgm_mgr

    def enrich_categories(
        self,
        nodes: list[CategoryNode],
        batch_size: int = 5,
        max_nodes: int = 0,
        existing_syn_only: bool = False,
    ) -> int:
        target_nodes = nodes
        if existing_syn_only:
            target_nodes = [n for n in nodes if n.syn_list]
        if max_nodes > 0:
            target_nodes = target_nodes[:max_nodes]

        total = len(target_nodes)
        updated = 0
        enriched_nodes: list[CategoryNode] = []
        logger.info(f"开始LLM分类关键词扩展: 共{total}个节点, 批量大小={batch_size}")

        for i in range(0, total, batch_size):
            batch = target_nodes[i : i + batch_size]
            try:
                results = self._batch_generate_synonyms(batch)
                for node, new_syns in zip(batch, results):
                    if not new_syns:
                        continue
                    merged = list(set(node.syn_list + new_syns))
                    node.syn_list = merged
                    self._update_db_synonyms(node.category_id, merged)
                    enriched_nodes.append(node)
                    updated += 1
                logger.info(
                    f"扩展进度: {min(i + batch_size, total)}/{total}, 本批更新{sum(1 for r in results if r)}个"
                )
            except Exception as e:
                logger.warning(f"批量扩展失败(起始索引{i}): {e}")
            time.sleep(0.5)

        if enriched_nodes and self._vec_mgr:
            logger.info(f"增量更新{len(enriched_nodes)}个节点的向量...")
            self._vec_mgr.update_category_vectors(enriched_nodes)

        if enriched_nodes and self._trgm_mgr:
            self._trgm_mgr.insert_category_texts(enriched_nodes)

        self._save_syn_cache(nodes)

        logger.info(f"LLM分类关键词扩展完成: 更新{updated}/{total}个节点")
        return updated

    @staticmethod
    def load_syn_cache() -> dict[str, list[str]]:
        if not os.path.exists(SYN_CACHE_FILE):
            return {}
        try:
            with open(SYN_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def apply_syn_cache(nodes: list[CategoryNode]) -> int:
        cache = CategoryEnricher.load_syn_cache()
        if not cache:
            return 0
        applied = 0
        for node in nodes:
            if node.category_id in cache:
                cached_syns = cache[node.category_id]
                merged = list(set(node.syn_list + cached_syns))
                if len(merged) > len(node.syn_list):
                    node.syn_list = merged
                    applied += 1
        logger.info(f"应用同义词缓存: {applied}个节点")
        return applied

    @staticmethod
    def _save_syn_cache(nodes: list[CategoryNode]) -> None:
        cache = CategoryEnricher.load_syn_cache()
        for node in nodes:
            if node.syn_list:
                existing = cache.get(node.category_id, [])
                merged = list(set(existing + node.syn_list))
                cache[node.category_id] = merged
        try:
            with open(SYN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info(f"同义词缓存已保存: {len(cache)}个节点")
        except Exception as e:
            logger.warning(f"同义词缓存保存失败: {e}")

    def _batch_generate_synonyms(self, nodes: list[CategoryNode]) -> list[list[str]]:
        items_text = "\n".join(
            [f"{i + 1}. {n.category_name}（现有同义词：{'、'.join(n.syn_list) if n.syn_list else '无'}）"
             for i, n in enumerate(nodes)]
        )
        prompt = f"""请为以下标准分类名称各生成3-5个同义词、别名或常见相关称呼。要求：
1. 同义词必须是该分类的等价或近等价表述
2. 不要生成下位词（如"苹果"不是"水果"的同义词）
3. 不要生成上位词
4. 返回JSON数组格式

分类列表：
{items_text}

请以JSON格式返回：
{{"results": [{{"index": 1, "synonyms": ["同义词1", "同义词2"]}}, {{"index": 2, "synonyms": ["同义词1", "同义词2"]}}]}}"""

        try:
            response = self._llm._call_llm(prompt, system_prompt="你是一个专业的产品分类标准化专家，精通各行业的专业术语和俗称。")
            result = self._llm._parse_json_response(response)
            results_data = result.get("results", [])
            index_map = {r.get("index", 0): r.get("synonyms", []) for r in results_data}
            return [index_map.get(i + 1, []) for i in range(len(nodes))]
        except Exception as e:
            logger.warning(f"批量同义词生成失败: {e}, 逐个生成")
            return [self._single_generate_synonyms(n) for n in nodes]

    def _single_generate_synonyms(self, node: CategoryNode) -> list[str]:
        syn_text = "、".join(node.syn_list) if node.syn_list else "无"
        prompt = f"""请为以下标准分类名称生成3-5个同义词、别名或常见相关称呼。要求：
1. 同义词必须是该分类的等价或近等价表述
2. 不要生成下位词或上位词
3. 不要与现有同义词重复

分类名称：{node.category_name}
现有同义词：{syn_text}

请以JSON格式返回：{{"synonyms": ["同义词1", "同义词2"]}}"""

        try:
            response = self._llm._call_llm(prompt, system_prompt="你是一个专业的产品分类标准化专家。")
            result = self._llm._parse_json_response(response)
            return result.get("synonyms", [])
        except Exception as e:
            logger.warning(f"同义词生成失败: category={node.category_name}, error={e}")
            return []

    def _update_db_synonyms(self, category_id: str, syn_list: list[str]) -> None:
        try:
            self._db.execute(
                """UPDATE category_vectors SET syn_list = %s, updated_at = CURRENT_TIMESTAMP
                   WHERE category_id = %s""",
                (syn_list, category_id),
            )
        except Exception as e:
            logger.warning(f"数据库同义词更新失败: category_id={category_id}, error={e}")

        try:
            self._db.execute(
                """UPDATE category_texts SET syn_list = %s
                   WHERE category_id = %s""",
                (syn_list, category_id),
            )
        except Exception:
            pass