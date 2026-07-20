from __future__ import annotations
import logging

from src.infrastructure.db_manager import DBConnectionManager
from src.models.category_node import CategoryNode
from src.models.index_result import TrgmSearchResult, ExactTextMatch


logger = logging.getLogger("TrgmIndexManager")


class TrgmIndexManager:
    def __init__(self, db: DBConnectionManager):
        self._db = db

    def create_trgm_index(self) -> None:
        self._db.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS category_texts (
                category_id         TEXT        PRIMARY KEY,
                category_name       TEXT        NOT NULL,
                syn_list            TEXT[]      DEFAULT '{}',
                category_pids       TEXT[]      DEFAULT '{}',
                category_group_name TEXT        NOT NULL DEFAULT '',
                created_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("category_texts 表创建/已存在")

    def insert_category_texts(self, nodes: list[CategoryNode]) -> int:
        db_syn_map = self._load_syn_list_from_db()
        if db_syn_map:
            logger.info(f"从数据库加载{len(db_syn_map)}个节点的同义词列表")

        from src.index.category_enricher import CategoryEnricher
        cache_applied = CategoryEnricher.apply_syn_cache(nodes)
        if cache_applied:
            logger.info(f"从同义词缓存应用{cache_applied}个节点的同义词")

        success = 0
        rows: list[tuple] = []
        for node in nodes:
            try:
                syn_list = node.syn_list
                if not syn_list and node.category_id in db_syn_map:
                    syn_list = db_syn_map[node.category_id]
                rows.append((
                    node.category_id, node.category_name, syn_list,
                    node.category_pids, node.category_group_name,
                ))
                success += 1
            except Exception as e:
                logger.warning(f"准备文本数据异常: category_id={node.category_id}, error={e}")

        if rows:
            try:
                self._db.execute_values_batch(
                    """INSERT INTO category_texts
                       (category_id, category_name, syn_list, category_pids, category_group_name)
                       VALUES %s
                       ON CONFLICT (category_id) DO UPDATE
                       SET category_name=EXCLUDED.category_name,
                           syn_list=EXCLUDED.syn_list,
                           category_pids=EXCLUDED.category_pids,
                           category_group_name=EXCLUDED.category_group_name,
                           updated_at=CURRENT_TIMESTAMP""",
                    rows,
                    page_size=500,
                )
            except Exception as e:
                logger.error(f"文本数据批量写入失败: {e}")
                success = 0

        try:
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_category_texts_name_trgm
                ON category_texts USING gin (category_name gin_trgm_ops);
            """)
        except Exception as e:
            logger.warning(f"category_name GIN索引创建失败: {e}")

        try:
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_category_texts_syn_trgm
                ON category_texts USING gin (syn_list);
            """)
        except Exception as e:
            logger.warning(f"syn_list GIN索引创建失败: {e}")

        try:
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_category_texts_pids
                ON category_texts USING gin (category_pids);
            """)
        except Exception as e:
            logger.warning(f"category_pids GIN索引创建失败: {e}")

        logger.info(f"文本数据写入完成: 成功{success}/{len(nodes)}条")
        return success

    def lookup_exact_match(self, query_text: str) -> ExactTextMatch | None:
        """产品名与分类名或同义词完全一致时返回对应节点。"""
        query = (query_text or "").strip()
        if not query:
            return None
        try:
            row = self._db.execute_one(
                """SELECT category_id, category_name
                   FROM category_texts
                   WHERE category_name = %s
                   LIMIT 1""",
                (query,),
            )
            if row:
                return ExactTextMatch(
                    category_id=row["category_id"],
                    category_name=row["category_name"],
                    match_type="name",
                )
            row = self._db.execute_one(
                """SELECT category_id, category_name
                   FROM category_texts
                   WHERE %s = ANY(syn_list)
                   LIMIT 1""",
                (query,),
            )
            if row:
                return ExactTextMatch(
                    category_id=row["category_id"],
                    category_name=row["category_name"],
                    match_type="synonym",
                )
        except Exception as e:
            logger.warning(f"精确文本匹配查询失败: {e}")
        return None

    def search_by_trgm(self, query_text: str, threshold: float = 0.3, limit: int = 20) -> list[TrgmSearchResult]:
        try:
            rows = self._db.execute(
                """SELECT category_id, category_name, MAX(sim) AS similarity FROM (
                    SELECT category_id, category_name,
                           similarity(category_name, %s) AS sim
                    FROM category_texts
                    WHERE similarity(category_name, %s) >= %s
                       OR category_name = %s
                    UNION ALL
                    SELECT ct.category_id, ct.category_name,
                           similarity(u.syn, %s) AS sim
                    FROM category_texts ct, unnest(ct.syn_list) AS u(syn)
                    WHERE similarity(u.syn, %s) >= %s
                ) sub
                GROUP BY category_id, category_name
                LIMIT %s""",
                (query_text, query_text, threshold, query_text,
                 query_text, query_text, threshold,
                 limit * 3),
            )
            syn_map = self._load_syn_list_for_ids([r["category_id"] for r in rows])
            scored: list[TrgmSearchResult] = []
            for r in rows:
                effective = self._effective_trgm_score(
                    query_text,
                    r["category_name"],
                    syn_map.get(r["category_id"], []),
                )
                if effective >= threshold:
                    scored.append(TrgmSearchResult(
                        category_id=r["category_id"],
                        category_name=r["category_name"],
                        similarity=effective,
                    ))
            scored.sort(
                key=lambda x: (
                    1 if x.category_name == query_text else 0,
                    x.similarity,
                ),
                reverse=True,
            )
            return scored[:limit]
        except Exception as e:
            logger.error(f"pg_trgm检索失败: {e}")
            return []

    def _load_syn_list_for_ids(self, category_ids: list[str]) -> dict[str, list[str]]:
        if not category_ids:
            return {}
        placeholders = ",".join(["%s"] * len(category_ids))
        rows = self._db.execute(
            f"SELECT category_id, syn_list FROM category_texts WHERE category_id IN ({placeholders})",
            tuple(category_ids),
        )
        result: dict[str, list[str]] = {}
        for r in rows:
            syn = r["syn_list"]
            if syn:
                result[r["category_id"]] = list(syn) if isinstance(syn, list) else []
        return result

    def _effective_trgm_score(self, query: str, category_name: str, syn_list: list[str]) -> float:
        """pg_trgm 对中文短同义词易误报 1.0，按名称优先并折扣同义词命中。"""
        if query == category_name:
            return 1.0

        name_sim = self.get_trgm_similarity(query, category_name)
        best_syn_sim = 0.0
        for syn in syn_list:
            if not syn:
                continue
            if query == syn:
                return 0.95
            syn_sim = self.get_trgm_similarity(query, syn)
            overlap = min(len(query), len(syn)) / max(len(query), len(syn), 1)
            best_syn_sim = max(best_syn_sim, syn_sim * overlap * 0.85)

        return max(name_sim, best_syn_sim)

    def search_by_substring(self, query_text: str, limit: int = 20) -> list[TrgmSearchResult]:
        try:
            rows = self._db.execute(
                """SELECT category_id, category_name,
                          similarity(category_name, %s) AS similarity
                   FROM category_texts
                   WHERE position(category_name in %s) > 0
                      OR position(%s in category_name) > 0
                   ORDER BY similarity DESC
                   LIMIT %s""",
                (query_text, query_text, query_text, limit),
            )
            return [
                TrgmSearchResult(
                    category_id=r["category_id"],
                    category_name=r["category_name"],
                    similarity=float(r["similarity"]),
                )
                for r in rows
            ]
        except Exception as e:
            logger.error(f"子串匹配检索失败: {e}")
            return []

    def _load_syn_list_from_db(self) -> dict[str, list[str]]:
        try:
            rows = self._db.execute("SELECT category_id, syn_list FROM category_vectors")
            if not rows:
                rows = self._db.execute("SELECT category_id, syn_list FROM category_texts")
            if not rows:
                return {}
            result = {}
            for r in rows:
                syn = r["syn_list"]
                if syn:
                    result[r["category_id"]] = list(syn) if isinstance(syn, list) else []
            return result
        except Exception:
            return {}

    def get_trgm_similarity(self, text1: str, text2: str) -> float:
        try:
            result = self._db.execute_one(
                "SELECT similarity(%s, %s) AS sim",
                (text1, text2),
            )
            return float(result["sim"]) if result else 0.0
        except Exception:
            return 0.0
