from __future__ import annotations
import hashlib
import logging
import pickle
import math
import struct
import os
import numpy as np
import jieba
from collections import Counter

from src.infrastructure.db_manager import DBConnectionManager
from src.models.category_node import CategoryNode
from src.models.index_result import VectorSearchResult


logger = logging.getLogger("VectorIndexManager")

TFIDF_DIM = 512
NGRAM_DIM = 512
AXIS_DIM = 35
TFIDF_TOTAL_DIM = TFIDF_DIM + NGRAM_DIM + AXIS_DIM

ONNX_EMBEDDING_DIM = 512
EMBEDDING_DIM = ONNX_EMBEDDING_DIM

SEMANTIC_AXES = [
    "农林牧渔", "食品饮料", "纺织服装", "木材家具", "造纸印刷",
    "石油化工", "医药", "建材", "钢铁有色", "金属制品",
    "通用设备", "专用设备", "汽车", "电气机械", "电子信息",
    "仪器仪表", "其他制造", "电力热力", "燃气水务", "建筑",
    "批发零售", "交通运输", "仓储邮政", "住宿餐饮", "信息技术",
    "金融", "房地产", "租赁商务", "科学研究", "水利环境",
    "居民服务", "教育", "卫生社会", "文化体育", "公共管理",
]


def _stable_hash(token: str, seed: int = 0) -> int:
    h = hashlib.md5((token + str(seed)).encode("utf-8")).digest()
    return struct.unpack("<I", h[:4])[0]


class VectorIndexManager:
    def __init__(self, db: DBConnectionManager, embedding_model: str = "onnx",
                 embedding_dimension: int = 0, base_url: str = "", api_key: str = ""):
        self._db = db
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._base_url = base_url
        self._api_key = api_key
        self._use_pgvector = False
        self._use_onnx = False
        self._onnx_embedder = None
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._category_ids: list[str] = []
        self._category_names: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}
        self._vocab: list[str] = []
        self._vocab_idx: dict[str, int] = {}
        self._axis_cache: dict[str, np.ndarray] = {}
        self._vector_matrix: np.ndarray | None = None
        self._matrix_category_ids: list[str] = []
        self._matrix_category_names: list[str] = []
        self._warmed_up = False
        self._pgvector_setup_attempted = False

        self._init_embedder()

    def _init_embedder(self) -> None:
        if self._embedding_model == "onnx":
            try:
                from src.index.onnx_embedder import ONNXEmbedder
                self._onnx_embedder = ONNXEmbedder()
                self._use_onnx = True
                logger.info(f"ONNX embedder initialized, dim={ONNX_EMBEDDING_DIM}")
            except Exception as e:
                logger.warning(f"ONNX embedder init failed: {e}, falling back to TF-IDF")
                self._use_onnx = False

    def _check_pgvector(self) -> None:
        try:
            rows = self._db.execute("SELECT typname FROM pg_type WHERE typname = 'vector'")
            if rows:
                self._use_pgvector = True
                logger.info("pgvector已安装，将使用vector类型检索")
            else:
                self._use_pgvector = False
                logger.info("未检测到pgvector，使用内存矩阵检索")
        except Exception:
            self._use_pgvector = False

    def ensure_pgvector_ready(self) -> bool:
        """尝试启用 pgvector 扩展、补齐 vec_search 列并创建 HNSW 索引。"""
        if not self._pgvector_setup_attempted:
            self._pgvector_setup_attempted = True
            try:
                self._db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                logger.warning(f"pgvector扩展不可用: {e}")

        self._check_pgvector()
        if not self._use_pgvector:
            return False

        if not self._has_vec_search_column():
            try:
                self._db.execute(
                    f"ALTER TABLE category_vectors ADD COLUMN IF NOT EXISTS "
                    f"vec_search vector({EMBEDDING_DIM})"
                )
                logger.info("已添加 category_vectors.vec_search 列")
            except Exception as e:
                logger.warning(f"添加 vec_search 列失败: {e}")
                self._use_pgvector = False
                return False

        backfilled = self._backfill_vec_search()
        if backfilled:
            logger.info(f"已从 embedding 回填 {backfilled} 条 vec_search")

        self._ensure_hnsw_index()
        return True

    def _has_vec_search_column(self) -> bool:
        try:
            rows = self._db.execute(
                """SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'category_vectors' AND column_name = 'vec_search'"""
            )
            return bool(rows)
        except Exception:
            return False

    def _backfill_vec_search(self, batch_size: int = 200) -> int:
        if not self._use_pgvector or not self._has_vec_search_column():
            return 0

        total = 0
        while True:
            rows = self._db.execute(
                """SELECT category_id, embedding FROM category_vectors
                   WHERE vec_search IS NULL AND embedding IS NOT NULL
                   LIMIT %s""",
                (batch_size,),
            )
            if not rows:
                break

            conn = self._db.get_connection()
            try:
                with conn.cursor() as cur:
                    for row in rows:
                        emb = row["embedding"]
                        if emb is None:
                            continue
                        if isinstance(emb, memoryview):
                            vec = pickle.loads(bytes(emb))
                        elif isinstance(emb, bytes):
                            vec = pickle.loads(emb)
                        else:
                            continue
                        vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
                        cur.execute(
                            "UPDATE category_vectors SET vec_search = %s::vector WHERE category_id = %s",
                            (vec_str, row["category_id"]),
                        )
                conn.commit()
                total += len(rows)
            except Exception as e:
                conn.rollback()
                logger.warning(f"vec_search回填失败: {e}")
                break
            finally:
                self._db.put_connection(conn)

        return total

    def _ensure_hnsw_index(self) -> None:
        if not self._use_pgvector:
            return
        try:
            self._db.execute(
                """CREATE INDEX IF NOT EXISTS idx_category_vectors_vec_search
                   ON category_vectors USING hnsw (vec_search vector_cosine_ops)"""
            )
            logger.info("HNSW向量索引已就绪")
        except Exception as e:
            logger.warning(f"HNSW索引创建失败: {e}")

    def warmup(self) -> None:
        """服务启动预热：启用 pgvector 或构建内存矩阵，并执行一次探针检索。"""
        if self._warmed_up:
            return

        pg_ok = self.ensure_pgvector_ready()
        if self._use_onnx and self._onnx_embedder:
            self._onnx_embedder.embed("预热")

        probe_vec = self.embed_query("预热")
        if pg_ok:
            self._search_pgvector(probe_vec, top_k=1)
            logger.info("向量索引预热完成: 模式=pgvector")
        else:
            self._ensure_vector_matrix()
            self._search_matrix(probe_vec, top_k=1)
            logger.info(
                f"向量索引预热完成: 模式=matrix, 向量数={len(self._matrix_category_ids)}"
            )

        self._warmed_up = True

    def _ensure_vector_matrix(self) -> None:
        if self._vector_matrix is not None:
            return
        if not self._vectors:
            self._load_vectors_from_db()
        if not self._vectors:
            return

        ids: list[str] = []
        names: list[str] = []
        rows: list[np.ndarray] = []
        for cat_id, vec in self._vectors.items():
            arr = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            ids.append(cat_id)
            idx = self._category_ids.index(cat_id) if cat_id in self._category_ids else -1
            names.append(self._category_names[idx] if idx >= 0 else cat_id)
            rows.append(arr)

        if not rows:
            return

        self._matrix_category_ids = ids
        self._matrix_category_names = names
        self._vector_matrix = np.vstack(rows)
        logger.info(f"内存向量矩阵已构建: {self._vector_matrix.shape[0]} x {self._vector_matrix.shape[1]}")

    def _search_matrix(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        self._ensure_vector_matrix()
        if self._vector_matrix is None or len(self._matrix_category_ids) == 0:
            return []

        query_arr = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return []
        query_arr = query_arr / query_norm

        sims = self._vector_matrix @ query_arr
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []

        if k == sims.shape[0]:
            top_idx = np.argsort(-sims)[:k]
        else:
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[VectorSearchResult] = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim <= 0.01:
                continue
            results.append(
                VectorSearchResult(
                    category_id=self._matrix_category_ids[idx],
                    category_name=self._matrix_category_names[idx],
                    similarity=sim,
                )
            )
        return results[:top_k]

    def create_vector_table(self) -> None:
        self._check_pgvector()
        if self._use_pgvector:
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS category_vectors (
                    category_id     TEXT        PRIMARY KEY,
                    category_name   TEXT        NOT NULL,
                    syn_list        TEXT[]      DEFAULT '{{}}',
                    embedding       BYTEA,
                    vec_search      vector({EMBEDDING_DIM}),
                    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
                );
            """)
        else:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS category_vectors (
                    category_id     TEXT        PRIMARY KEY,
                    category_name   TEXT        NOT NULL,
                    syn_list        TEXT[]      DEFAULT '{}',
                    embedding       BYTEA,
                    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
                );
            """)
        logger.info("category_vectors 表创建/已存在")

    def insert_category_vectors(self, nodes: list[CategoryNode]) -> int:
        self._category_ids = []
        self._category_names = []

        db_syn_map = self._load_syn_list_from_db()
        if db_syn_map:
            logger.info(f"从数据库加载{len(db_syn_map)}个节点的同义词列表")

        from src.index.category_enricher import CategoryEnricher
        cache_applied = CategoryEnricher.apply_syn_cache(nodes)
        if cache_applied:
            logger.info(f"从同义词缓存应用{cache_applied}个节点的同义词")

        for node in nodes:
            if node.category_id in db_syn_map and not node.syn_list:
                node.syn_list = db_syn_map[node.category_id]
            self._category_ids.append(node.category_id)
            self._category_names.append(node.category_name)

        if self._use_onnx:
            return self._insert_onnx_vectors(nodes)
        else:
            return self._insert_tfidf_vectors(nodes)

    def _insert_onnx_vectors(self, nodes: list[CategoryNode]) -> int:
        texts = []
        for node in nodes:
            text_parts = [node.category_name] + node.syn_list
            texts.append(" ".join(text_parts))

        total = len(texts)
        chunk_size = 16
        success = 0

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_texts = texts[chunk_start:chunk_end]
            chunk_nodes = nodes[chunk_start:chunk_end]

            try:
                embeddings = self._onnx_embedder.embed_batch(chunk_texts)
            except Exception as e:
                logger.warning(f"ONNX embedding batch failed (start={chunk_start}): {e}")
                embeddings = [np.zeros(ONNX_EMBEDDING_DIM, dtype=np.float32)] * len(chunk_texts)

            chunk_rows = []
            chunk_vec_strs = []
            for i, (node, emb) in enumerate(zip(chunk_nodes, embeddings)):
                try:
                    vec = emb.astype(np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    self._vectors[node.category_id] = vec
                    embedding_bytes = pickle.dumps(vec)
                    chunk_rows.append((node.category_id, node.category_name, node.syn_list, embedding_bytes))
                    if self._use_pgvector:
                        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
                        chunk_vec_strs.append((node.category_id, vec_str))
                    success += 1
                except Exception as e:
                    logger.warning(f"向量处理异常: category_id={node.category_id}, error={e}")

            self._write_chunk_to_db(chunk_rows, chunk_vec_strs)
            logger.info(f"ONNX向量写入进度: {chunk_end}/{total}")

        if self._use_pgvector:
            try:
                self._db.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_category_vectors_vec_search
                    ON category_vectors USING hnsw (vec_search vector_cosine_ops);
                """)
            except Exception as e:
                logger.warning(f"HNSW索引创建失败: {e}")

        logger.info(f"ONNX向量写入完成: 成功{success}/{total}条")
        return success

    def _insert_tfidf_vectors(self, nodes: list[CategoryNode]) -> int:
        self._doc_tokens = []
        for node in nodes:
            text_parts = [node.category_name] + node.syn_list
            text = " ".join(text_parts)
            tokens = self._tokenize(text)
            self._doc_tokens.append(tokens)

        n_docs = len(self._doc_tokens)
        df: dict[str, int] = Counter()
        for tokens in self._doc_tokens:
            for t in set(tokens):
                df[t] += 1
        self._idf = {t: math.log(n_docs / (1 + c)) for t, c in df.items()}
        self._vocab = sorted(self._idf.keys())
        self._build_axis_cache(nodes)

        chunk_size = 2000
        total = len(nodes)
        success = 0

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_nodes = nodes[chunk_start:chunk_end]
            chunk_rows = []
            chunk_vec_strs = []

            for i, node in enumerate(chunk_nodes):
                idx = chunk_start + i
                try:
                    tfidf_vec = self._tokens_to_tfidf_vec(self._doc_tokens[idx])
                    ngram_vec = self._text_to_ngram_vec(node.category_name, node.syn_list)
                    axis_vec = self._axis_cache.get(node.category_id, np.zeros(AXIS_DIM, dtype=np.float32))
                    vec = np.concatenate([tfidf_vec, ngram_vec, axis_vec])
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    self._vectors[node.category_id] = vec
                    embedding_bytes = pickle.dumps(vec)
                    chunk_rows.append((node.category_id, node.category_name, node.syn_list, embedding_bytes))
                    if self._use_pgvector:
                        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
                        chunk_vec_strs.append((node.category_id, vec_str))
                    success += 1
                except Exception as e:
                    logger.warning(f"向量计算异常: category_id={node.category_id}, error={e}")

            self._write_chunk_to_db(chunk_rows, chunk_vec_strs)
            logger.info(f"TF-IDF向量写入进度: {chunk_end}/{total}")

        if self._use_pgvector:
            try:
                self._db.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_category_vectors_vec_search
                    ON category_vectors USING hnsw (vec_search vector_cosine_ops);
                """)
            except Exception as e:
                logger.warning(f"HNSW索引创建失败: {e}")

        logger.info(f"TF-IDF向量写入完成: 成功{success}/{total}条")
        return success

    def _write_chunk_to_db(self, rows: list[tuple], vec_strs: list[tuple]) -> None:
        if not rows:
            return
        try:
            self._db.execute_values_batch(
                """INSERT INTO category_vectors (category_id, category_name, syn_list, embedding)
                   VALUES %s
                   ON CONFLICT (category_id) DO UPDATE
                   SET category_name=EXCLUDED.category_name,
                       syn_list=EXCLUDED.syn_list,
                       embedding=EXCLUDED.embedding,
                       updated_at=CURRENT_TIMESTAMP""",
                rows,
                page_size=500,
            )
        except Exception as e:
            logger.error(f"批量写入失败: {e}")

        if self._use_pgvector and vec_strs:
            try:
                self._db.execute_values_batch(
                    """UPDATE category_vectors AS cv
                       SET vec_search = v.vec::vector,
                           updated_at = CURRENT_TIMESTAMP
                       FROM (VALUES %s) AS v(category_id, vec)
                       WHERE cv.category_id = v.category_id""",
                    vec_strs,
                    template="(%s, %s)",
                    page_size=500,
                )
            except Exception as e:
                logger.warning(f"vec_search批量更新失败: {e}")

    def update_category_vectors(self, nodes: list[CategoryNode]) -> int:
        success = 0
        for node in nodes:
            try:
                if self._use_onnx:
                    text_parts = [node.category_name] + node.syn_list
                    text = " ".join(text_parts)
                    vec = self._onnx_embedder.embed(text)
                else:
                    if not self._idf:
                        self._rebuild_from_db()
                    text_parts = [node.category_name] + node.syn_list
                    tokens = self._tokenize(" ".join(text_parts))
                    tfidf_vec = self._tokens_to_tfidf_vec(tokens)
                    ngram_vec = self._text_to_ngram_vec(node.category_name, node.syn_list)
                    axis_vec = self._text_to_axis_vec(node.category_name)
                    for syn in node.syn_list:
                        syn_axis = self._text_to_axis_vec(syn)
                        axis_vec = np.maximum(axis_vec, syn_axis)
                    vec = np.concatenate([tfidf_vec, ngram_vec, axis_vec])

                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._vectors[node.category_id] = vec

                embedding_bytes = pickle.dumps(vec)
                vec_str = "[" + ",".join(str(v) for v in vec) + "]"

                if self._use_pgvector:
                    self._db.execute(
                        """INSERT INTO category_vectors (category_id, category_name, syn_list, embedding, vec_search)
                           VALUES (%s, %s, %s, %s, %s::vector)
                           ON CONFLICT (category_id) DO UPDATE
                           SET category_name=EXCLUDED.category_name,
                               syn_list=EXCLUDED.syn_list,
                               embedding=EXCLUDED.embedding,
                               vec_search=EXCLUDED.vec_search,
                               updated_at=CURRENT_TIMESTAMP""",
                        (node.category_id, node.category_name, node.syn_list, embedding_bytes, vec_str),
                    )
                else:
                    self._db.execute(
                        """INSERT INTO category_vectors (category_id, category_name, syn_list, embedding)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (category_id) DO UPDATE
                           SET category_name=EXCLUDED.category_name,
                               syn_list=EXCLUDED.syn_list,
                               embedding=EXCLUDED.embedding,
                               updated_at=CURRENT_TIMESTAMP""",
                        (node.category_id, node.category_name, node.syn_list, embedding_bytes),
                    )
                success += 1
            except Exception as e:
                logger.warning(f"更新向量异常: category_id={node.category_id}, error={e}")
        logger.info(f"增量向量更新完成: 成功{success}/{len(nodes)}条")
        return success

    def _build_axis_cache(self, nodes: list[CategoryNode]) -> None:
        for node in nodes:
            vec = np.zeros(AXIS_DIM, dtype=np.float32)
            text = node.category_name
            for i, axis in enumerate(SEMANTIC_AXES):
                axis_tokens = set(jieba.lcut(axis))
                node_tokens = set(jieba.lcut(text))
                for syn in node.syn_list:
                    node_tokens.update(jieba.lcut(syn))
                overlap = len(axis_tokens & node_tokens)
                if overlap > 0:
                    vec[i] = overlap / max(len(axis_tokens), 1)
                for at in axis_tokens:
                    if at in text:
                        vec[i] = max(vec[i], 0.5)
            self._axis_cache[node.category_id] = vec

    def search_by_vector(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        if not self._use_pgvector:
            self._check_pgvector()
        if self._use_pgvector:
            return self._search_pgvector(query_vector, top_k)
        return self._search_matrix(query_vector, top_k)

    def _search_pgvector(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        try:
            rows = self._db.execute(
                """SELECT category_id, category_name, 1 - (vec_search <=> %s::vector) AS similarity
                   FROM category_vectors
                   WHERE vec_search IS NOT NULL
                   ORDER BY vec_search <=> %s::vector
                   LIMIT %s""",
                (vec_str, vec_str, top_k),
            )
            return [
                VectorSearchResult(
                    category_id=r["category_id"],
                    category_name=r["category_name"],
                    similarity=float(r["similarity"]),
                )
                for r in rows
            ]
        except Exception as e:
            logger.error(f"pgvector检索失败: {e}, 回退到内存矩阵")
            return self._search_matrix(query_vector, top_k)

    def embed_query(self, text: str) -> list[float]:
        if self._use_onnx and self._onnx_embedder:
            vec = self._onnx_embedder.embed(text)
            return vec.tolist()
        if not self._idf:
            self._rebuild_from_db()
        tokens = self._tokenize(text)
        tfidf_vec = self._tokens_to_tfidf_vec(tokens)
        ngram_vec = self._text_to_ngram_vec(text, [])
        axis_vec = self._text_to_axis_vec(text)
        vec = np.concatenate([tfidf_vec, ngram_vec, axis_vec])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _load_vectors_from_db(self) -> None:
        try:
            rows = self._db.execute("SELECT category_id, category_name, embedding FROM category_vectors")
            if not rows:
                return
            self._category_ids = []
            self._category_names = []
            self._vectors = {}
            for r in rows:
                emb = r["embedding"]
                if emb is None:
                    continue
                if isinstance(emb, memoryview):
                    vec = pickle.loads(bytes(emb))
                elif isinstance(emb, bytes):
                    vec = pickle.loads(emb)
                else:
                    continue
                self._category_ids.append(r["category_id"])
                self._category_names.append(r["category_name"])
                self._vectors[r["category_id"]] = vec
            logger.info(f"从数据库加载{len(self._vectors)}个向量到内存")
            self._vector_matrix = None
        except Exception as e:
            logger.error(f"从数据库加载向量失败: {e}")

    def _load_syn_list_from_db(self) -> dict[str, list[str]]:
        try:
            rows = self._db.execute("SELECT category_id, syn_list FROM category_vectors")
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

    def _text_to_ngram_vec(self, name: str, syn_list: list[str]) -> np.ndarray:
        vec = np.zeros(NGRAM_DIM, dtype=np.float32)
        texts = [name] + syn_list
        for text in texts:
            chars = text.replace(" ", "")
            for n in (2, 3):
                for j in range(len(chars) - n + 1):
                    gram = chars[j:j + n]
                    idx1 = _stable_hash(gram, seed=0) % NGRAM_DIM
                    idx2 = _stable_hash(gram, seed=1) % NGRAM_DIM
                    vec[idx1] += 0.5
                    vec[idx2] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _text_to_axis_vec(self, text: str) -> np.ndarray:
        vec = np.zeros(AXIS_DIM, dtype=np.float32)
        text_tokens = set(jieba.lcut(text))
        for i, axis in enumerate(SEMANTIC_AXES):
            axis_tokens = set(jieba.lcut(axis))
            overlap = len(axis_tokens & text_tokens)
            if overlap > 0:
                vec[i] = overlap / max(len(axis_tokens), 1)
            for at in axis_tokens:
                if at in text:
                    vec[i] = max(vec[i], 0.5)
        return vec

    def _rebuild_from_db(self) -> None:
        try:
            rows = self._db.execute("SELECT category_id, category_name, syn_list FROM category_vectors")
            if not rows:
                return
            self._category_ids = []
            self._category_names = []
            self._doc_tokens = []
            for r in rows:
                self._category_ids.append(r["category_id"])
                self._category_names.append(r["category_name"])
                syn_list = r["syn_list"] if r["syn_list"] else []
                text_parts = [r["category_name"]] + list(syn_list)
                tokens = self._tokenize(" ".join(text_parts))
                self._doc_tokens.append(tokens)
            n_docs = len(self._doc_tokens)
            df: dict[str, int] = Counter()
            for tokens in self._doc_tokens:
                for t in set(tokens):
                    df[t] += 1
            self._idf = {t: math.log(n_docs / (1 + c)) for t, c in df.items()}
            self._vocab = sorted(self._idf.keys())
            logger.info(f"从数据库重建IDF完成: {len(self._idf)}个词")
        except Exception as e:
            logger.error(f"从数据库重建IDF失败: {e}")

    def _tokenize(self, text: str) -> list[str]:
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if len(t.strip()) > 0]

    def _tokens_to_tfidf_vec(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(TFIDF_DIM, dtype=np.float32)
        if not tokens:
            return vec
        tf = Counter(tokens)
        total = len(tokens)
        for t, c in tf.items():
            idf_val = self._idf.get(t, math.log(max(len(self._idf), 100) / 2) if self._idf else 1.0)
            weight = (c / total) * idf_val
            idx1 = _stable_hash(t, seed=0) % TFIDF_DIM
            idx2 = _stable_hash(t, seed=42) % TFIDF_DIM
            vec[idx1] += weight
            if idx2 != idx1:
                vec[idx2] += weight * 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
