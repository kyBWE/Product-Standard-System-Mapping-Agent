# -*- coding: utf-8 -*-
"""从 DB 全量重建 bge-m3-1024 向量（无需 Excel）。"""
from __future__ import annotations
import os, sys, time, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RebuildFull")

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.index.vector_index_manager import VectorIndexManager
from src.models.category_node import CategoryNode

def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    llm = config.get_llm_config()
    emb = config.get_embedding_config()

    rows = db.execute(
        "SELECT category_id, category_name, syn_list, category_group_name FROM category_texts"
    )
    # 重建前保留旧表同义词（可能比 texts 更完整）
    vec_syn = {}
    try:
        for r in db.execute("SELECT category_id, syn_list FROM category_vectors"):
            syn = r["syn_list"] or []
            if syn:
                vec_syn[str(r["category_id"])] = list(syn) if not isinstance(syn, list) else syn
    except Exception as e:
        logger.warning(f"load vec syn: {e}")

    nodes = []
    for r in rows:
        cid = str(r["category_id"])
        syn = r["syn_list"] or []
        if not isinstance(syn, list):
            syn = list(syn) if syn else []
        if not syn and cid in vec_syn:
            syn = vec_syn[cid]
        nodes.append(CategoryNode(
            category_id=cid,
            category_name=r["category_name"] or "",
            syn_list=syn,
            category_group_name=r.get("category_group_name") or "",
        ))
    logger.info(f"从 category_texts 加载 {len(nodes)} 个节点, 含同义词节点={sum(1 for n in nodes if n.syn_list)}")

    # 清空旧向量并去掉 512 维列，避免维度冲突
    try:
        db.execute("DROP INDEX IF EXISTS idx_category_vectors_vec_search")
    except Exception as e:
        logger.warning(f"drop index: {e}")
    db.execute("DELETE FROM category_vectors")
    try:
        db.execute("ALTER TABLE category_vectors DROP COLUMN IF EXISTS vec_search")
    except Exception as e:
        logger.warning(f"drop vec_search: {e}")

    vec_mgr = VectorIndexManager(
        db,
        embedding_model=llm.embedding_model,
        embedding_dimension=llm.embedding_dimension,
        base_url=llm.base_url,
        api_key=llm.api_key,
        embedding_config=emb,
    )
    vec_mgr.create_vector_table()
    # 表已存在时 CREATE IF NOT EXISTS 不会补列；显式加回 1024 维
    dim = llm.embedding_dimension or 1024
    try:
        db.execute(f"ALTER TABLE category_vectors ADD COLUMN IF NOT EXISTS vec_search vector({dim})")
        logger.info(f"已确保 vec_search vector({dim}) 列存在")
    except Exception as e:
        logger.warning(f"add vec_search: {e}")
    vec_mgr._use_pgvector = True

    t0 = time.time()
    success = vec_mgr._insert_api_vectors(nodes)
    logger.info(f"写入完成 success={success}/{len(nodes)} 耗时={time.time()-t0:.1f}s")

    ok = vec_mgr.ensure_pgvector_ready()
    logger.info(f"pgvector ready={ok}")

    # 校验维度与覆盖
    dim_rows = db.execute(
        "SELECT vector_dims(vec_search) AS dim, COUNT(*) AS n FROM category_vectors "
        "WHERE vec_search IS NOT NULL GROUP BY 1"
    )
    for r in dim_rows:
        logger.info(f"dim={r['dim']} count={r['n']}")

    q = vec_mgr.embed_query("罗茨风机")
    hits = vec_mgr.search_by_vector(q, top_k=5)
    for h in hits:
        logger.info(f"  {h.category_name}({h.category_id}) sim={h.similarity:.4f}")

    db.close()
    logger.info("全部完成")

if __name__ == "__main__":
    main()
