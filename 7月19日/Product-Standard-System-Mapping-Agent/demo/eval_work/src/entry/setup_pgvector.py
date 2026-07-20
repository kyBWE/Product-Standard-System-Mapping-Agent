from __future__ import annotations
"""启用 pgvector 扩展、回填 vec_search 并创建 HNSW 索引。"""
import sys

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.index.vector_index_manager import VectorIndexManager


def setup_pgvector(config_path: str = "config.yaml") -> bool:
    config = ConfigManager(config_path)
    db = DBConnectionManager(config.get_db_config())
    db.initialize()

    llm_config = config.get_llm_config()
    vec_mgr = VectorIndexManager(
        db,
        embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
    )

    pg_ok = vec_mgr.ensure_pgvector_ready()
    vec_mgr.warmup()

    if pg_ok:
        rows = db.execute_one(
            "SELECT COUNT(*) AS total, COUNT(vec_search) AS filled FROM category_vectors"
        )
        print(f"pgvector 已启用: 总向量={rows['total']}, vec_search已填={rows['filled']}")
    else:
        print("pgvector 扩展未安装，已回退到内存矩阵检索（启动预热后同样可用）")
        print("Windows + PostgreSQL 18 需编译安装 pgvector，见 scripts/install_pgvector_win.md")

    db.close()
    return pg_ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    ok = setup_pgvector(path)
    sys.exit(0 if ok else 1)
