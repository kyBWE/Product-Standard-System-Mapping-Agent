from __future__ import annotations
import logging
import sys

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.index.category_enricher import CategoryEnricher
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger


def enrich_categories(
    config_path: str = "config.yaml",
    max_nodes: int = 0,
    batch_size: int = 5,
    existing_syn_only: bool = False,
) -> int:
    logger = StructuredLogger()

    try:
        config = ConfigManager(config_path)
    except FileNotFoundError as e:
        logger.error("EnrichCategories", f"配置文件加载失败: {e}")
        return 0

    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")

    db = DBConnectionManager(db_config)
    try:
        db.initialize()
    except Exception as e:
        logger.error("EnrichCategories", f"数据库连接失败: {e}")
        return 0

    reader = ExcelDataReader()
    try:
        nodes, _ = reader.load_standard_system(standard_file)
    except Exception as e:
        logger.error("EnrichCategories", f"标准体系加载失败: {e}")
        db.close()
        return 0

    llm = LLMAdapter(llm_config)
    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
    )
    trgm_mgr = TrgmIndexManager(db)
    enricher = CategoryEnricher(db, llm, vec_mgr=vec_mgr, trgm_mgr=trgm_mgr)

    updated = enricher.enrich_categories(
        nodes,
        batch_size=batch_size,
        max_nodes=max_nodes,
        existing_syn_only=existing_syn_only,
    )

    db.close()
    logger.info("EnrichCategories", f"分类扩展全部完成: 更新{updated}个节点")
    return updated


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    r = enrich_categories(cfg, max_nodes=mx, batch_size=bs)
    print(f"分类扩展完成: 更新{r}个节点")
