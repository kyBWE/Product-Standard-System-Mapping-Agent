from __future__ import annotations
import logging
import sys

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.engine.page_index_engine import PageIndexEngine
from src.engine.rag_match_engine import RAGMatchEngine
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger
from src.models.enums import EngineType
from src.models.match_result import MatchResult
from src.orchestration.match_orchestrator import MatchOrchestrator
from src.orchestration.result_exporter import ResultExporter
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler


def run_matching(
    config_path: str = "config.yaml",
    engine_type: str = "rag",
    product_name: str | None = None,
) -> list[MatchResult]:
    """
    匹配测试入口函数
    执行产品-标准匹配，支持单条测试与批量匹配
    """
    logger = StructuredLogger()

    try:
        config = ConfigManager(config_path)
    except FileNotFoundError as e:
        logger.error("MatchRunner", f"配置文件加载失败: {e}")
        return []

    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    output_dir = config.get("data.output_dir", "./output")
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")

    db = DBConnectionManager(db_config)
    try:
        db.initialize()
    except Exception as e:
        logger.error("MatchRunner", f"数据库连接失败: {e}")
        return []

    llm = LLMAdapter(llm_config)
    trgm_mgr = TrgmIndexManager(db)
    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
    )

    etype = EngineType.RAG_VECTOR if engine_type.lower() == "rag" else EngineType.PAGE_INDEX

    if etype == EngineType.PAGE_INDEX:
        reader = ExcelDataReader()
        try:
            nodes, _ = reader.load_standard_system(standard_file)
        except Exception as e:
            logger.error("MatchRunner", f"标准体系加载失败: {e}")
            db.close()
            return []
        tree = PageIndexTree()
        tree.build_tree(nodes)
        page_engine = PageIndexEngine(tree, llm)
        rag_engine = None
    else:
        rag_engine = RAGMatchEngine(vec_mgr, trgm_mgr, llm, match_config, enable_llm=match_config.enable_llm)
        page_engine = None

    if rag_engine is None:
        rag_engine = RAGMatchEngine(vec_mgr, trgm_mgr, llm, match_config, enable_llm=match_config.enable_llm)
    if page_engine is None:
        tree = PageIndexTree()
        page_engine = PageIndexEngine(tree, llm)

    excel_reader = ExcelDataReader()
    scheduler = SelfEvolveScheduler(llm, db, excel_reader, match_config, standard_file)
    exporter = ResultExporter(output_dir)

    orchestrator = MatchOrchestrator(
        rag_engine=rag_engine,
        page_engine=page_engine,
        scheduler=scheduler,
        trgm_mgr=trgm_mgr,
        exporter=exporter,
        output_dir=output_dir,
    )

    if product_name:
        results = [orchestrator.run_single(product_name, etype)]
    else:
        product_file = config.get("data.company_product_file", "temp_company_product_0522_1.xlsx")
        try:
            products = excel_reader.load_company_products(product_file)
        except Exception as e:
            logger.error("MatchRunner", f"产品数据加载失败: {e}")
            db.close()
            return []
        results = orchestrator.run_batch(products, etype)

    db.close()
    logger.info("MatchRunner", f"匹配完成: 共{len(results)}条")
    return results


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    eng = sys.argv[2] if len(sys.argv) > 2 else "rag"
    pn = sys.argv[3] if len(sys.argv) > 3 else None
    r = run_matching(cfg, eng, pn)
    for item in r:
        print(f"{item.product_name} -> {item.matched_category_id} ({item.confidence:.4f}, {item.match_status.value})")
