from __future__ import annotations
import logging
import sys

from src.data.excel_reader import ExcelDataReader
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger
from src.models.api_result import LoadDataResult


def load_data(config_path: str = "config.yaml") -> LoadDataResult:
    """
    数据加载入口函数
    从Excel读取标准体系与产品数据，构建全部索引并写入PostgreSQL
    """
    logger = StructuredLogger()
    result = LoadDataResult()

    try:
        config = ConfigManager(config_path)
    except FileNotFoundError as e:
        logger.error("ConfigLoader", f"配置文件加载失败: {e}")
        result.errors.append(str(e))
        return result

    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    data_config = {
        "standard_file": config.get("data.standard_system_file", "产品标准体系.xlsx"),
        "product_file": config.get("data.company_product_file", "temp_company_product_0522_1.xlsx"),
        "output_dir": config.get("data.output_dir", "./output"),
    }

    db = DBConnectionManager(db_config)
    try:
        db.initialize()
        logger.info("LoadData", "数据库连接初始化成功")
    except Exception as e:
        logger.error("LoadData", f"数据库连接失败: {e}")
        result.errors.append(f"数据库连接失败: {e}")
        return result

    reader = ExcelDataReader()

    try:
        nodes, skipped = reader.load_standard_system(data_config["standard_file"])
        result.total_categories = len(nodes)
        result.skipped_rows = skipped
        logger.info("LoadData", f"标准体系加载完成: {len(nodes)}条, 跳过{skipped}条")
    except Exception as e:
        logger.error("LoadData", f"标准体系加载失败: {e}")
        result.errors.append(f"标准体系加载失败: {e}")
        db.close()
        return result

    try:
        product_count = reader.count_company_products(data_config["product_file"])
        result.total_products = product_count
        logger.info("LoadData", f"产品数据加载完成: {product_count}条")
    except Exception as e:
        logger.error("LoadData", f"产品数据加载失败: {e}")
        result.errors.append(f"产品数据加载失败: {e}")

    tree = PageIndexTree()
    try:
        tree.build_tree(nodes)
        result.page_index_status = True
        logger.info("LoadData", "PageIndex树形索引构建成功")
    except Exception as e:
        logger.error("LoadData", f"PageIndex索引构建失败: {e}")
        result.errors.append(f"PageIndex索引构建失败: {e}")

    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
    )
    try:
        vec_mgr.create_vector_table()
        success = vec_mgr.insert_category_vectors(nodes)
        result.vector_index_status = success > 0
        vec_mgr.ensure_pgvector_ready()
        logger.info("LoadData", f"向量索引构建完成: 成功{success}条")
    except Exception as e:
        logger.error("LoadData", f"向量索引构建失败: {e}")
        result.errors.append(f"向量索引构建失败: {e}")

    trgm_mgr = TrgmIndexManager(db)
    try:
        trgm_mgr.create_trgm_index()
        success = trgm_mgr.insert_category_texts(nodes)
        result.trgm_index_status = success > 0
        logger.info("LoadData", f"文本索引构建完成: 成功{success}条")
    except Exception as e:
        logger.error("LoadData", f"文本索引构建失败: {e}")
        result.errors.append(f"文本索引构建失败: {e}")

    db.close()
    logger.info("LoadData", f"数据加载全部完成: {result}")
    return result


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    result = load_data(config_path)
    print(f"数据加载结果: 总分类={result.total_categories}, 总产品={result.total_products}")
    print(f"向量索引={result.vector_index_status}, 文本索引={result.trgm_index_status}, 树形索引={result.page_index_status}")
    if result.errors:
        print(f"错误: {result.errors}")
