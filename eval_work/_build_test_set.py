import json, sys, os, random, time, logging
import openpyxl

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.engine.llm_adapter import LLMAdapter
from src.index.vector_index_manager import VectorIndexManager
from src.index.trgm_index_manager import TrgmIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.data.excel_reader import ExcelDataReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildTestSet")

def main():
    random.seed(42)

    config = ConfigManager("config.yaml")
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    embedding_config = config.get_embedding_config()

    db = DBConnectionManager(db_config)
    db.initialize()

    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
        embedding_config=embedding_config,
    )
    vec_mgr.ensure_pgvector_ready()
    vec_mgr.warmup()

    trgm_mgr = TrgmIndexManager(db)

    reader = ExcelDataReader()
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")
    nodes, _ = reader.load_standard_system(standard_file)
    cat_map = {n.category_id: {"name": n.category_name, "path": n.category_group_name or ""} for n in nodes}

    wb = openpyxl.load_workbook("temp_company_product_0522_1.xlsx", read_only=True)
    ws = wb.active
    all_products = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and isinstance(row[0], str) and len(row[0].strip()) > 1:
            all_products.append(row[0].strip())
    wb.close()
    logger.info(f"加载{len(all_products)}条产品名")

    sample = random.sample(all_products, min(200, len(all_products)))
    logger.info(f"随机抽取{len(sample)}条")

    llm = LLMAdapter(llm_config)

    results = []
    for idx, pname in enumerate(sample):
        try:
            query_vec = vec_mgr.embed_query(pname)
            vec_results = vec_mgr.search_by_vector(query_vec, top_k=10)
            trgm_results = trgm_mgr.search_by_trgm(pname, threshold=0.1, limit=10)

            candidates = {}
            for r in vec_results:
                candidates[r.category_id] = r.category_name
            for r in trgm_results:
                candidates[r.category_id] = r.category_name

            if not candidates:
                continue

            cand_list = [(cid, cname, []) for cid, cname in list(candidates.items())[:15]]
            sel_idx, conf, reason = llm.select_best_category(pname, cand_list)

            if sel_idx is not None and sel_idx < len(cand_list):
                gt_id = cand_list[sel_idx][0]
                if gt_id in cat_map:
                    results.append({
                        "product_name": pname,
                        "ground_truth": gt_id,
                        "ground_truth_name": cat_map[gt_id]["name"],
                        "path": cat_map[gt_id]["path"],
                    })
        except Exception as e:
            logger.warning(f"处理失败: {pname}, {e}")

        if (idx + 1) % 20 == 0:
            logger.info(f"进度: {idx+1}/{len(sample)}, 成功{len(results)}条")

    logger.info(f"标注完成: {len(results)}/{len(sample)}条有效")

    with open("test_set_random_200.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: test_set_random_200.json")

    db.close()

if __name__ == "__main__":
    main()
