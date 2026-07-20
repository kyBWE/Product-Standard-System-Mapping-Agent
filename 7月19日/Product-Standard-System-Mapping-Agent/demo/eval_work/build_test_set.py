from __future__ import annotations
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.engine.page_index_engine import PageIndexEngine
from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.rerank_adapter import RerankAdapter
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EngineType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("BuildTestSet")


@dataclass
class TestSetItem:
    product_name: str
    rag_result: str | None = None
    rag_rerank_result: str | None = None
    page_index_result: str | None = None
    page_index_force_result: str | None = None
    ground_truth: str | None = None
    ground_truth_source: str = ""
    rag_confidence: float = 0.0
    rag_rerank_confidence: float = 0.0
    page_index_confidence: float = 0.0
    page_index_force_confidence: float = 0.0


PROGRESS_FILE = "test_set_progress.txt"


def write_progress(completed: int, total: int, phase: str):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(f"{phase}|{completed}|{total}|{time.strftime('%H:%M:%S')}\n")


def init_engines(config_path: str = "config.yaml"):
    config = ConfigManager(config_path)
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    rerank_config = config.get_rerank_config()

    db = DBConnectionManager(db_config)
    db.initialize()

    llm = LLMAdapter(llm_config)
    trgm_mgr = TrgmIndexManager(db)
    vec_mgr = VectorIndexManager(
        db,
        embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )
    vec_mgr.ensure_pgvector_ready()
    vec_mgr.warmup()

    rag_engine = RAGMatchEngine(
        vec_mgr, trgm_mgr, llm, match_config,
        enable_llm=match_config.enable_llm,
        fine_match_mode="llm",
        engine_type=EngineType.RAG_VECTOR,
    )

    rerank_adapter = RerankAdapter(rerank_config) if rerank_config.api_key else None
    rag_rerank_engine = RAGMatchEngine(
        vec_mgr, trgm_mgr, llm, match_config,
        enable_llm=match_config.enable_rerank,
        rerank=rerank_adapter,
        fine_match_mode="rerank",
        engine_type=EngineType.RAG_RERANK,
    )

    reader = ExcelDataReader()
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")
    nodes, _ = reader.load_standard_system(standard_file)
    page_tree = PageIndexTree()
    page_tree.build_tree(nodes)

    page_engine = PageIndexEngine(page_tree, llm, force_llm_each_layer=False)
    page_engine_force = PageIndexEngine(page_tree, llm, force_llm_each_layer=True)

    return {
        "rag": rag_engine,
        "rag_rerank": rag_rerank_engine,
        "page_index": page_engine,
        "page_index_force": page_engine_force,
        "llm": llm,
        "db": db,
    }


def load_product_names(config_path: str = "config.yaml") -> list[str]:
    config = ConfigManager(config_path)
    product_file = config.get("data.company_product_file", "temp_company_product_0522_1.xlsx")
    reader = ExcelDataReader()
    products = reader.load_company_products(product_file)
    return list(dict.fromkeys(products))


def run_single_product(engines: dict, product_name: str) -> TestSetItem:
    item = TestSetItem(product_name=product_name)

    engine_map = {
        "rag": engines["rag"],
        "rag_rerank": engines["rag_rerank"],
        "page_index": engines["page_index"],
        "page_index_force": engines["page_index_force"],
    }

    for name, engine in engine_map.items():
        try:
            result = engine.match(product_name)
            cat_id = result.matched_category_id
            conf = result.confidence
            if name == "rag":
                item.rag_result = cat_id
                item.rag_confidence = conf
            elif name == "rag_rerank":
                item.rag_rerank_result = cat_id
                item.rag_rerank_confidence = conf
            elif name == "page_index":
                item.page_index_result = cat_id
                item.page_index_confidence = conf
            elif name == "page_index_force":
                item.page_index_force_result = cat_id
                item.page_index_force_confidence = conf
        except Exception as e:
            logger.warning(f"引擎 {name} 匹配失败: {product_name}, error={e}")

    return item


def determine_ground_truth(item: TestSetItem, engines: dict) -> TestSetItem:
    results = [
        item.rag_result,
        item.rag_rerank_result,
        item.page_index_result,
        item.page_index_force_result,
    ]
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        item.ground_truth = None
        item.ground_truth_source = "all_failed"
        return item

    counter = Counter(valid_results)
    most_common_id, most_common_count = counter.most_common(1)[0]

    if most_common_count >= 3:
        item.ground_truth = most_common_id
        item.ground_truth_source = f"voting_{most_common_count}/4"
        return item

    if most_common_count == 2 and len(counter) == 1:
        item.ground_truth = most_common_id
        item.ground_truth_source = "voting_2/2"
        return item

    if most_common_count == 2 and len(counter) >= 2:
        db = engines["db"]
        category_names = []
        for cat_id in counter.keys():
            row = db.execute_one(
                "SELECT category_name FROM category_vectors WHERE category_id = %s",
                (cat_id,),
            )
            if row:
                category_names.append(f"{row['category_name']}(id={cat_id})")
            else:
                category_names.append(cat_id)

        candidate_list = list(counter.keys())
        prompt = (
            f"产品\"{item.product_name}\"应该归属以下哪个标准分类？\n"
        )
        for i, (cat_id, count) in enumerate(counter.most_common()):
            name = category_names[i] if i < len(category_names) else cat_id
            prompt += f"{chr(65+i)}. {name}，得票{count}\n"
        prompt += (
            f"\n请只回复选项字母(A/B/C/D)，不要输出其他内容。"
        )

        try:
            from openai import OpenAI
            config = ConfigManager("config.yaml")
            llm_config = config.get_llm_config()
            client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
            for attempt in range(2):
                resp = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    timeout=30,
                )
                answer = resp.choices[0].message.content.strip().upper()
                for ch in answer:
                    if ch in "ABCD":
                        idx = ord(ch) - ord('A')
                        if idx < len(candidate_list):
                            item.ground_truth = candidate_list[idx]
                            item.ground_truth_source = "llm_arbitrate"
                            return item
            item.ground_truth = most_common_id
            item.ground_truth_source = "llm_arbitrate_fallback_top_vote"
        except Exception as e:
            logger.warning(f"LLM仲裁失败: {item.product_name}, error={e}")
            item.ground_truth = most_common_id
            item.ground_truth_source = "top_vote_fallback"

        return item

    item.ground_truth = most_common_id
    item.ground_truth_source = "top_vote_1"
    return item


def main():
    config_path = "config.yaml"
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_workers = 4
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("初始化引擎...")
    write_progress(0, sample_size, "init")
    engines = init_engines(config_path)

    logger.info("加载产品数据...")
    all_products = load_product_names(config_path)
    logger.info(f"产品总数: {len(all_products)}")

    if len(all_products) < sample_size:
        sample_size = len(all_products)
        logger.warning(f"产品数不足目标抽样数, 使用全部 {sample_size} 条")

    random.seed(42)
    sampled = random.sample(all_products, sample_size)
    logger.info(f"随机抽样 {sample_size} 条产品")

    logger.info("开始四引擎并行匹配...")
    items: list[TestSetItem] = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for name in sampled:
            future = executor.submit(run_single_product, engines, name)
            future_map[future] = name

        for future in as_completed(future_map):
            name = future_map[future]
            try:
                item = future.result()
                items.append(item)
            except Exception as e:
                logger.error(f"匹配异常: {name}, error={e}")
                items.append(TestSetItem(product_name=name))
            completed += 1
            write_progress(completed, sample_size, "matching")
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                speed = completed / elapsed
                eta = (sample_size - completed) / speed if speed > 0 else 0
                logger.info(f"匹配进度: {completed}/{sample_size} ({speed:.1f}条/s, ETA={eta:.0f}s)")

    elapsed = time.time() - start_time
    logger.info(f"匹配完成, 耗时={elapsed:.1f}s, 平均={elapsed/sample_size:.2f}s/条")

    logger.info("确定 ground truth...")
    for i, item in enumerate(items):
        determine_ground_truth(item, engines)
        if (i + 1) % 50 == 0:
            write_progress(i + 1, len(items), "ground_truth")
        if (i + 1) % 100 == 0:
            logger.info(f"ground truth 进度: {i+1}/{len(items)}")

    gt_counts = Counter(item.ground_truth_source for item in items)
    logger.info(f"ground truth 来源统计: {dict(gt_counts)}")

    output_path = os.path.join(output_dir, "test_set_200.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in items], f, ensure_ascii=False, indent=2)
    logger.info(f"测试集已保存: {output_path}")

    write_progress(sample_size, sample_size, "done")
    engines["db"].close()
    logger.info("完成!")


if __name__ == "__main__":
    main()