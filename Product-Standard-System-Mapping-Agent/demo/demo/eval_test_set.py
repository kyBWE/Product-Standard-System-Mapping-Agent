from __future__ import annotations
import json
import logging
import os
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
logger = logging.getLogger("EvalTestSet")

TEST_SET_PATH = "output/test_set_1000_fixed.json"
OUTPUT_PATH = "output/eval_results.json"
MAX_WORKERS = 4
PROGRESS_FILE = "output/eval_progress.txt"


ENGINE_KEYS = ["rag", "rag_rerank", "page_index", "page_index_force"]
RESULT_KEYS = ["rag_result", "rag_rerank_result", "page_index_result", "page_index_force_result"]
CONFIDENCE_KEYS = ["rag_confidence", "rag_rerank_confidence", "page_index_confidence", "page_index_force_confidence"]


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
        "db": db,
    }


def evaluate(limit=0):
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if limit > 0:
        test_set = test_set[:limit]

    logger.info(f"加载测试集: {len(test_set)} 条")

    logger.info("初始化引擎...")
    engines = init_engines()

    engine_map = {
        "rag": engines["rag"],
        "rag_rerank": engines["rag_rerank"],
        "page_index": engines["page_index"],
        "page_index_force": engines["page_index_force"],
    }

    results_per_engine = {k: {"correct": 0, "total": 0, "no_match": 0, "confidences": [], "times": []} for k in ENGINE_KEYS}

    total = len(test_set)
    completed = 0
    start_time = time.time()
    progress_fh = open(PROGRESS_FILE, "w", encoding="utf-8")

    def write_progress(msg):
        progress_fh.write(msg + "\n")
        progress_fh.flush()

    write_progress(f"评测开始: 共{total}条, 并发={MAX_WORKERS}")
    write_progress("=" * 70)

    def process_item(item):
        gt = item["ground_truth"]
        row_results = {}
        for ek, engine in engine_map.items():
            t0 = time.time()
            try:
                result = engine.match(item["product_name"])
                elapsed = time.time() - t0
                predicted = result.matched_category_id
                conf = result.confidence
            except Exception as e:
                elapsed = time.time() - t0
                predicted = None
                conf = 0.0
                logger.warning(f"引擎 {ek} 匹配异常: {item['product_name']}, error={e}")
            row_results[ek] = {"predicted": predicted, "confidence": conf, "time": elapsed, "ground_truth": gt}
        return row_results

    logger.info("开始评测...")
    all_row_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for item in test_set:
            future = executor.submit(process_item, item)
            future_map[future] = item

        for future in as_completed(future_map):
            row_results = future.result()
            all_row_results.append(row_results)
            completed += 1
            elapsed = time.time() - start_time
            speed = completed / elapsed
            eta = (total - completed) / speed if speed > 0 else 0
            pct = completed / total * 100

            item = future_map[future]
            pname = item["product_name"][:25]
            line_parts = [f"[{completed}/{total} {pct:.1f}%] {pname}"]
            for ek in ENGINE_KEYS:
                r = row_results[ek]
                hit = "✓" if r["predicted"] == r["ground_truth"] else "✗" if r["predicted"] else "-"
                line_parts.append(f"  {ek}: {r['predicted'] or 'NONE':<8} conf={r['confidence']:.3f} {hit}")
            line_parts.append(f"  速度={speed:.1f}条/s 剩余={eta:.0f}s")
            write_progress("\n".join(line_parts))

    elapsed = time.time() - start_time
    logger.info(f"评测完成, 耗时={elapsed:.1f}s")

    for row in all_row_results:
        for ek in ENGINE_KEYS:
            r = row[ek]
            results_per_engine[ek]["total"] += 1
            if r["predicted"] is None:
                results_per_engine[ek]["no_match"] += 1
            elif r["predicted"] == r["ground_truth"]:
                results_per_engine[ek]["correct"] += 1
            results_per_engine[ek]["confidences"].append(r["confidence"])
            results_per_engine[ek]["times"].append(r["time"])

    summary = {}
    for ek in ENGINE_KEYS:
        r = results_per_engine[ek]
        total_items = r["total"]
        correct = r["correct"]
        no_match = r["no_match"]
        confs = r["confidences"]
        times = r["times"]
        accuracy = correct / total_items if total_items > 0 else 0
        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_time = sum(times) / len(times) if times else 0
        p95_time = sorted(times)[int(len(times) * 0.95)] if times else 0
        summary[ek] = {
            "total": total_items,
            "correct": correct,
            "no_match": no_match,
            "accuracy": round(accuracy, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_time_ms": round(avg_time * 1000, 1),
            "p95_time_ms": round(p95_time * 1000, 1),
        }

    logger.info("=" * 60)
    logger.info("评测结果汇总:")
    logger.info(f"{'引擎':<20} {'准确率':<10} {'平均置信度':<12} {'平均耗时ms':<12} {'P95耗时ms':<12} {'无匹配数':<10}")
    for ek in ENGINE_KEYS:
        s = summary[ek]
        logger.info(f"{ek:<20} {s['accuracy']:<10.4f} {s['avg_confidence']:<12.4f} {s['avg_time_ms']:<12.1f} {s['p95_time_ms']:<12.1f} {s['no_match']:<10}")

    write_progress("=" * 70)
    write_progress(f"评测完成! 耗时={elapsed:.1f}s")
    write_progress(f"{'引擎':<20} {'准确率':<10} {'平均置信度':<12} {'平均耗时ms':<12} {'P95耗时ms':<12} {'无匹配数':<10}")
    for ek in ENGINE_KEYS:
        s = summary[ek]
        write_progress(f"{ek:<20} {s['accuracy']:<10.4f} {s['avg_confidence']:<12.4f} {s['avg_time_ms']:<12.1f} {s['p95_time_ms']:<12.1f} {s['no_match']:<10}")
    progress_fh.close()

    output = {
        "summary": summary,
        "total_items": total,
        "eval_time": round(elapsed, 1),
        "test_set_path": TEST_SET_PATH,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"评测结果已保存: {OUTPUT_PATH}")

    engines["db"].close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    evaluate(limit=n)