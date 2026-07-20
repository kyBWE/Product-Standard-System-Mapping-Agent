from __future__ import annotations
import json, sys, os, time, logging

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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("QuickEval")
logger.setLevel(logging.INFO)

TEST_SET = "test_set_random_200.json"
OUTPUT = "output/eval_random_200.json"

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = ConfigManager("config.yaml")
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    rerank_config = config.get_rerank_config()
    embedding_config = config.get_embedding_config()

    db = DBConnectionManager(db_config)
    db.initialize()

    llm = LLMAdapter(llm_config)
    trgm_mgr = TrgmIndexManager(db)
    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
        embedding_config=embedding_config,
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

    page_engine = PageIndexEngine(
        page_tree, llm, force_llm_each_layer=False,
        vec_mgr=vec_mgr, rerank=rerank_adapter, trgm_mgr=trgm_mgr
    )

    engines = {
        "rag_rerank": rag_rerank_engine,
        "page_index": page_engine,
    }

    with open(TEST_SET, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if limit > 0:
        test_set = test_set[:limit]

    total = len(test_set)
    logger.info(f"评测 {total} 条, 引擎: {list(engines.keys())}")

    results = {ek: {"correct": 0, "total": 0, "no_match": 0, "times": []} for ek in engines}
    details = []

    for idx, item in enumerate(test_set):
        gt = item["ground_truth"]
        pname = item["product_name"]
        row = {"product_name": pname, "ground_truth": gt, "engines": {}}

        for ek, engine in engines.items():
            t0 = time.time()
            try:
                result = engine.match(pname)
                elapsed = time.time() - t0
                predicted = result.matched_category_id
                conf = result.confidence
            except Exception as e:
                elapsed = time.time() - t0
                predicted = None
                conf = 0.0
                logger.warning(f"{ek} 异常: {pname}, {e}")

            hit = predicted == gt
            results[ek]["total"] += 1
            if predicted is None:
                results[ek]["no_match"] += 1
            elif hit:
                results[ek]["correct"] += 1
            results[ek]["times"].append(elapsed)

            row["engines"][ek] = {
                "predicted": predicted,
                "confidence": round(conf, 4),
                "time_ms": round(elapsed * 1000, 1),
                "hit": hit,
            }

        details.append(row)
        if (idx + 1) % 10 == 0 or idx == total - 1:
            acc = {ek: f"{r['correct']}/{r['total']}={r['correct']/r['total']:.1%}" for ek, r in results.items()}
            logger.info(f"[{idx+1}/{total}] {acc}")

    summary = {}
    for ek, r in results.items():
        t = r["total"]
        c = r["correct"]
        avg_t = sum(r["times"]) / len(r["times"]) if r["times"] else 0
        summary[ek] = {
            "accuracy": round(c / t, 4) if t else 0,
            "correct": c, "total": t,
            "no_match": r["no_match"],
            "avg_time_ms": round(avg_t * 1000, 1),
        }

    output = {
        "gt_source": "test_set_llm_v2_200.json",
        "embedding": "bge-m3-1024-full",
        "summary": summary,
        "total_items": total,
        "details": details,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    for ek, s in summary.items():
        logger.info(f"{ek}: {s['accuracy']:.1%} ({s['correct']}/{s['total']}) avg={s['avg_time_ms']:.0f}ms no_match={s['no_match']}")

    db.close()

if __name__ == "__main__":
    main()