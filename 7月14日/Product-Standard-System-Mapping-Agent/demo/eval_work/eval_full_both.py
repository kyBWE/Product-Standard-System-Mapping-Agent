# -*- coding: utf-8 -*-
"""在 bge-m3-1024-full 索引上评测两个 LLM 测试集（rag_rerank）。"""
from __future__ import annotations
import json, os, sys, time, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.engine.llm_adapter import LLMAdapter
from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.rerank_adapter import RerankAdapter
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EngineType
from src.orchestration.eval_scoring import soft_match

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("EvalFullBoth")
logger.setLevel(logging.INFO)

TEST_SETS = [
    ("../../test_set_llm_200.json", "output/eval_llm200_full.json"),
    ("../../test_set_llm_v2_200.json", "output/eval_llmv2_full.json"),
]


def eval_one(engine, test_set, gt_source: str, out_path: str,
             id2name: dict[str, str], id2group: dict[str, str]):
    results = {
        "correct": 0, "correct_near": 0, "correct_soft": 0,
        "total": 0, "no_match": 0, "times": [], "confs": [],
    }
    details = []
    total = len(test_set)
    logger.info(f"开始评测 {gt_source}: {total} 条")

    for idx, item in enumerate(test_set):
        gt = str(item["ground_truth"])
        pname = item["product_name"]
        t0 = time.time()
        predicted = None
        conf = 0.0
        err = None
        try:
            result = engine.match(pname)
            predicted = result.matched_category_id
            if predicted is not None:
                predicted = str(predicted)
            conf = float(result.confidence or 0.0)
        except Exception as e:
            err = str(e)
            logger.warning(f"异常: {pname}: {e}")
        elapsed = time.time() - t0

        m = soft_match(predicted, gt, id2name, id2group, item.get("path") or "")
        results["total"] += 1
        results["times"].append(elapsed)
        results["confs"].append(conf)
        if predicted is None:
            results["no_match"] += 1
        if m["strict"]:
            results["correct"] += 1
        if m["near"]:
            results["correct_near"] += 1
        if m["soft"]:
            results["correct_soft"] += 1

        details.append({
            "product_name": pname,
            "predicted": predicted,
            "predicted_name": id2name.get(predicted, "") if predicted else None,
            "ground_truth": gt,
            "ground_truth_name": item.get("ground_truth_name"),
            "confidence": round(conf, 4),
            "correct": m["strict"],
            "correct_near": m["near"],
            "correct_soft": m["soft"],
            "relation": m["relation"],
            "lca_depth": m["lca_depth"],
            "time_ms": round(elapsed * 1000, 1),
            "error": err,
        })

        if (idx + 1) % 10 == 0 or idx == total - 1:
            acc = results["correct"] / results["total"] if results["total"] else 0
            soft_acc = results["correct_soft"] / results["total"] if results["total"] else 0
            logger.info(
                f"[{idx+1}/{total}] strict={acc:.1%} soft={soft_acc:.1%} "
                f"({results['correct']}/{results['total']}) no_match={results['no_match']}"
            )

    times = sorted(results["times"])
    p95 = times[int(len(times) * 0.95)] if times else 0
    n = results["total"] or 1
    summary = {
        "gt_source": os.path.basename(gt_source.replace("\\", "/")),
        "embedding": "bge-m3-1024-full",
        "engine": "rag_rerank",
        "accuracy": round(results["correct"] / n, 4),
        "accuracy_near": round(results["correct_near"] / n, 4),
        "accuracy_soft": round(results["correct_soft"] / n, 4),
        "correct": results["correct"],
        "correct_near": results["correct_near"],
        "correct_soft": results["correct_soft"],
        "total": results["total"],
        "no_match": results["no_match"],
        "avg_confidence": round(sum(results["confs"]) / len(results["confs"]), 4) if results["confs"] else 0,
        "avg_time_ms": round(sum(results["times"]) / len(results["times"]) * 1000, 1) if results["times"] else 0,
        "p95_time_ms": round(p95 * 1000, 1),
        "details": details,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"完成 {summary['gt_source']}: strict={summary['accuracy']:.1%} "
        f"near={summary['accuracy_near']:.1%} soft={summary['accuracy_soft']:.1%} "
        f"({summary['correct']}/{summary['total']}) -> {out_path}"
    )
    return summary


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    rerank_config = config.get_rerank_config()
    embedding_config = config.get_embedding_config()

    # 校验 full 索引
    dim_rows = db.execute(
        "SELECT vector_dims(vec_search) AS dim, COUNT(*) AS n "
        "FROM category_vectors WHERE vec_search IS NOT NULL GROUP BY 1"
    )
    logger.info(f"当前向量索引: {dim_rows}")
    if not dim_rows or dim_rows[0]["dim"] != 1024 or dim_rows[0]["n"] < 20000:
        raise RuntimeError(f"索引不是 bge-m3-1024-full: {dim_rows}")

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

    text_rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in text_rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in text_rows}

    rerank_adapter = RerankAdapter(rerank_config) if rerank_config.api_key else None
    engine = RAGMatchEngine(
        vec_mgr, trgm_mgr, llm, match_config,
        enable_llm=match_config.enable_rerank,
        rerank=rerank_adapter,
        fine_match_mode="rerank",
        engine_type=EngineType.RAG_RERANK,
    )

    all_summaries = []
    for in_path, out_path in TEST_SETS:
        with open(in_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        if limit > 0:
            test_set = test_set[:limit]
        s = eval_one(engine, test_set, in_path, out_path, id2name, id2group)
        all_summaries.append({
            "gt_source": s["gt_source"],
            "accuracy": s["accuracy"],
            "accuracy_near": s["accuracy_near"],
            "accuracy_soft": s["accuracy_soft"],
            "correct": s["correct"],
            "correct_near": s["correct_near"],
            "correct_soft": s["correct_soft"],
            "total": s["total"],
            "no_match": s["no_match"],
            "avg_confidence": s["avg_confidence"],
            "avg_time_ms": s["avg_time_ms"],
            "output": out_path,
        })

    cmp_path = "output/eval_full_compare.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": "bge-m3-1024-full", "sets": all_summaries}, f, ensure_ascii=False, indent=2)
    logger.info(f"对比摘要 -> {cmp_path}")
    for s in all_summaries:
        logger.info(
            f"  {s['gt_source']}: strict={s['accuracy']:.1%} "
            f"near={s['accuracy_near']:.1%} soft={s['accuracy_soft']:.1%} "
            f"({s['correct']}/{s['total']})"
        )

    db.close()


if __name__ == "__main__":
    main()
