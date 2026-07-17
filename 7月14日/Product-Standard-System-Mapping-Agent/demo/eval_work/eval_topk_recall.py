# -*- coding: utf-8 -*-
"""扫描不同 Top-K：粗召回 Recall@K 与强制 rerank 后 Accuracy@K。

对每条样本只做一次 coarse@max_k 与一次 rerank（top_n=max_k），再切片得各 K。
exact 同名快路径计为各 K 召回/命中。
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.rerank_adapter import RerankAdapter
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EngineType

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("EvalTopK")
logger.setLevel(logging.INFO)

KS = [5, 10, 20, 50]
MAX_K = max(KS)

TEST_SETS = [
    ("../../test_set_llm_200_refined.json", "llm200_refined"),
    ("../../test_set_llm_v2_200.json", "llmv2"),
]


def _pick_best(candidates, scores_map, k: int) -> str | None:
    pool = candidates[:k]
    if not pool:
        return None
    best = max(pool, key=lambda c: (scores_map.get(c.category_id, 0.0), c.coarse_score))
    return str(best.category_id)


def eval_set(engine: RAGMatchEngine, rerank: RerankAdapter, items: list, label: str) -> dict:
    totals = {k: {
        "recall": 0, "coarse_top1": 0, "rerank_acc": 0, "n": 0,
    } for k in KS}
    extras = {"exact": 0, "no_coarse": 0, "total": 0}
    details = []
    t_all = time.time()

    for idx, item in enumerate(items):
        pname = item["product_name"]
        gt = str(item["ground_truth"])
        extras["total"] += 1
        row = {"product_name": pname, "ground_truth": gt, "exact": False}

        exact = engine._trgm_mgr.lookup_exact_match(pname)
        if exact:
            extras["exact"] += 1
            row["exact"] = True
            eid = str(exact.category_id)
            row["coarse_ids"] = [eid]
            row["coarse_top1"] = eid
            row["rerank_pred"] = {str(k): eid for k in KS}
            hit = eid == gt
            for k in KS:
                totals[k]["n"] += 1
                if hit:
                    totals[k]["recall"] += 1
                    totals[k]["coarse_top1"] += 1
                    totals[k]["rerank_acc"] += 1
            details.append(row)
        else:
            engine._config.coarse_top_k = MAX_K
            try:
                candidates = engine._coarse_recall(pname)
            except Exception as e:
                logger.warning(f"粗召回失败 {pname}: {e}")
                candidates = []
            if not candidates:
                extras["no_coarse"] += 1
                row["coarse_ids"] = []
                row["coarse_top1"] = None
                row["rerank_pred"] = {str(k): None for k in KS}
                for k in KS:
                    totals[k]["n"] += 1
                details.append(row)
            else:
                ids = [str(c.category_id) for c in candidates]
                row["coarse_ids"] = ids
                row["coarse_top1"] = ids[0]
                # force rerank on full MAX_K list
                engine._rerank = rerank
                # temporarily raise top_n via object
                old_top = rerank._config.top_n
                rerank._config.top_n = MAX_K
                try:
                    scored = engine._fine_match_rerank(pname, candidates[:MAX_K])
                except Exception as e:
                    logger.warning(f"rerank失败 {pname}: {e}")
                    scored = []
                finally:
                    rerank._config.top_n = old_top

                score_map = {str(s.category_id): float(s.final_confidence) for s in scored}
                # also keep pure rerank score for tie-break already in final_confidence
                row["rerank_pred"] = {}
                for k in KS:
                    totals[k]["n"] += 1
                    pool_ids = ids[:k]
                    if gt in pool_ids:
                        totals[k]["recall"] += 1
                    if ids[0] == gt:
                        totals[k]["coarse_top1"] += 1
                    pred = _pick_best(candidates, score_map, k)
                    row["rerank_pred"][str(k)] = pred
                    if pred == gt:
                        totals[k]["rerank_acc"] += 1
                details.append(row)

        if (idx + 1) % 10 == 0 or idx == len(items) - 1:
            r50 = totals[50]["recall"] / totals[50]["n"] if totals[50]["n"] else 0
            a50 = totals[50]["rerank_acc"] / totals[50]["n"] if totals[50]["n"] else 0
            logger.info(
                f"[{label} {idx+1}/{len(items)}] Recall@50={r50:.1%} "
                f"RerankAcc@50={a50:.1%} exact={extras['exact']}"
            )

    summary_ks = {}
    for k in KS:
        n = totals[k]["n"] or 1
        summary_ks[str(k)] = {
            "recall": round(totals[k]["recall"] / n, 4),
            "recall_count": totals[k]["recall"],
            "coarse_top1_acc": round(totals[k]["coarse_top1"] / n, 4),
            "coarse_top1_count": totals[k]["coarse_top1"],
            "rerank_acc": round(totals[k]["rerank_acc"] / n, 4),
            "rerank_acc_count": totals[k]["rerank_acc"],
            "total": totals[k]["n"],
        }

    return {
        "label": label,
        "embedding": "bge-m3-1024-full",
        "ks": KS,
        "note": (
            "strict ID only; coarse once@50; forced rerank once with top_n=50; "
            "Accuracy@K = argmax final_conf among coarse[:K]; exact shortcut counted as hit"
        ),
        "extras": extras,
        "elapsed_sec": round(time.time() - t_all, 1),
        "by_k": summary_ks,
        "details": details,
    }


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    llm = config.get_llm_config()
    match = config.get_match_config()
    emb = config.get_embedding_config()
    rr_cfg = config.get_rerank_config()

    dim_rows = db.execute(
        "SELECT vector_dims(vec_search) AS dim, COUNT(*) AS n "
        "FROM category_vectors WHERE vec_search IS NOT NULL GROUP BY 1"
    )
    logger.info(f"索引: {dim_rows}")
    if not dim_rows or dim_rows[0]["dim"] != 1024 or dim_rows[0]["n"] < 20000:
        raise RuntimeError(f"需要 full 1024 索引: {dim_rows}")

    match.coarse_top_k = MAX_K
    trgm = TrgmIndexManager(db)
    vec = VectorIndexManager(
        db, embedding_model=llm.embedding_model,
        embedding_dimension=llm.embedding_dimension,
        base_url=llm.base_url, api_key=llm.api_key, embedding_config=emb,
    )
    vec.ensure_pgvector_ready()
    vec.warmup()
    rerank = RerankAdapter(rr_cfg)
    rerank._config.top_n = MAX_K
    engine = RAGMatchEngine(
        vec, trgm, None, match,
        enable_llm=True, rerank=rerank,
        fine_match_mode="rerank", engine_type=EngineType.RAG_RERANK,
    )

    all_out = {"embedding": "bge-m3-1024-full", "ks": KS, "sets": []}
    Path("output").mkdir(exist_ok=True)

    for path, label in TEST_SETS:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if limit > 0:
            items = items[:limit]
        logger.info(f"==== {label} n={len(items)} ====")
        result = eval_set(engine, rerank, items, label)
        result["gt_source"] = Path(path).name
        out_path = Path(f"output/eval_topk_{label}.json")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        brief = {"gt_source": result["gt_source"], "by_k": result["by_k"],
                 "extras": result["extras"], "elapsed_sec": result["elapsed_sec"]}
        all_out["sets"].append(brief)
        logger.info(f"写入 {out_path}")
        for k, v in result["by_k"].items():
            logger.info(
                f"  K={k}: Recall={v['recall']:.1%} "
                f"CoarseTop1={v['coarse_top1_acc']:.1%} "
                f"RerankAcc={v['rerank_acc']:.1%}"
            )

    cmp = Path("output/eval_topk_compare.json")
    cmp.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"对比摘要 -> {cmp}")
    db.close()


if __name__ == "__main__":
    main()
