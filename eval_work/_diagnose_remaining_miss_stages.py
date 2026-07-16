# -*- coding: utf-8 -*-
"""分析 llm_soft 仍未命中：GT 是否进粗召回、粗排名次、是否跳过精排。"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

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


def diagnose_one(engine: RAGMatchEngine, product: str, gt: str, id2name: dict) -> dict:
    # replicate coarse + shortcut logic without calling external rerank when possible
    cleaned = product  # preprocess inside match; call internals
    from src.engine.query_preprocessor import preprocess_query
    q = preprocess_query(product) or product.strip()

    exact = engine._trgm_mgr.lookup_exact_match(q)
    if exact and str(exact.category_id) == str(gt):
        return {"stage": "exact_hit_gt", "gt_in_coarse": True, "gt_rank": 1, "pred": str(exact.category_id)}
    if exact:
        return {
            "stage": "exact_wrong_shortcut",
            "gt_in_coarse": None,
            "gt_rank": None,
            "pred": str(exact.category_id),
            "pred_name": exact.category_name,
            "note": "精确/同名短路打到错误节点，未进入正常粗召回融合",
        }

    try:
        candidates = engine._coarse_recall(q)
    except Exception as e:
        return {"stage": "coarse_error", "error": str(e), "gt_in_coarse": False}

    if not candidates:
        return {"stage": "coarse_empty", "gt_in_coarse": False, "gt_rank": None}

    ids = [str(c.category_id) for c in candidates]
    gt_rank = ids.index(str(gt)) + 1 if str(gt) in ids else None
    best = candidates[0]

    # check skip fine / abort
    skip_fine = False
    skip_reason = ""
    if q == best.category_name:
        skip_fine, skip_reason = True, "top1_exact_name"
    else:
        syn_map = engine._load_syn_list(candidates)
        if engine._has_exact_synonym(q, best, syn_map):
            skip_fine, skip_reason = True, "top1_exact_synonym"

    ambiguous = False if skip_fine else engine._is_ambiguous_top(q, candidates)
    if not skip_fine and not ambiguous:
        if best.coarse_score >= 0.85:
            skip_fine, skip_reason = True, "coarse>=0.85"
        elif best.trgm_similarity >= 0.8 and best.coarse_score >= 0.7 and best.vector_similarity >= 0.5:
            skip_fine, skip_reason = True, "high_trgm_fusion"
        elif best.coarse_score >= 0.75 and best.vector_similarity >= 0.7:
            skip_fine, skip_reason = True, "high_vec_fusion"
        elif best.trgm_similarity >= 0.6 and best.coarse_score >= 0.65 and len(candidates) == 1:
            skip_fine, skip_reason = True, "single_candidate"
        elif q in best.category_name and best.vector_similarity >= 0.6:
            skip_fine, skip_reason = True, "product_in_name"
        elif best.category_name in q and best.vector_similarity >= 0.6:
            skip_fine, skip_reason = True, "name_in_product"
        elif (
            best.coarse_score >= 0.5
            and best.vector_similarity >= 0.6
            and len(candidates) >= 2
            and (candidates[0].coarse_score - candidates[1].coarse_score) >= 0.05
        ):
            skip_fine, skip_reason = True, "top1_margin"

    abort = False
    if not skip_fine:
        abort = engine._should_abort_coarse(q, candidates)

    # top candidates snapshot
    top = [
        {
            "rank": i + 1,
            "id": str(c.category_id),
            "name": c.category_name,
            "coarse": round(c.coarse_score, 4),
            "vec": round(c.vector_similarity, 4),
            "trgm": round(c.trgm_similarity, 4),
            "is_gt": str(c.category_id) == str(gt),
        }
        for i, c in enumerate(candidates[:10])
    ]

    if gt_rank is None:
        stage = "recall_miss"  # GT 未进 TopK
    elif skip_fine:
        stage = "skip_fine_wrong"  # 跳过精排，Top1 错
    elif abort:
        stage = "abort_no_match"  # 粗分过低放弃
    else:
        stage = "rerank_or_fusion_miss"  # 进了精排但最终没选中 GT（或被融合排错）

    return {
        "stage": stage,
        "gt_in_coarse": gt_rank is not None,
        "gt_rank": gt_rank,
        "coarse_top1_id": str(best.category_id),
        "coarse_top1_name": best.category_name,
        "coarse_top1_score": round(best.coarse_score, 4),
        "skip_fine": skip_fine,
        "skip_reason": skip_reason,
        "ambiguous_top": ambiguous,
        "abort": abort,
        "top10": top,
        "gt_name": id2name.get(str(gt), ""),
    }


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    embedding_config = config.get_embedding_config()

    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}

    llm = LLMAdapter(llm_config)
    trgm = TrgmIndexManager(db)
    vec = VectorIndexManager(
        db,
        embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        embedding_config=embedding_config,
    )
    vec.ensure_pgvector_ready()
    # 不 warmup 全量；embed_query 会现查
    engine = RAGMatchEngine(
        vec, trgm, llm, match_config,
        enable_llm=False,  # 只诊断粗召回/跳过逻辑，不触发 rerank API
        rerank=None,
        fine_match_mode="rerank",
        engine_type=EngineType.RAG_RERANK,
    )

    report = {}
    for fname in ["eval_llm200_full_llm_soft.json", "eval_llmv2_full_llm_soft.json"]:
        data = json.load(open(f"output/{fname}", encoding="utf-8"))
        misses = [x for x in data["details"] if not x.get("correct_llm_soft")]
        if limit:
            misses = misses[:limit]
        results = []
        stages = Counter()
        for i, row in enumerate(misses, 1):
            info = diagnose_one(engine, row["product_name"], str(row["ground_truth"]), id2name)
            info.update({
                "product": row["product_name"],
                "gt": str(row["ground_truth"]),
                "pred_eval": row.get("predicted"),
                "pred_name_eval": row.get("predicted_name"),
                "eval_relation": row.get("relation"),
                "llm_reason": row.get("llm_judge_reason"),
            })
            # 若评测有预测且 gt 在粗召回，精排阶段更细：
            if info["stage"] == "rerank_or_fusion_miss" and info.get("gt_rank"):
                # 说明粗召回有 GT，最终评测结果不是 GT → 精排或跳过导致
                if str(row.get("predicted")) == info.get("coarse_top1_id"):
                    # 最终=粗排Top1 → 可能跳过精排，或精排仍选 Top1
                    info["failure_hint"] = "final_equals_coarse_top1"
                else:
                    info["failure_hint"] = "rerank_changed_away_from_or_kept_wrong"
            stages[info["stage"]] += 1
            results.append(info)
            if i % 10 == 0:
                print(f"{fname} {i}/{len(misses)} stages={dict(stages)}", flush=True)

        # secondary classification using eval pred vs coarse
        for r in results:
            if r["stage"] != "rerank_or_fusion_miss":
                continue
            pred = str(r.get("pred_eval")) if r.get("pred_eval") is not None else None
            if pred and pred == r.get("coarse_top1_id"):
                # 最终与粗排一致：更像「跳过精排」或「精排确认了错误 Top1」
                # 用 skip_fine / abort 再拆
                if r.get("skip_fine"):
                    r["stage"] = "skip_fine_wrong"
                else:
                    r["stage"] = "rerank_kept_wrong_top1"
            elif r.get("gt_in_coarse") and pred and pred != r["gt"]:
                r["stage"] = "rerank_picked_non_gt"
            stages = Counter(x["stage"] for x in results)

        summary = {
            "file": fname,
            "miss_total": len(results),
            "stage_counts": dict(stages),
            "recall_hit_rate": round(
                sum(1 for x in results if x.get("gt_in_coarse")) / len(results), 4
            ) if results else 0,
            "cases": results,
        }
        report[fname] = summary
        print(fname, "stages", dict(stages), "gt_in_coarse", summary["recall_hit_rate"])

    out = "output/remaining_miss_stage_diagnosis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("wrote", out)
    db.close()


if __name__ == "__main__":
    main()
