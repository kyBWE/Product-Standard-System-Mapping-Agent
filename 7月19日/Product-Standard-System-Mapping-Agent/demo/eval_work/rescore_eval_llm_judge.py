# -*- coding: utf-8 -*-
"""对 soft 未命中样本做大模型合理性兜底，合理则计为 llm_soft 命中。"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.orchestration.eval_scoring import format_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLMJudgeRescore")

INPUTS = [
    Path("output/eval_llm200_full_rescored.json"),
    Path("output/eval_llmv2_full_rescored.json"),
]


def judge_file(
    path: Path,
    llm: LLMAdapter,
    id2name: dict[str, str],
    id2group: dict[str, str],
    limit: int = 0,
    batch_size: int = 10,
) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    details = data["details"]
    need = [i for i, d in enumerate(details)
            if not d.get("correct_soft") and d.get("predicted") is not None]
    if limit > 0:
        need = need[:limit]
    logger.info(f"{path.name}: soft未命中待判定 {len(need)} 条, batch_size={batch_size}")

    ok_count = 0
    t0 = time.time()

    for batch_start in range(0, len(need), batch_size):
        batch_indices = need[batch_start:batch_start + batch_size]
        batch_items = []
        for idx in batch_indices:
            row = details[idx]
            pred = str(row["predicted"])
            gt = str(row["ground_truth"])
            pred_name = row.get("predicted_name") or id2name.get(pred, "")
            gt_name = row.get("ground_truth_name") or id2name.get(gt, "")
            pred_path = format_path(pred, id2name, id2group)
            gt_path = format_path(gt, id2name, id2group)
            batch_items.append({
                "product_name": row["product_name"],
                "predicted_name": pred_name,
                "predicted_path": pred_path,
                "gt_name": gt_name,
                "gt_path": gt_path,
            })

        results = llm.batch_judge_mapping_reasonable(batch_items)

        for j, idx in enumerate(batch_indices):
            reasonable, conf, reason = results[j]
            row = details[idx]
            row["llm_reasonable"] = reasonable
            row["llm_judge_confidence"] = round(conf, 4)
            row["llm_judge_reason"] = reason
            row["correct_llm_soft"] = bool(row.get("correct_soft") or reasonable)
            if reasonable:
                ok_count += 1

        done = min(batch_start + batch_size, len(need))
        logger.info(
            f"  [{done}/{len(need)}] llm_accept={ok_count} "
            f"batch={batch_start + 1}-{done}"
        )

    for row in details:
        if row.get("correct_soft"):
            row["correct_llm_soft"] = True
            if "llm_reasonable" not in row:
                row["llm_reasonable"] = None
        elif row.get("predicted") is None:
            row["correct_llm_soft"] = False
            row["llm_reasonable"] = False
            row["llm_judge_reason"] = row.get("llm_judge_reason") or "no_match"
        elif "correct_llm_soft" not in row:
            row["correct_llm_soft"] = False

    total = len(details) or 1
    llm_soft = sum(1 for d in details if d.get("correct_llm_soft"))
    judged_accept = sum(1 for d in details if d.get("llm_reasonable") is True)
    summary = {
        **{k: v for k, v in data.items() if k != "details"},
        "note": (data.get("note") or "") + "+llm_reasonableness_judge",
        "accuracy_llm_soft": round(llm_soft / total, 4),
        "correct_llm_soft": llm_soft,
        "llm_judged": len(need),
        "llm_accepted": judged_accept,
        "llm_judge_seconds": round(time.time() - t0, 1),
        "details": details,
    }
    out = path.with_name(path.stem.replace("_rescored", "") + "_llm_soft.json")
    if "llm200" in path.name:
        out = Path("output/eval_llm200_full_llm_soft_glm51.json")
    elif "llmv2" in path.name:
        out = Path("output/eval_llmv2_full_llm_soft_glm51.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"完成 {path.name}: soft={data.get('accuracy_soft'):.1%} "
        f"llm_soft={summary['accuracy_llm_soft']:.1%} "
        f"(+{judged_accept} by LLM) -> {out}"
    )
    return {
        "gt_source": data.get("gt_source"),
        "accuracy_strict": data.get("accuracy_strict"),
        "accuracy_soft": data.get("accuracy_soft"),
        "accuracy_llm_soft": summary["accuracy_llm_soft"],
        "correct_llm_soft": llm_soft,
        "total": len(details),
        "llm_judged": len(need),
        "llm_accepted": judged_accept,
        "output": str(out),
    }


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}
    llm = LLMAdapter(config.get_llm_config())

    briefs = []
    for p in INPUTS:
        if not p.exists():
            logger.warning(f"missing {p}")
            continue
        briefs.append(judge_file(p, llm, id2name, id2group, limit=limit))

    cmp = Path("output/eval_full_compare_llm_soft_glm51.json")
    cmp.write_text(
        json.dumps({"embedding": "bge-m3-1024-full", "sets": briefs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"对比摘要 -> {cmp}")
    for b in briefs:
        logger.info(
            f"  {b['gt_source']}: strict={b['accuracy_strict']:.1%} "
            f"soft={b['accuracy_soft']:.1%} llm_soft={b['accuracy_llm_soft']:.1%} "
            f"(LLM采纳 {b['llm_accepted']}/{b['llm_judged']})"
        )
    db.close()


if __name__ == "__main__":
    main()
