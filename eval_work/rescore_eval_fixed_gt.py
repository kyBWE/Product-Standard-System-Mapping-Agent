# -*- coding: utf-8 -*-
"""用修正后的 GT + 已有预测重算严格/宽松准确率（不调 API）。"""
from __future__ import annotations
import json
import os
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.orchestration.eval_scoring import soft_match

ROOT = Path(__file__).resolve().parents[2]

PAIRS = [
    (
        ROOT / "test_set_llm_200.json",
        Path("output/eval_llm200_full.json"),
        Path("output/eval_llm200_full_rescored.json"),
    ),
    (
        ROOT / "test_set_llm_v2_200.json",
        Path("output/eval_llmv2_full.json"),
        Path("output/eval_llmv2_full_rescored.json"),
    ),
]


def rescore(ts_path: Path, eval_path: Path, id2name, id2group) -> dict:
    ts = {item["product_name"]: item for item in json.loads(ts_path.read_text(encoding="utf-8"))}
    old = json.loads(eval_path.read_text(encoding="utf-8"))

    strict = near = soft = no_match = 0
    details = []
    for row in old["details"]:
        pname = row["product_name"]
        item = ts.get(pname)
        if not item:
            # 产品名对不上时沿用旧 GT
            gt = str(row["ground_truth"])
            gt_name = row.get("ground_truth_name")
            gt_path = ""
        else:
            gt = str(item["ground_truth"])
            gt_name = item.get("ground_truth_name")
            gt_path = item.get("path") or ""

        pred = row["predicted"]
        if pred is not None:
            pred = str(pred)
        m = soft_match(pred, gt, id2name, id2group, gt_path)
        if pred is None:
            no_match += 1
        if m["strict"]:
            strict += 1
        if m["near"]:
            near += 1
        if m["soft"]:
            soft += 1

        details.append({
            **{k: row[k] for k in ("product_name", "predicted", "confidence", "time_ms", "error") if k in row},
            "ground_truth": gt,
            "ground_truth_name": gt_name,
            "predicted_name": id2name.get(pred, "") if pred else None,
            "correct_strict": m["strict"],
            "correct_near": m["near"],
            "correct_soft": m["soft"],
            "relation": m["relation"],
            "lca_depth": m["lca_depth"],
            "gt_fixed": bool(item and item.get("ground_truth_fix")),
        })

    total = len(details)
    summary = {
        "gt_source": ts_path.name,
        "embedding": old.get("embedding", "bge-m3-1024-full"),
        "engine": old.get("engine", "rag_rerank"),
        "note": "rescored_with_fixed_gt_and_soft_metrics",
        "total": total,
        "no_match": no_match,
        "accuracy_strict": round(strict / total, 4) if total else 0,
        "correct_strict": strict,
        "accuracy_near": round(near / total, 4) if total else 0,
        "correct_near": near,
        "accuracy_soft": round(soft / total, 4) if total else 0,
        "correct_soft": soft,
        "previous_accuracy": old.get("accuracy"),
        "details": details,
    }
    return summary


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}

    compare = {"embedding": "bge-m3-1024-full", "sets": []}
    for ts, ev, out in PAIRS:
        s = rescore(ts, ev, id2name, id2group)
        out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        brief = {
            "gt_source": s["gt_source"],
            "previous_accuracy": s["previous_accuracy"],
            "accuracy_strict": s["accuracy_strict"],
            "accuracy_near": s["accuracy_near"],
            "accuracy_soft": s["accuracy_soft"],
            "correct_strict": s["correct_strict"],
            "correct_near": s["correct_near"],
            "correct_soft": s["correct_soft"],
            "total": s["total"],
            "output": str(out),
        }
        compare["sets"].append(brief)
        print(
            f"{s['gt_source']}: prev={s['previous_accuracy']:.1%} "
            f"strict={s['accuracy_strict']:.1%} near={s['accuracy_near']:.1%} "
            f"soft={s['accuracy_soft']:.1%}"
        )

    cmp_path = Path("output/eval_full_compare_rescored.json")
    cmp_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"compare -> {cmp_path}")
    db.close()


if __name__ == "__main__":
    main()
