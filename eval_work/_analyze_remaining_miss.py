# -*- coding: utf-8 -*-
"""Analyze remaining misses after GT fix + soft scoring."""
from __future__ import annotations
import json
import os
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}

    out = {}
    for fname in ["eval_llm200_full_rescored.json", "eval_llmv2_full_rescored.json"]:
        d = json.load(open(f"output/{fname}", encoding="utf-8"))
        misses = [x for x in d["details"] if not x.get("correct_soft")]
        by_rel = Counter(x.get("relation") for x in misses)
        # finer buckets for explanation
        buckets = Counter()
        samples = {k: [] for k in [
            "no_match", "cross_domain", "same_top", "same_broad",
            "literal_synonym_branch", "pred_more_specific_elsewhere",
            "gt_vague_pred_concrete", "confusable_near_wrong",
        ]}

        for x in misses:
            pred = x.get("predicted")
            gt = str(x["ground_truth"])
            pname = x["product_name"]
            rel = x.get("relation") or "unrelated"
            pred_name = x.get("predicted_name") or (id2name.get(str(pred), "") if pred else "")
            gt_name = x.get("ground_truth_name") or id2name.get(gt, "")
            pg = id2group.get(str(pred), "") if pred else ""
            gg = id2group.get(gt, "")

            bucket = rel
            # refine
            if pred is None:
                bucket = "no_match"
            elif rel in ("cross_domain", "same_top"):
                # check if names are near synonyms
                if pname in pred_name or pred_name in pname:
                    bucket = "literal_synonym_branch"
                elif any(t in pred_name for t in [pname[:2], pname[-2:]] if len(pname) >= 2):
                    bucket = "pred_more_specific_elsewhere"
                else:
                    bucket = rel
            elif rel == "same_broad":
                bucket = "same_broad"

            # GT broader name contained?
            if pred and gt_name and len(gt_name) <= 4 and pname not in gt_name and pname in (pred_name or ""):
                bucket = "gt_vague_pred_concrete"

            buckets[bucket] += 1
            sample = {
                "product": pname,
                "gt": f"{gt_name}({gt})",
                "gt_group": gg,
                "pred": f"{pred_name}({pred})" if pred else None,
                "pred_group": pg,
                "conf": x.get("confidence"),
                "relation": rel,
                "bucket": bucket,
            }
            key = bucket if bucket in samples else rel
            if key not in samples:
                samples[key] = []
            if len(samples[key]) < 10:
                samples[key].append(sample)

        out[fname] = {
            "total": d["total"],
            "soft_acc": d["accuracy_soft"],
            "strict_acc": d["accuracy_strict"],
            "remain_miss": len(misses),
            "by_relation": dict(by_rel),
            "by_bucket": dict(buckets),
            "samples": {k: v for k, v in samples.items() if v},
            "high_conf_misses": sorted(
                [x for x in misses if (x.get("confidence") or 0) >= 0.85],
                key=lambda z: -(z.get("confidence") or 0),
            )[:20],
        }
        # normalize high_conf for readability
        out[fname]["high_conf_misses"] = [
            {
                "product": x["product_name"],
                "gt": f"{x.get('ground_truth_name')}({x['ground_truth']})",
                "pred": f"{x.get('predicted_name')}({x['predicted']})",
                "conf": x.get("confidence"),
                "relation": x.get("relation"),
            }
            for x in out[fname]["high_conf_misses"]
            if x.get("predicted") is not None
        ]

        print(fname, "miss", len(misses), "buckets", dict(buckets))

    with open("output/remaining_miss_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    db.close()
    print("wrote output/remaining_miss_analysis.json")


if __name__ == "__main__":
    main()
