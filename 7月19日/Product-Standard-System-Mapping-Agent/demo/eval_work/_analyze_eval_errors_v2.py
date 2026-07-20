# -*- coding: utf-8 -*-
"""Clean UTF-8 analysis of wrong mappings vs GT."""
from __future__ import annotations
import json
import os
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager


def split_group(g: str) -> list[str]:
    if not g:
        return []
    # paths use " > ", groups use ","
    if " > " in g:
        return [x.strip() for x in g.split(" > ") if x.strip()]
    return [x.strip() for x in g.split(",") if x.strip()]


def lca(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}
    name_nodes: dict[str, list[str]] = {}
    for r in rows:
        name_nodes.setdefault(r["category_name"], []).append(str(r["category_id"]))

    out = {"sets": []}

    for fname, tsname in [
        ("eval_llm200_full.json", "../../test_set_llm_200.json"),
        ("eval_llmv2_full.json", "../../test_set_llm_v2_200.json"),
    ]:
        d = json.load(open(f"output/{fname}", encoding="utf-8"))
        ts_items = json.load(open(tsname, encoding="utf-8"))
        ts = {item["product_name"]: item for item in ts_items}

        wrong_rows = []
        cats = Counter()
        for x in d["details"]:
            if x["correct"]:
                continue
            prod = x["product_name"]
            gt = str(x["ground_truth"])
            pred = str(x["predicted"]) if x["predicted"] is not None else None
            gt_name = x.get("ground_truth_name") or id2name.get(gt, "")
            pred_name = id2name.get(pred, "") if pred else ""
            gt_path = split_group(ts.get(prod, {}).get("path", ""))
            if not gt_path:
                gt_path = split_group(id2group.get(gt, "")) + ([gt_name] if gt_name else [])
            pred_path = split_group(id2group.get(pred, "")) + ([pred_name] if pred_name else []) if pred else []

            depth = lca(gt_path, pred_path) if pred else 0
            exact_ids = name_nodes.get(prod, [])

            # judgment labels
            labels = []
            if pred is None:
                labels.append("no_match")
            else:
                if pred_name == prod or prod == pred_name:
                    labels.append("pred_exact_name_match")
                elif prod in pred_name:
                    labels.append("pred_contains_product")
                if exact_ids and pred in exact_ids:
                    labels.append("pred_is_literal_category")
                if exact_ids and gt not in exact_ids:
                    labels.append("gt_not_literal_but_literal_exists")
                if depth >= max(len(gt_path), len(pred_path), 1) - 1 and depth >= 2:
                    labels.append("sibling_near")
                elif depth >= 3:
                    labels.append("same_subtree")
                elif depth >= 2:
                    labels.append("same_broad")
                elif depth >= 1:
                    labels.append("same_top_only")
                else:
                    labels.append("cross_domain")

                # heuristic: GT looks wrong if literal category exists and pred is that literal
                if "pred_is_literal_category" in labels and "gt_not_literal_but_literal_exists" in labels:
                    labels.append("LIKELY_GT_WEAKER_THAN_PRED")
                # pred more specific child of GT?
                if gt_name and pred_name and gt_name in " > ".join(pred_path) and pred_name != gt_name:
                    labels.append("pred_deeper_under_gt_path")
                if pred_name and gt_name and pred_name in " > ".join(gt_path) and pred_name != gt_name:
                    labels.append("pred_is_ancestor_of_gt")

            primary = (
                "LIKELY_GT_ISSUE" if "LIKELY_GT_WEAKER_THAN_PRED" in labels
                else "NEAR_MISS" if ("sibling_near" in labels or "same_subtree" in labels)
                else "BROAD_RELATED" if "same_broad" in labels
                else "FAR_WRONG" if "cross_domain" in labels or "same_top_only" in labels
                else "NO_MATCH" if "no_match" in labels
                else "OTHER"
            )
            cats[primary] += 1

            wrong_rows.append({
                "product": prod,
                "gt_id": gt,
                "gt_name": gt_name,
                "gt_path": " > ".join(gt_path),
                "pred_id": pred,
                "pred_name": pred_name,
                "pred_path": " > ".join(pred_path),
                "lca_depth": depth,
                "confidence": round(x["confidence"], 3),
                "labels": labels,
                "verdict": primary,
            })

        # summary stats
        n_wrong = len(wrong_rows)
        n_right = d["correct"]
        likely_gt = sum(1 for r in wrong_rows if r["verdict"] == "LIKELY_GT_ISSUE")
        near = sum(1 for r in wrong_rows if r["verdict"] == "NEAR_MISS")
        far = sum(1 for r in wrong_rows if r["verdict"] == "FAR_WRONG")

        # If we count LIKELY_GT_ISSUE as "reasonable pred", adjusted acc
        adj_correct = n_right + likely_gt
        adj_acc = adj_correct / d["total"] if d["total"] else 0
        soft_correct = n_right + likely_gt + near
        soft_acc = soft_correct / d["total"] if d["total"] else 0

        # pick representative samples
        by_v = {k: [] for k in ["LIKELY_GT_ISSUE", "NEAR_MISS", "BROAD_RELATED", "FAR_WRONG"]}
        for r in wrong_rows:
            if r["verdict"] in by_v and len(by_v[r["verdict"]]) < 12:
                by_v[r["verdict"]].append(r)

        set_report = {
            "file": fname,
            "accuracy": d["accuracy"],
            "correct": d["correct"],
            "total": d["total"],
            "wrong": n_wrong,
            "verdict_counts": dict(cats),
            "adjusted_acc_if_literal_gt_accepted": round(adj_acc, 4),
            "soft_acc_plus_near_miss": round(soft_acc, 4),
            "samples": by_v,
            "high_conf_far_wrong": [
                r for r in wrong_rows
                if r["verdict"] == "FAR_WRONG" and r["confidence"] >= 0.8
            ][:15],
        }
        out["sets"].append(set_report)
        print(fname)
        print("  official acc", d["accuracy"])
        print("  verdicts", dict(cats))
        print("  adj (literal OK)", round(adj_acc, 3), "soft (+near)", round(soft_acc, 3))

    with open("output/eval_error_analysis_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote output/eval_error_analysis_v2.json")
    db.close()


if __name__ == "__main__":
    main()
