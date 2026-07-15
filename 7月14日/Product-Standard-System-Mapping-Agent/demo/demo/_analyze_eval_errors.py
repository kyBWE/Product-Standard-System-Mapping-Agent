# -*- coding: utf-8 -*-
"""Analyze eval wrong mappings vs GT."""
from __future__ import annotations
import json
import os
import re
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager


def path_parts(p: str) -> list[str]:
    if not p:
        return []
    return [x.strip() for x in re.split(r"\s*>\s*", p) if x.strip()]


def lca_depth(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def classify(prod: str, gt_name: str, pred_name: str, gt_path: str, pred_path: str) -> str:
    gp, pp = path_parts(gt_path), path_parts(pred_path)
    if not pred_name:
        return "no_match"
    # exact name contain
    if prod in pred_name or pred_name in prod:
        return "pred_name_exactish"
    if gp and pp:
        depth = lca_depth(gp, pp)
        if depth >= max(len(gp), len(pp)) - 1 and abs(len(gp) - len(pp)) <= 1:
            return "sibling_or_near"  # same parent or adjacent
        if depth >= 3:
            return "same_subtree_depth>=3"
        if depth >= 2:
            return "same_broad_category"
        if depth >= 1:
            return "same_top_domain_only"
        return "cross_domain"
    # fallback name overlap
    s1, s2 = set(gt_name), set(pred_name)
    ov = len(s1 & s2) / max(len(s1 | s2), 1)
    if ov >= 0.4:
        return "name_similar_no_path"
    return "unclear"


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()

    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}

    # build path from page_index / treenode if possible
    id2path = {}
    for table in ("category_texts",):
        try:
            cols = db.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name=%s",
                (table,),
            )
            print("cols", table, [c["column_name"] for c in cols])
        except Exception as e:
            print(e)

    # try load paths from test sets for GT; for pred search name-only or from another source
    # page index tree file?
    try:
        trows = db.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        print("tables:", [r["tablename"] for r in trows])
    except Exception as e:
        print(e)

    # Load GT path from test sets
    gt_meta = {}
    for ts in [
        "../../test_set_llm_200.json",
        "../../test_set_llm_v2_200.json",
    ]:
        with open(ts, encoding="utf-8") as f:
            for item in json.load(f):
                gt_meta[str(item["ground_truth"])] = {
                    "name": item.get("ground_truth_name"),
                    "path": item.get("path", ""),
                    "product": item.get("product_name"),
                }

    # For predicted paths: try match via group + climb using same path prefix from siblings?
    # Use category_group_name as weak signal; also search if name appears in known GT paths.

    # Also search DB for synonyms containing product name to judge if better node exists
    syn_rows = db.execute(
        "SELECT category_id, category_name, syn_list FROM category_vectors WHERE syn_list IS NOT NULL"
    )
    # Build reverse: product -> nodes with exact name match
    name_nodes = {}
    for r in rows:
        name_nodes.setdefault(r["category_name"], []).append(str(r["category_id"]))

    reports = []
    for fname, tsname in [
        ("eval_llm200_full.json", "../../test_set_llm_200.json"),
        ("eval_llmv2_full.json", "../../test_set_llm_v2_200.json"),
    ]:
        with open(f"output/{fname}", encoding="utf-8") as f:
            d = json.load(f)
        with open(tsname, encoding="utf-8") as f:
            ts = {item["product_name"]: item for item in json.load(f)}

        wrong = [x for x in d["details"] if not x["correct"]]
        right = [x for x in d["details"] if x["correct"]]
        cats = Counter()
        samples = {k: [] for k in [
            "sibling_or_near", "same_subtree_depth>=3", "same_broad_category",
            "same_top_domain_only", "cross_domain", "pred_name_exactish",
            "no_match", "name_similar_no_path", "unclear",
        ]}

        # Check if product name exists as a category somewhere else
        better_exact = 0
        pred_more_specific = 0
        gt_suspicious = 0

        print("=" * 70)
        print(fname, "acc=", d["accuracy"], f"({d['correct']}/{d['total']})")
        print("wrong conf avg=", round(sum(x["confidence"] for x in wrong) / len(wrong), 3) if wrong else 0)
        print("right conf avg=", round(sum(x["confidence"] for x in right) / len(right), 3) if right else 0)

        for x in wrong:
            prod = x["product_name"]
            gt = str(x["ground_truth"])
            pred = str(x["predicted"]) if x["predicted"] is not None else None
            gt_item = ts.get(prod, {})
            gt_name = x.get("ground_truth_name") or id2name.get(gt, "")
            gt_path = gt_item.get("path") or ""
            pred_name = id2name.get(pred, "") if pred else ""
            pred_path = ""  # unknown unless we find another test item pointing here
            # reconstruct weak pred path from group
            if pred:
                g = id2group.get(pred, "")
                if g:
                    pred_path = g + " > " + pred_name
                else:
                    pred_path = pred_name

            cat = classify(prod, gt_name, pred_name, gt_path, pred_path)
            # refine: if pred group appears in gt path
            if pred and cat in ("cross_domain", "unclear", "same_top_domain_only"):
                g = id2group.get(pred, "")
                if g and g in gt_path:
                    cat = "same_broad_category"
            cats[cat] += 1

            # exact category name = product?
            exact_ids = name_nodes.get(prod, [])
            if exact_ids and gt not in exact_ids:
                better_exact += 1
                gt_suspicious += 1

            # pred name contains product more than gt?
            if pred_name and (prod in pred_name) and (prod not in gt_name):
                pred_more_specific += 1
                gt_suspicious += 1

            sample = {
                "product": prod,
                "gt": f"{gt_name}({gt})",
                "pred": f"{pred_name}({pred})",
                "gt_path": gt_path,
                "pred_group": id2group.get(pred, "") if pred else "",
                "conf": x["confidence"],
                "cat": cat,
            }
            if len(samples[cat]) < 8:
                samples[cat].append(sample)

        print("error taxonomy:", dict(cats))
        print("wrong cases with exact category named=product but GT elsewhere:", better_exact)
        print("pred name contains product, GT doesn't:", pred_more_specific)

        # print samples across key buckets
        for key in [
            "sibling_or_near", "same_subtree_depth>=3", "same_broad_category",
            "cross_domain", "pred_name_exactish",
        ]:
            if not samples[key]:
                continue
            print(f"\n--- samples [{key}] ---")
            for s in samples[key][:5]:
                print(f"  Q: {s['product']}  conf={s['conf']:.2f}")
                print(f"     GT  : {s['gt']}")
                print(f"           {s['gt_path']}")
                print(f"     PRED: {s['pred']} | group={s['pred_group']}")

        reports.append({
            "file": fname,
            "accuracy": d["accuracy"],
            "cats": dict(cats),
            "better_exact": better_exact,
            "pred_more_specific": pred_more_specific,
            "samples": {k: v for k, v in samples.items() if v},
        })

    # Special deep dive: 罗茨风机 / 环氧底漆 / searchable better nodes
    print("\n" + "=" * 70)
    print("Deep dive: does taxonomy contain more specific nodes?")
    for q in ["罗茨", "鼓风机", "环氧底漆", "环氧", "底漆", "雾炮", "杀毒", "碳纤维"]:
        hits = db.execute(
            """SELECT category_id, category_name, category_group_name
               FROM category_texts
               WHERE category_name LIKE %s
               ORDER BY category_id LIMIT 15""",
            (f"%{q}%",),
        )
        print(f"\nLIKE %{q}% ({len(hits)} shown):")
        for h in hits:
            print(f"  {h['category_id']}: {h['category_name']} | {h.get('category_group_name')}")

    with open("output/eval_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    db.close()
    print("\nWrote output/eval_error_analysis.json")


if __name__ == "__main__":
    main()
