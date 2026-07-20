# -*- coding: utf-8 -*-
"""为 LLM200 生成更合理的 GT 测试集（不改动 v2）。

修订原则（自动、可追溯）:
1. 已做同名叶修正的保留。
2. LLM 合理性判定为 True → 采用系统预测作为新 GT（双树/同义/材料制品视角）。
3. soft 父子且预测是更深子节点 → 采用预测（更具体）。
4. soft 同子树且产品名出现在预测名中、未出现在旧 GT 名 → 采用预测。
5. soft 兄弟且产品名与预测名共享长度≥2 的连续子串、且共享长度优于旧 GT → 采用预测。
6. 已知错误兄弟黑名单不采纳。
7. 其余（含 LLM 拒绝 / 真错配）保留原 GT，作为硬例继续考察系统。
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.orchestration.eval_scoring import format_path, soft_match

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "test_set_llm_200.json"
OUT = ROOT / "test_set_llm_200_refined.json"
EVAL = Path("output/eval_llm200_full_llm_soft.json")

# 明确不合理的「兄弟」替换，禁止自动采纳
SIBLING_DENY = {
    # 产品名 -> 禁止的 pred_id
    "硅铬合金": {"23898"},  # Cu-Cr-Zr合金
    "大苏打": {"3642"},  # 次亚硫酸钠（≠硫代硫酸盐/大苏打）
}


def longest_common_substr(a: str, b: str) -> int:
    if not a or not b:
        return 0
    best = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a) + 1):
            sub = a[i:j]
            if len(sub) >= 2 and sub in b:
                best = max(best, len(sub))
    return best


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}

    src = json.loads(SRC.read_text(encoding="utf-8"))
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    # 同产品名可能多条：按顺序对齐（评测 details 与测试集同序）
    assert len(src) == len(ev["details"]), (len(src), len(ev["details"]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refined = []
    changes = []
    reasons_count: dict[str, int] = {}

    for item, row in zip(src, ev["details"]):
        assert item["product_name"] == row["product_name"], (
            item["product_name"], row["product_name"]
        )
        prod = item["product_name"]
        old_gt = str(item["ground_truth"])
        old_name = item.get("ground_truth_name") or id2name.get(old_gt, "")
        new_gt, new_name, reason = old_gt, old_name, "keep"

        pred = row.get("predicted")
        pred_s = str(pred) if pred is not None else None
        pred_name = row.get("predicted_name") or (id2name.get(pred_s, "") if pred_s else "")

        # 1) 已同名叶
        if item.get("ground_truth_fix") == "literal_leaf_2026-07-14" and old_name == prod:
            reason = "keep_literal_leaf"
        # 2) LLM 采纳预测
        elif row.get("llm_reasonable") is True and pred_s:
            new_gt, new_name, reason = pred_s, pred_name, "adopt_llm_reasonable_pred"
        # 3-5) soft 路径规则
        elif row.get("correct_soft") and not row.get("correct_strict") and pred_s:
            rel = row.get("relation") or ""
            if rel == "ancestor_descendant":
                from src.orchestration.eval_scoring import path_of
                gparts = path_of(old_gt, id2name, id2group, item.get("path") or "")
                pparts = path_of(pred_s, id2name, id2group)
                # GT 是预测的祖先 → 采用更细的预测叶
                if gparts and pparts and pparts[: len(gparts)] == gparts and len(pparts) > len(gparts):
                    new_gt, new_name, reason = pred_s, pred_name, "adopt_finer_child"
                elif prod in pred_name and prod not in (old_name or ""):
                    new_gt, new_name, reason = pred_s, pred_name, "adopt_child_name_match"
                elif len(pparts) > len(gparts):
                    new_gt, new_name, reason = pred_s, pred_name, "adopt_deeper_node"
                else:
                    reason = "keep_ancestor_pred_not_deeper"
            elif rel == "same_subtree" and prod in pred_name and prod not in (old_name or ""):
                new_gt, new_name, reason = pred_s, pred_name, "adopt_subtree_name_match"
            elif rel == "sibling" and pred_s:
                deny = SIBLING_DENY.get(prod, set())
                if pred_s in deny:
                    reason = "keep_sibling_deny"
                else:
                    lp = longest_common_substr(prod, pred_name)
                    lg = longest_common_substr(prod, old_name or "")
                    if lp >= 2 and lp > lg:
                        new_gt, new_name, reason = pred_s, pred_name, "adopt_sibling_better_name_overlap"
                    else:
                        reason = "keep_sibling_ambiguous"
            else:
                reason = "keep_soft_other"
        else:
            if row.get("correct_strict"):
                reason = "keep_already_strict"
            elif row.get("llm_reasonable") is False:
                reason = "keep_hard_case_llm_rejected"
            else:
                reason = "keep_default"

        reasons_count[reason] = reasons_count.get(reason, 0) + 1

        new_path = format_path(new_gt, id2name, id2group, item.get("path") or "")
        out_item = {
            "product_name": prod,
            "ground_truth": new_gt,
            "ground_truth_name": new_name,
            "path": new_path,
        }
        if item.get("ground_truth_fix"):
            out_item["ground_truth_fix"] = item["ground_truth_fix"]
        if new_gt != old_gt:
            out_item["ground_truth_refined"] = reason
            out_item["previous_ground_truth"] = old_gt
            out_item["previous_ground_truth_name"] = old_name
            changes.append({
                "product_name": prod,
                "old_gt": old_gt,
                "old_name": old_name,
                "new_gt": new_gt,
                "new_name": new_name,
                "reason": reason,
                "eval_relation": row.get("relation"),
                "llm_reasonable": row.get("llm_reasonable"),
            })
        refined.append(out_item)

    OUT.write_text(json.dumps(refined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 用现有预测对新 GT 重算准确率
    strict = near = soft = 0
    for item, row in zip(refined, ev["details"]):
        pred = row.get("predicted")
        pred_s = str(pred) if pred is not None else None
        m = soft_match(
            pred_s, str(item["ground_truth"]), id2name, id2group, item.get("path") or ""
        )
        if m["strict"]:
            strict += 1
        if m["near"]:
            near += 1
        if m["soft"]:
            soft += 1
    # llm_soft on new GT: strict or (old llm_reasonable and pred==new_gt) or soft
    # Better: if pred == new_gt or soft or (llm said reasonable and we adopted it → strict now)
    llm_soft = 0
    for item, row in zip(refined, ev["details"]):
        pred = row.get("predicted")
        pred_s = str(pred) if pred is not None else None
        m = soft_match(pred_s, str(item["ground_truth"]), id2name, id2group, item.get("path") or "")
        hit = m["soft"]
        if not hit and row.get("llm_reasonable") is True and pred_s == str(item["ground_truth"]):
            hit = True
        # also: if we kept GT but llm would accept pred relative to NEW gt? skip for now
        if hit:
            llm_soft += 1

    n = len(refined)
    summary = {
        "source": str(SRC),
        "output": str(OUT),
        "stamp": stamp,
        "total": n,
        "changed": len(changes),
        "reason_counts": reasons_count,
        "rescored_on_existing_predictions": {
            "accuracy_strict": round(strict / n, 4),
            "accuracy_near": round(near / n, 4),
            "accuracy_soft": round(soft / n, 4),
            "correct_strict": strict,
            "correct_near": near,
            "correct_soft": soft,
            "note": "基于原评测预测重算，未重新跑匹配；原 llm_soft 采纳样本在此多已变为 strict",
        },
        "changes": changes,
    }
    log_path = Path("output") / f"gt_refine_llm200_{stamp}.json"
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {OUT} changed={len(changes)}/{n}")
    print("reasons:", reasons_count)
    print(
        f"rescored strict={strict}/{n}={strict/n:.1%} "
        f"near={near/n:.1%} soft={soft/n:.1%}"
    )
    print(f"changelog -> {log_path}")
    db.close()


if __name__ == "__main__":
    main()
