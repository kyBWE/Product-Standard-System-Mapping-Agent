# -*- coding: utf-8 -*-
"""按误差分析结论修正测试集 GT，并为评测加入宽松指标。

规则（仅自动修正高把握项）:
1) 若 category_texts 存在 category_name == product_name 的节点，且当前 GT 不是该节点之一，
   则把 GT 改成该「同名叶」（多个同名时优先路径与旧 GT 同顶域/最长公共前缀更深者）。
2) 同步更新 ground_truth_name / path。
3) 写 backup 与变更日志。
"""
from __future__ import annotations
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager

ROOT = Path(__file__).resolve().parents[2]
TEST_FILES = [
    ROOT / "test_set_llm_200.json",
    ROOT / "test_set_llm_v2_200.json",
]


def split_path(path_or_group: str) -> list[str]:
    if not path_or_group:
        return []
    if " > " in path_or_group:
        return [x.strip() for x in path_or_group.split(" > ") if x.strip()]
    return [x.strip() for x in path_or_group.split(",") if x.strip()]


def lca_depth(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_path_str(group: str, name: str) -> str:
    parts = split_path(group)
    if name and (not parts or parts[-1] != name):
        parts = parts + [name]
    return " > ".join(parts)


def pick_literal(cands: list[dict], old_gt: str, old_path: list[str], id2group: dict) -> dict:
    """从同名候选中选最合理的一个。"""
    if len(cands) == 1:
        return cands[0]
    scored = []
    for c in cands:
        cid = str(c["category_id"])
        gpath = split_path(id2group.get(cid, "")) + [c["category_name"]]
        depth = lca_depth(old_path, gpath)
        # 同名且相对旧 GT 路径更近优先；其次路径更深（更具体）
        scored.append((depth, len(gpath), cid == old_gt, c))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def fix_one_file(path: Path, name_nodes: dict, id2name: dict, id2group: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    for i, item in enumerate(data):
        prod = item.get("product_name") or ""
        gt = str(item.get("ground_truth"))
        cands = name_nodes.get(prod)
        if not cands:
            continue
        cand_ids = {str(c["category_id"]) for c in cands}
        if gt in cand_ids:
            continue
        old_path = split_path(item.get("path") or "")
        if not old_path:
            old_path = split_path(id2group.get(gt, "")) + ([id2name.get(gt, "")] if id2name.get(gt) else [])
        chosen = pick_literal(cands, gt, old_path, id2group)
        new_id = str(chosen["category_id"])
        new_name = chosen["category_name"]
        new_path = build_path_str(id2group.get(new_id, ""), new_name)
        change = {
            "index": i,
            "product_name": prod,
            "old_gt": gt,
            "old_name": item.get("ground_truth_name"),
            "old_path": item.get("path"),
            "new_gt": new_id,
            "new_name": new_name,
            "new_path": new_path,
            "reason": "literal_category_name_equals_product",
        }
        item["ground_truth"] = new_id
        item["ground_truth_name"] = new_name
        item["path"] = new_path
        item["ground_truth_fix"] = "literal_leaf_2026-07-14"
        changes.append(change)
    return {"data": data, "changes": changes}


def main():
    config = ConfigManager("config.yaml")
    db = DBConnectionManager(config.get_db_config())
    db.initialize()
    rows = db.execute(
        "SELECT category_id, category_name, category_group_name FROM category_texts"
    )
    id2name = {str(r["category_id"]): r["category_name"] for r in rows}
    id2group = {str(r["category_id"]): (r.get("category_group_name") or "") for r in rows}
    name_nodes: dict[str, list[dict]] = {}
    for r in rows:
        name_nodes.setdefault(r["category_name"], []).append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_logs = {}
    for path in TEST_FILES:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        backup = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.copy2(path, backup)
        result = fix_one_file(path, name_nodes, id2name, id2group)
        path.write_text(
            json.dumps(result["data"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_logs[path.name] = {
            "backup": str(backup),
            "fixed_count": len(result["changes"]),
            "changes": result["changes"],
        }
        print(f"{path.name}: fixed {len(result['changes'])} -> backup {backup.name}")

    log_path = Path("output") / f"gt_literal_fix_{stamp}.json"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"changelog -> {log_path}")
    db.close()


if __name__ == "__main__":
    main()
