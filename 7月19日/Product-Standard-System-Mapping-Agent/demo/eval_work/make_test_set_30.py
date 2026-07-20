# -*- coding: utf-8 -*-
"""从 refined LLM200 抽取 30 条分层小集（易/中/难）。"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "test_set_llm_200_refined.json"
EVAL = Path(__file__).resolve().parent / "output" / "eval_llm200_full_llm_soft.json"
# fallback to 7月14 demo output
if not EVAL.exists():
    EVAL = Path(
        r"c:\Users\hp\Desktop\课设3\7月14日\Product-Standard-System-Mapping-Agent"
        r"\demo\demo\output\eval_llm200_full_llm_soft.json"
    )

items = json.loads(SRC.read_text(encoding="utf-8"))
ev = json.loads(EVAL.read_text(encoding="utf-8"))
by: dict[str, dict] = {}
for d in ev["details"]:
    by.setdefault(d["product_name"], d)

easy, med, hard = [], [], []
seen = set()
for it in items:
    name = it["product_name"]
    if name in seen:
        continue
    seen.add(name)
    d = by.get(name)
    if not d:
        continue
    if not d.get("correct_llm_soft"):
        hard.append(it)
    elif d.get("correct_strict") and not it.get("ground_truth_refined"):
        easy.append(it)
    else:
        med.append(it)

print("buckets", len(easy), len(med), len(hard))

sel = []
for bucket, tag, n in ((easy, "easy", 12), (med, "medium", 10), (hard, "hard", 8)):
    for x in bucket[:n]:
        sel.append(
            {
                "product_name": x["product_name"],
                "ground_truth": str(x["ground_truth"]),
                "ground_truth_name": x.get("ground_truth_name", ""),
                "path": x.get("path", ""),
                "difficulty": tag,
            }
        )

print(Counter(x["difficulty"] for x in sel), "n=", len(sel))
for x in sel:
    print(f"{x['difficulty']:6} {x['product_name']} -> {x['ground_truth_name']}")

dest = ROOT / "test_set_llm_30.json"
dest.write_text(json.dumps(sel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
txt = ROOT / "test_set_llm_30_names.txt"
txt.write_text("\n".join(x["product_name"] for x in sel) + "\n", encoding="utf-8")

alt_root = Path(r"c:\Users\hp\Desktop\课设3\7月14日\Product-Standard-System-Mapping-Agent")
(alt_root / "test_set_llm_30.json").write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
(alt_root / "test_set_llm_30_names.txt").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
print("wrote", dest)
print("wrote", txt)
