# -*- coding: utf-8 -*-
"""从 refined LLM200 生成 3 份互不重叠的 30 条小集（各含 easy/medium/hard）。"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "test_set_llm_200_refined.json"
EVAL = Path(__file__).resolve().parent / "output" / "eval_llm200_full_llm_soft.json"
if not EVAL.exists():
    EVAL = Path(
        r"c:\Users\hp\Desktop\课设3\7月14日\Product-Standard-System-Mapping-Agent"
        r"\demo\demo\output\eval_llm200_full_llm_soft.json"
    )
ALT = Path(r"c:\Users\hp\Desktop\课设3\7月14日\Product-Standard-System-Mapping-Agent")

N_EASY, N_MED, N_HARD = 12, 10, 8
SPLITS = 3  # A/B/C


def main():
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

    need_e, need_m, need_h = N_EASY * SPLITS, N_MED * SPLITS, N_HARD * SPLITS
    print(
        f"available easy={len(easy)} med={len(med)} hard={len(hard)}; "
        f"need {need_e}/{need_m}/{need_h}"
    )
    if len(easy) < need_e or len(med) < need_m or len(hard) < need_h:
        # 不够时从相邻桶补（仍保证三份产品名不重合）
        print("WARNING: bucket shortage, will top-up from other buckets")

    def take(lst: list, n: int) -> list:
        out, lst[:] = lst[:n], lst[n:]
        return out

    sets = []
    used = set()
    for i in range(SPLITS):
        e = take(easy, N_EASY)
        m = take(med, N_MED)
        h = take(hard, N_HARD)
        # top-up if short
        pool = easy + med + hard
        while len(e) + len(m) + len(h) < 30 and pool:
            x = pool.pop(0)
            if x["product_name"] in used:
                continue
            # put into whichever short
            if len(e) < N_EASY:
                e.append(x)
            elif len(m) < N_MED:
                m.append(x)
            else:
                h.append(x)
        # rebuild remaining pools without taken
        taken_names = {x["product_name"] for x in e + m + h}
        used |= taken_names
        easy = [x for x in easy if x["product_name"] not in used]
        med = [x for x in med if x["product_name"] not in used]
        hard = [x for x in hard if x["product_name"] not in used]

        sel = []
        for bucket, tag in ((e, "easy"), (m, "medium"), (h, "hard")):
            for x in bucket:
                sel.append(
                    {
                        "product_name": x["product_name"],
                        "ground_truth": str(x["ground_truth"]),
                        "ground_truth_name": x.get("ground_truth_name", ""),
                        "path": x.get("path", ""),
                        "difficulty": tag,
                        "split": f"set{i + 1}",
                    }
                )
        sets.append(sel)
        print(f"set{i+1}", Counter(x["difficulty"] for x in sel), "n=", len(sel))

    # overlap check
    names = [set(x["product_name"] for x in s) for s in sets]
    for a in range(SPLITS):
        for b in range(a + 1, SPLITS):
            ov = names[a] & names[b]
            print(f"overlap set{a+1}&set{b+1}: {len(ov)}")
            assert not ov, ov

    # write set1 overwrite old 30; set2/set3 new
    labels = ["30", "30b", "30c"]
    for sel, lab in zip(sets, labels):
        dest = ROOT / f"test_set_llm_{lab}.json"
        txt = ROOT / f"test_set_llm_{lab}_names.txt"
        dest.write_text(json.dumps(sel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        txt.write_text("\n".join(x["product_name"] for x in sel) + "\n", encoding="utf-8")
        (ALT / dest.name).write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
        (ALT / txt.name).write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
        print("wrote", dest)

    # index readme
    idx = {
        "source": str(SRC.name),
        "note": "三份 30 条小集，产品名互不重合；各约 easy12/medium10/hard8",
        "files": [
            {
                "json": f"test_set_llm_{lab}.json",
                "names": f"test_set_llm_{lab}_names.txt",
                "count": len(sets[i]),
                "difficulties": dict(Counter(x["difficulty"] for x in sets[i])),
            }
            for i, lab in enumerate(labels)
        ],
        "paths": {
            "jul17": str(ROOT),
            "jul14": str(ALT),
        },
    }
    (ROOT / "test_set_llm_30_index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ALT / "test_set_llm_30_index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("index -> test_set_llm_30_index.json")


if __name__ == "__main__":
    main()
