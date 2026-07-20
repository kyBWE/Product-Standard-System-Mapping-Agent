import json

d = json.load(open(r"E:\Code\Projects\Product-Standard-System-Mapping-Agent-main\Product-Standard-System-Mapping-Agent\demo\demo\output\eval_2engine_v2.json", "r", encoding="utf-8"))
details = d["details"]
seen = set()
unique = []
for r in details:
    key = (r["product_name"], r["ground_truth"])
    if key not in seen:
        seen.add(key)
        unique.append(r)

wrong_rr = [r for r in unique if not r["engines"]["rag_rerank"]["correct"]]
wrong_pi = [r for r in unique if not r["engines"]["page_index"]["correct"]]

# 分析RAG+Rerank失败模式
sibling_miss = 0
near_miss = 0
far_miss = 0
no_match = 0
for r in wrong_rr:
    pred = r["engines"]["rag_rerank"]["predicted"]
    gt = r["ground_truth"]
    if pred is None:
        no_match += 1
    else:
        try:
            p = int(pred)
            g = int(gt)
            diff = abs(p - g)
            if diff <= 5:
                sibling_miss += 1
            elif diff <= 50:
                near_miss += 1
            else:
                far_miss += 1
        except ValueError:
            far_miss += 1

print(f"=== RAG+Rerank 失败分析 ({len(wrong_rr)}条) ===")
print(f"  无匹配: {no_match}")
print(f"  邻近错(ID差≤5): {sibling_miss} ← 同级/近亲混淆")
print(f"  近距错(ID差6-50): {near_miss} ← 同分支错")
print(f"  远距错(ID差>50): {far_miss} ← 完全偏移")

# 检查gt类别的同义词情况
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

c = ConfigManager("config.yaml")
db = DBConnectionManager(c.get_db_config())
db.initialize()

gt_ids = [r["ground_truth"] for r in unique]
gt_id_list = ",".join(f"'{g}'" for g in set(gt_ids))
rows = db.execute(f"SELECT category_id, category_name, syn_list, category_group_name FROM category_texts WHERE category_id IN ({gt_id_list})")
cat_info = {r["category_id"]: r for r in rows}

empty_syn = 0
short_syn = 0
good_syn = 0
for gid in set(gt_ids):
    info = cat_info.get(gid)
    if not info:
        continue
    syns = info["syn_list"] or []
    if len(syns) == 0:
        empty_syn += 1
    elif len(syns) <= 2:
        short_syn += 1
    else:
        good_syn += 1

print(f"\n=== GT类别同义词覆盖 ({len(set(gt_ids))}个) ===")
print(f"  无同义词: {empty_syn}")
print(f"  1-2个同义词: {short_syn}")
print(f"  3+个同义词: {good_syn}")

# 检查embedding文本长度
rows2 = db.execute(f"SELECT category_id, category_name, syn_list, category_group_name FROM category_texts WHERE category_id IN ({gt_id_list})")
emb_lengths = []
for r in rows2:
    text = r["category_name"]
    if r["syn_list"]:
        text += " " + " ".join(r["syn_list"])
    emb_lengths.append(len(text))
avg_len = sum(emb_lengths) / len(emb_lengths) if emb_lengths else 0
print(f"\n=== Embedding文本长度 ===")
print(f"  平均: {avg_len:.1f}字符")
print(f"  最短: {min(emb_lengths)}")
print(f"  最长: {max(emb_lengths)}")

# 示例：展示几个典型失败case
print(f"\n=== 典型失败案例 ===")
for r in wrong_rr[:10]:
    pred = r["engines"]["rag_rerank"]["predicted"]
    gt = r["ground_truth"]
    info = cat_info.get(gt, {})
    syns = info.get("syn_list", [])
    pname = r["product_name"]
    print(f"  {pname:<22} pred={pred or 'NONE':<8} gt={gt:<8} syns={syns[:3]}")

db.close()