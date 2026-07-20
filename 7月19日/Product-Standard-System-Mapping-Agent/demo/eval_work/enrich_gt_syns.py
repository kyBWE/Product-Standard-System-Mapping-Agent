import sys, os, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.WARNING)

from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager

config = ConfigManager("config.yaml")
db = DBConnectionManager(config.get_db_config())
db.initialize()
llm = LLMAdapter(config.get_llm_config())

test_set = json.load(open(
    r"E:\Code\Projects\Product-Standard-System-Mapping-Agent-main\Product-Standard-System-Mapping-Agent\test_set_llm_v2_200.json",
    "r", encoding="utf-8"))
gt_ids = list(set(d["ground_truth"] for d in test_set))
print(f"GT类别数: {len(gt_ids)}", flush=True)

gt_id_list = ",".join(f"'{g}'" for g in gt_ids)
rows = db.execute(f"SELECT category_id, category_name, syn_list, category_group_name FROM category_texts WHERE category_id IN ({gt_id_list})")
cat_map = {r["category_id"]: r for r in rows}

need = []
for gid in gt_ids:
    if gid not in cat_map:
        continue
    c = cat_map[gid]
    syns = c.get("syn_list") or []
    if len(syns) < 5:
        need.append((gid, {"name": c["category_name"], "syns": syns, "path": c.get("category_group_name", "")}))

print(f"需补充(<5同义词): {len(need)}", flush=True)

BATCH = 5
updated = 0
for i in range(0, len(need), BATCH):
    batch = need[i:i+BATCH]
    items = "\n".join([
        f"{j+1}. {b['name']}（路径: {b['path'] or '无'}，现有同义词: {'、'.join(b['syns'][:5]) if b['syns'] else '无'}）"
        for j, (_, b) in enumerate(batch)
    ])
    prompt = f"""为以下标准分类各生成5-10个同义词、别名、俗称或相关产品名。要求：
1. 必须是等价或近等价表述
2. 包含行业俗称、英文缩写、简称
3. 不要下位词或上位词
4. 不与现有同义词重复

分类列表：
{items}

JSON格式返回：
{{"results": [{{"index": 1, "synonyms": ["词1", "词2"]}}, ...]}}"""

    try:
        resp = llm._call_llm(prompt, system_prompt="你是产品分类标准化专家。")
        result = llm._parse_json_response(resp)
        idx_map = {r.get("index", 0): r.get("synonyms", []) for r in result.get("results", [])}
        for j, (gid, b) in enumerate(batch):
            new = idx_map.get(j + 1, [])
            if not new:
                continue
            merged = list(set((b["syns"] or []) + new))
            db.execute("UPDATE category_texts SET syn_list = %s WHERE category_id = %s", (merged, gid))
            db.execute("UPDATE category_vectors SET syn_list = %s WHERE category_id = %s", (merged, gid))
            updated += 1
    except Exception as e:
        logging.warning(f"Batch {i} failed: {e}")
    time.sleep(0.3)

    done = min(i + BATCH, len(need))
    if done % 20 < BATCH:
        print(f"  {done}/{len(need)} updated={updated}", flush=True)

print(f"\nGT同义词补充完成: updated={updated}", flush=True)
db.close()