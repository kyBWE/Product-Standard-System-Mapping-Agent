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

rows = db.execute("SELECT category_id, category_name, syn_list, category_group_name FROM category_texts ORDER BY category_id")
all_cats = [{"id": r["category_id"], "name": r["category_name"], "syns": r["syn_list"] or [], "path": r["category_group_name"] or ""} for r in rows]
print(f"总类别: {len(all_cats)}", flush=True)

need_enrich = [c for c in all_cats if len(c["syns"]) < 3]
print(f"需补充同义词(<3个): {len(need_enrich)}", flush=True)

BATCH_SIZE = 5
updated = 0
errors = 0

for i in range(0, len(need_enrich), BATCH_SIZE):
    batch = need_enrich[i:i+BATCH_SIZE]
    items_text = "\n".join([
        f"{j+1}. {b['name']}（路径: {b['path'] or '无'}，现有同义词: {'、'.join(b['syns']) if b['syns'] else '无'}）"
        for j, b in enumerate(batch)
    ])

    prompt = f"""请为以下标准分类名称各生成5-8个同义词、别名、俗称或常见相关产品名称。要求：
1. 同义词必须是该分类的等价或近等价表述
2. 包含行业俗称、英文缩写、简称等
3. 不要生成下位词或上位词
4. 不要与现有同义词重复

分类列表：
{items_text}

请以JSON格式返回：
{{"results": [{{"index": 1, "synonyms": ["同义词1", "同义词2"]}}, {{"index": 2, "synonyms": ["同义词1", "同义词2"]}}]}}"""

    try:
        response = llm._call_llm(prompt, system_prompt="你是产品分类标准化专家，精通各行业专业术语和俗称。")
        result = llm._parse_json_response(response)
        results_data = result.get("results", [])
        index_map = {r.get("index", 0): r.get("synonyms", []) for r in results_data}

        for j, b in enumerate(batch):
            new_syns = index_map.get(j + 1, [])
            if not new_syns:
                continue
            merged = list(set(b["syns"] + new_syns))
            try:
                db.execute(
                    "UPDATE category_texts SET syn_list = %s WHERE category_id = %s",
                    (merged, b["id"]),
                )
                db.execute(
                    "UPDATE category_vectors SET syn_list = %s WHERE category_id = %s",
                    (merged, b["id"]),
                )
                updated += 1
            except Exception as e:
                errors += 1
    except Exception as e:
        logging.warning(f"Batch {i} failed: {e}")
        errors += BATCH_SIZE

    done = min(i + BATCH_SIZE, len(need_enrich))
    if done % 100 < BATCH_SIZE:
        print(f"  进度: {done}/{len(need_enrich)} 更新={updated} 错误={errors}", flush=True)
    time.sleep(0.3)

print(f"\n同义词补充完成: 更新={updated} 错误={errors}", flush=True)
db.close()