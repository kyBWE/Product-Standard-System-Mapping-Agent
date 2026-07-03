from __future__ import annotations
import json
import logging
import os
import sys
import time
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("FixGroundTruth")

INPUT_PATH = "output/test_set_1000.json"
OUTPUT_PATH = "output/test_set_1000_fixed.json"
NEED_FIX_SOURCES = {"llm_arbitrate_fallback_top_vote", "top_vote_fallback", "top_vote_1"}


def get_category_names(db: DBConnectionManager, cat_ids: list[str]) -> dict[str, str]:
    names = {}
    for cat_id in cat_ids:
        row = db.execute_one(
            "SELECT category_name FROM category_vectors WHERE category_id = %s",
            (cat_id,),
        )
        if row:
            names[cat_id] = row["category_name"]
        else:
            names[cat_id] = cat_id
    return names


def llm_arbitrate(client: OpenAI, model: str, product_name: str, candidates: list[tuple[str, str, int]]) -> str | None:
    prompt = f'产品"{product_name}"应该归属以下哪个标准分类？\n'
    for i, (cat_id, cat_name, votes) in enumerate(candidates):
        prompt += f"{chr(65+i)}. {cat_name}(id={cat_id})，得票{votes}\n"
    prompt += "\n请只回复选项字母(A/B/C/D)，不要输出其他内容。"

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                timeout=30,
            )
            answer = resp.choices[0].message.content.strip().upper()
            for ch in answer:
                if ch in "ABCD":
                    idx = ord(ch) - ord('A')
                    if idx < len(candidates):
                        return candidates[idx][0]
        except Exception as e:
            logger.warning(f"LLM仲裁失败(第{attempt+1}次): {product_name}, error={e}")
            time.sleep(1)
    return None


def main():
    config = ConfigManager("config.yaml")
    llm_config = config.get_llm_config()
    db_config = config.get_db_config()

    db = DBConnectionManager(db_config)
    db.initialize()

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    model = llm_config.model

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    need_fix = [item for item in data if item["ground_truth_source"] in NEED_FIX_SOURCES]
    logger.info(f"总条数: {len(data)}, 需要修复: {len(need_fix)}")

    fixed = 0
    failed = 0
    for i, item in enumerate(need_fix):
        results = [
            item["rag_result"],
            item["rag_rerank_result"],
            item["page_index_result"],
            item["page_index_force_result"],
        ]
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            failed += 1
            continue

        counter = Counter(valid_results)
        cat_ids = list(counter.keys())
        cat_names = get_category_names(db, cat_ids)

        candidates = [(cat_id, cat_names.get(cat_id, cat_id), count) for cat_id, count in counter.most_common()]

        answer = llm_arbitrate(client, model, item["product_name"], candidates)
        if answer:
            item["ground_truth"] = answer
            item["ground_truth_source"] = "llm_arbitrate_fixed"
            fixed += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"进度: {i+1}/{len(need_fix)} (成功={fixed}, 失败={failed})")
        time.sleep(0.1)

    logger.info(f"修复完成: 成功={fixed}, 失败={failed}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {OUTPUT_PATH}")

    from collections import Counter as C2
    sources = C2(item["ground_truth_source"] for item in data)
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        logger.info(f"  {k}: {v}")

    db.close()


if __name__ == "__main__":
    main()