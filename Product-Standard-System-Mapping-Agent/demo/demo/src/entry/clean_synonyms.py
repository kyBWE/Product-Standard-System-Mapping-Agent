from __future__ import annotations
import logging
import sys

from src.data.synonym_sanitizer import sanitize_syn_list
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger


def clean_synonyms_in_db(config_path: str = "config.yaml") -> dict[str, int]:
    """清洗数据库中 category_vectors / category_texts 的泛词同义词。"""
    logger = StructuredLogger()
    stats = {"vectors_updated": 0, "texts_updated": 0, "synonyms_removed": 0}

    config = ConfigManager(config_path)
    db = DBConnectionManager(config.get_db_config())
    db.initialize()

    for table in ("category_vectors", "category_texts"):
        rows = db.execute(
            f"SELECT category_id, category_name, syn_list FROM {table} WHERE syn_list IS NOT NULL"
        )
        batch: list[tuple] = []
        for row in rows:
            syn_list = list(row["syn_list"] or [])
            cleaned, removed = sanitize_syn_list(syn_list, row["category_name"])
            if not removed:
                continue
            stats["synonyms_removed"] += len(removed)
            batch.append((cleaned, row["category_id"]))

        if batch:
            db.execute_values_batch(
                f"""UPDATE {table} AS t
                    SET syn_list = v.syn_list,
                        updated_at = CURRENT_TIMESTAMP
                    FROM (VALUES %s) AS v(syn_list, category_id)
                    WHERE t.category_id = v.category_id""",
                batch,
                template="(%s::text[], %s)",
                page_size=500,
            )
            key = "vectors_updated" if table == "category_vectors" else "texts_updated"
            stats[key] = len(batch)

    db.close()
    logger.info("CleanSynonyms", f"数据库同义词清洗完成: {stats}")
    return stats


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    result = clean_synonyms_in_db(cfg)
    print(
        f"清洗完成: vectors={result['vectors_updated']}, "
        f"texts={result['texts_updated']}, removed={result['synonyms_removed']}"
    )
