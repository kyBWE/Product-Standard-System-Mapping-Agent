import json, sys, os, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
import psycopg2, yaml

logging.basicConfig(level=logging.WARNING)

def main():
    config = ConfigManager("config.yaml")
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    embedding_config = config.get_embedding_config()

    db = DBConnectionManager(db_config)
    db.initialize()

    vec_mgr = VectorIndexManager(
        db, embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url, api_key=llm_config.api_key,
        embedding_config=embedding_config,
    )
    vec_mgr.ensure_pgvector_ready()
    vec_mgr.warmup()

    with open("../../test_set_llm_v2_200.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    max_k = 200
    not_found_items = []

    for item in test_set:
        gt = item["ground_truth"]
        pname = item["product_name"]

        query_vec = vec_mgr.embed_query(pname)
        vec_results = vec_mgr.search_by_vector(query_vec, top_k=max_k)
        trgm_results = TrgmIndexManager(db).search_by_trgm(pname, threshold=0.1, limit=max_k)

        vec_ids = {r.category_id for r in vec_results}
        trgm_ids = {r.category_id for r in trgm_results}

        if gt not in vec_ids and gt not in trgm_ids:
            # Find best vec similarity for GT
            not_found_items.append({
                "product_name": pname,
                "gt_id": gt,
                "gt_name": item.get("ground_truth_name", ""),
                "path": item.get("path", ""),
            })

    print(f"GT not in top-200 candidates: {len(not_found_items)} items\n")

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dbc = cfg["database"]
    conn = psycopg2.connect(
        host=dbc["host"], port=dbc["port"],
        dbname=dbc["database"], user=dbc["user"], password=dbc["password"]
    )
    cur = conn.cursor()

    for item in not_found_items:
        gt_id = item["gt_id"]
        cur.execute("SELECT category_name, syn_list FROM category_vectors WHERE category_id = %s", (gt_id,))
        row = cur.fetchone()
        cat_name = row[0] if row else "UNKNOWN"
        syns = list(row[1]) if row and row[1] else []

        cur.execute("SELECT category_name, syn_list, category_group_name FROM category_texts WHERE category_id = %s", (gt_id,))
        row2 = cur.fetchone()
        group_name = row2[2] if row2 else ""

        print(f"产品: {item['product_name']}")
        print(f"  GT: {gt_id} {cat_name}")
        print(f"  路径: {item['path'][:60]}")
        print(f"  分组: {group_name}")
        print(f"  同义词: {syns[:5]}{'...' if len(syns)>5 else ''}")
        print()

    conn.close()
    db.close()

if __name__ == "__main__":
    main()