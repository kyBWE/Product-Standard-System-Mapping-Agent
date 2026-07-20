import json, sys, os, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager

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

    trgm_mgr = TrgmIndexManager(db)

    with open("../../test_set_llm_v2_200.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    total = len(test_set)
    max_k = 200

    all_items = []
    for idx, item in enumerate(test_set):
        gt = item["ground_truth"]
        pname = item["product_name"]

        query_vec = vec_mgr.embed_query(pname)
        vec_results = vec_mgr.search_by_vector(query_vec, top_k=max_k)
        trgm_results = trgm_mgr.search_by_trgm(pname, threshold=0.1, limit=max_k)

        vec_ids = {r.category_id: r.similarity for r in vec_results}
        trgm_ids = {r.category_id: r.similarity for r in trgm_results}
        all_ids = set(vec_ids.keys()) | set(trgm_ids.keys())

        ranked = []
        for cid in all_ids:
            vs = vec_ids.get(cid, 0)
            ts = trgm_ids.get(cid, 0)
            score = 0.6 * vs + 0.4 * ts
            if ts >= 0.8:
                score = 0.3 * vs + 0.7 * ts
            ranked.append((cid, score))

        ranked.sort(key=lambda x: -x[1])
        all_items.append((gt, pname, ranked, vec_ids, trgm_ids))

        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx+1}/{total}", flush=True)

    for top_k in [5, 10, 20, 30, 50, 100, 200]:
        hits = 0
        for gt, pname, ranked, vec_ids, trgm_ids in all_items:
            top_ids = set(cid for cid, _ in ranked[:top_k])
            if gt in top_ids:
                hits += 1
        print(f"top_k={top_k:>3}: {hits}/{total} = {hits/total:.1%}")

    print("\n=== GT not in top-200 (vec+trgm) ===")
    not_found = []
    for gt, pname, ranked, vec_ids, trgm_ids in all_items:
        all_cids = set(cid for cid, _ in ranked)
        if gt not in all_cids:
            in_vec = gt in vec_ids
            in_trgm = gt in trgm_ids
            not_found.append((pname, gt, in_vec, in_trgm))

    print(f"Count: {len(not_found)}")
    for pname, gt, iv, it in not_found[:20]:
        print(f"  {pname[:25]:<25} GT={gt:<6} vec={'Y' if iv else 'N'} trgm={'Y' if it else 'N'}")

    print("\n=== GT in vec but not in trgm (vec-only contribution) ===")
    vec_only = 0
    for gt, pname, ranked, vec_ids, trgm_ids in all_items:
        if gt in vec_ids and gt not in trgm_ids:
            vec_only += 1
    print(f"Count: {vec_only}")

    db.close()

if __name__ == "__main__":
    main()
