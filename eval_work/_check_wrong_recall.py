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

    eval_data = json.load(open('output/eval_2engine_v3.json', 'r', encoding='utf-8'))
    wrong_items = [x for x in eval_data['details'] if not x['engines']['rag_rerank'].get('hit', False)]

    gt_in_top5 = 0
    gt_in_top10 = 0
    gt_in_top30 = 0
    gt_in_top50 = 0
    gt_not_found = 0
    gt_ranks = []

    for item in wrong_items:
        gt = item['ground_truth']
        pname = item['product_name']

        query_vec = vec_mgr.embed_query(pname)
        vec_results = vec_mgr.search_by_vector(query_vec, top_k=50)
        trgm_results = trgm_mgr.search_by_trgm(pname, threshold=0.1, limit=50)

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

        gt_rank = None
        for i, (cid, _) in enumerate(ranked):
            if cid == gt:
                gt_rank = i + 1
                break

        if gt_rank is None:
            gt_not_found += 1
        else:
            gt_ranks.append(gt_rank)
            if gt_rank <= 5:
                gt_in_top5 += 1
            if gt_rank <= 10:
                gt_in_top10 += 1
            if gt_rank <= 30:
                gt_in_top30 += 1
            if gt_rank <= 50:
                gt_in_top50 += 1

    total = len(wrong_items)
    print(f"RAG+Rerank wrong predictions: {total}")
    print(f"GT in top-5:  {gt_in_top5}/{total} = {gt_in_top5/total:.1%}")
    print(f"GT in top-10: {gt_in_top10}/{total} = {gt_in_top10/total:.1%}")
    print(f"GT in top-30: {gt_in_top30}/{total} = {gt_in_top30/total:.1%}")
    print(f"GT in top-50: {gt_in_top50}/{total} = {gt_in_top50/total:.1%}")
    print(f"GT not found: {gt_not_found}/{total}")

    if gt_ranks:
        print(f"GT rank (when found): median={sorted(gt_ranks)[len(gt_ranks)//2]}, mean={sum(gt_ranks)/len(gt_ranks):.1f}")

    print("\n=== GT rank > 10 (rerank can't reach) ===")
    for item in wrong_items:
        gt = item['ground_truth']
        pname = item['product_name']
        query_vec = vec_mgr.embed_query(pname)
        vec_results = vec_mgr.search_by_vector(query_vec, top_k=50)
        trgm_results = trgm_mgr.search_by_trgm(pname, threshold=0.1, limit=50)
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
        gt_rank = None
        for i, (cid, _) in enumerate(ranked):
            if cid == gt:
                gt_rank = i + 1
                break
        if gt_rank is not None and gt_rank > 10:
            rr_pred = item['engines']['rag_rerank'].get('predicted') or 'NONE'
            print(f"  {pname[:22]:<22} GT={gt:<6} rank={gt_rank:<4} RR_pred={rr_pred}")

    db.close()

if __name__ == "__main__":
    main()