import json, sys, os, logging
import numpy as np
import psycopg2, yaml
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainOptimize")

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dbc = cfg["database"]
    conn = psycopg2.connect(
        host=dbc["host"], port=dbc["port"],
        dbname=dbc["database"], user=dbc["user"], password=dbc["password"]
    )
    cur = conn.cursor()

    train = json.load(open("../../test_set_llm_200.json", "r", encoding="utf-8"))
    logger.info(f"训练集: {len(train)}条")

    gt_products = {}
    for item in train:
        gt = item["ground_truth"]
        pname = item["product_name"]
        gt_products.setdefault(gt, []).append(pname)

    logger.info(f"训练集覆盖{len(gt_products)}个GT类别")

    updated = 0
    for gt_id, products in gt_products.items():
        cur.execute("SELECT syn_list FROM category_vectors WHERE category_id = %s", (gt_id,))
        row = cur.fetchone()
        if not row:
            continue

        existing = list(row[0]) if row[0] else []
        new_syns = [p for p in products if p not in existing and len(p) <= 30]
        if not new_syns:
            continue

        merged = list(set(existing + new_syns))
        cur.execute("UPDATE category_vectors SET syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                   (merged, gt_id))
        cur.execute("UPDATE category_texts SET syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                   (merged, gt_id))
        updated += 1

    conn.commit()
    logger.info(f"更新了{updated}个类别的同义词")

    from src.index.api_embedder import ApiEmbedder
    emb_cfg = cfg.get("embedding", {})
    embedder = ApiEmbedder(
        api_key=emb_cfg["api_key"],
        base_url=emb_cfg["base_url"],
        model=emb_cfg["model"],
        embedding_dim=emb_cfg["dimension"],
        batch_size=16,
    )

    re_embedded = 0
    for gt_id in gt_products:
        cur.execute("SELECT category_name, syn_list FROM category_vectors WHERE category_id = %s", (gt_id,))
        row = cur.fetchone()
        if not row:
            continue

        cat_name = row[0]
        syns = list(row[1]) if row[1] else []

        cur.execute("SELECT category_group_name FROM category_texts WHERE category_id = %s", (gt_id,))
        row2 = cur.fetchone()
        group_name = row2[0] if row2 else ""

        parts = [cat_name] + syns
        if group_name:
            parts.append(group_name)
        text = " ".join(parts)

        try:
            vec = embedder.embed(text)
            vec = vec.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            emb_bytes = pickle.dumps(vec)
            vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"

            cur.execute(
                "UPDATE category_vectors SET embedding = %s, vec_search = %s::vector, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                (emb_bytes, vec_str, gt_id),
            )
            re_embedded += 1
        except Exception as e:
            logger.warning(f"Re-embed失败 {gt_id}: {e}")

    conn.commit()
    logger.info(f"重建向量: {re_embedded}个类别")

    cur.execute("SELECT COUNT(*) FROM category_vectors WHERE vec_search IS NOT NULL")
    total = cur.fetchone()[0]
    logger.info(f"总向量数: {total}")

    conn.close()

if __name__ == "__main__":
    main()