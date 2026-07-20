import json, sys, os, time, logging
import numpy as np
import psycopg2, yaml
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MANUAL_SYNS = {
    "4813": ["电子级氢氟酸", "氢氟酸", "电子级酸", "电子化学品", "湿电子化学品"],
    "24679": ["RVV线", "软护套线", "RVV软线", "护套电缆", "软电缆", "RVV电缆"],
    "1391": ["面粉", "小麦粉", "谷物粉", "碾磨粉", "粮食加工品"],
    "26445": ["人形机器人", "Atlas机器人", "仿人机器人", "双足机器人", "波士顿动力"],
    "12540": ["STP", "信令转接点", "七号信令STP", "信令转接设备"],
    "22018": ["吐温80", "Tween 80", "聚山梨酯80", "乳化剂80", "增溶乳化剂"],
    "14131": ["ERP系统", "ERP软件", "企业资源计划", "企业管理软件", "软件开发"],
    "9201": ["定型机", "拉幅定型机", "热定型机", "印染定型设备"],
    "8915": ["均质机阀组", "均质阀", "高压均质阀", "乳品设备零件"],
    "24538": ["EVA热熔胶", "封边条胶", "EVA封边胶", "家具封边胶", "热熔封边胶"],
    "11060": ["RTO焚烧炉", "蓄热式焚烧炉", "RTO", "蓄热式热力焚烧炉", "废气焚烧炉"],
    "5828": ["氧化铝陶瓷基板", "陶瓷基板", "氧化铝基板", "技术陶瓷"],
    "26593": ["NPB", "发光层主体材料", "OLED主体材料", "有机发光主体"],
    "10817": ["振动按摩器", "按摩器", "机械按摩器", "振动治疗器", "物理治疗器械"],
    "10798": ["筋膜枪", "按摩枪", "肌筋膜按摩器", "深层按摩器", "震动按摩枪"],
    "27220": ["CAR-T", "CAR-T疗法", "细胞免疫治疗", "基因治疗", "嵌合抗原受体T细胞"],
    "5052": ["生理盐水", "氯化钠注射液", "输液", "注射用盐水"],
    "25112": ["排线", "FPC排线", "柔性排线", "微细线材", "通讯排线"],
    "10153": ["OCT", "光学相干断层扫描", "OCT扫描仪", "激光诊断仪", "断层成像仪"],
    "3686": ["碱式氯化铝", "聚合氯化铝", "PAC", "羟基氯化铝", "絮凝剂"],
    "12883": ["X射线探测器", "平板探测器", "DR探测器", "X射线平板", "DR平板"],
    "839": ["松香", "树脂", "松脂", "天然树脂", "松香树脂"],
    "20165": ["调音台", "混音器", "数字调音台", "音频混音台", "专业音响"],
    "8006": ["高压清洗机", "高压水枪", "喷水清洗机", "高压冲洗机", "清洗器"],
}

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dbc = cfg["database"]
    conn = psycopg2.connect(
        host=dbc["host"], port=dbc["port"],
        dbname=dbc["database"], user=dbc["user"], password=dbc["password"]
    )
    cur = conn.cursor()

    updated = 0
    for cat_id, new_syns in MANUAL_SYNS.items():
        cur.execute("SELECT syn_list FROM category_vectors WHERE category_id = %s", (cat_id,))
        row = cur.fetchone()
        if not row:
            logging.warning(f"category_id {cat_id} not found in category_vectors")
            continue

        existing = list(row[0]) if row[0] else []
        merged = list(set(existing + new_syns))
        if len(merged) > len(existing):
            cur.execute("UPDATE category_vectors SET syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                       (merged, cat_id))
            cur.execute("UPDATE category_texts SET syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                       (merged, cat_id))
            added = [s for s in new_syns if s not in existing]
            logging.info(f"{cat_id}: added {added}")
            updated += 1

    conn.commit()
    logging.info(f"Updated {updated} categories with manual synonyms")

    from src.index.api_embedder import ApiEmbedder
    emb_cfg = cfg.get("embedding", {})
    embedder = ApiEmbedder(
        api_key=emb_cfg["api_key"],
        base_url=emb_cfg["base_url"],
        model=emb_cfg["model"],
        embedding_dim=emb_cfg["dimension"],
        batch_size=16,
    )

    for cat_id, new_syns in MANUAL_SYNS.items():
        cur.execute("SELECT category_name, syn_list FROM category_vectors WHERE category_id = %s", (cat_id,))
        row = cur.fetchone()
        if not row:
            continue

        cat_name = row[0]
        syns = list(row[1]) if row[1] else []

        cur.execute("SELECT category_group_name FROM category_texts WHERE category_id = %s", (cat_id,))
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
                (emb_bytes, vec_str, cat_id),
            )
            logging.info(f"Re-embedded {cat_id} ({cat_name}) with enriched text ({len(text)} chars)")
        except Exception as e:
            logging.warning(f"Failed to re-embed {cat_id}: {e}")

    conn.commit()
    logging.info("All dead-zone categories re-embedded")

    cur.execute("SELECT COUNT(*) FROM category_vectors WHERE vec_search IS NOT NULL")
    total = cur.fetchone()[0]
    logging.info(f"Total vectors with vec_search: {total}")

    conn.close()

if __name__ == "__main__":
    main()