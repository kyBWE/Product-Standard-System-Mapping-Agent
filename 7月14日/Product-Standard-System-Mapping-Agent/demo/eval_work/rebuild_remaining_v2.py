import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

from src.data.excel_reader import ExcelDataReader
from src.data.synonym_sanitizer import sanitize_nodes
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager

config = ConfigManager("config.yaml")
db = DBConnectionManager(config.get_db_config())
db.initialize()
llm_config = config.get_llm_config()
embedding_config = config.get_embedding_config()

reader = ExcelDataReader()
nodes, _ = reader.load_standard_system(config.get("data.standard_system_file", "产品标准体系.xlsx"))
sanitize_nodes(nodes)
print(f"加载{len(nodes)}个节点", flush=True)

vec_mgr = VectorIndexManager(db, embedding_model=llm_config.embedding_model,
    embedding_dimension=llm_config.embedding_dimension,
    base_url=llm_config.base_url, api_key=llm_config.api_key,
    embedding_config=embedding_config)

try:
    db.execute("ALTER TABLE category_vectors DROP COLUMN IF EXISTS vec_search")
except Exception:
    pass

existing_ids = set()
try:
    rows = db.execute("SELECT category_id FROM category_vectors")
    existing_ids = {r["category_id"] for r in rows}
except Exception:
    pass

db_syn_map = vec_mgr._load_syn_list_from_db()
from src.index.category_enricher import CategoryEnricher
CategoryEnricher.apply_syn_cache(nodes)

remaining = []
for node in nodes:
    if node.category_id in db_syn_map and not node.syn_list:
        node.syn_list = db_syn_map[node.category_id]
    if node.category_id not in existing_ids:
        remaining.append(node)

print(f"已有{len(existing_ids)}条, 剩余{len(remaining)}条", flush=True)

if remaining:
    vec_mgr._category_ids = []
    vec_mgr._category_names = []
    for node in nodes:
        vec_mgr._category_ids.append(node.category_id)
        vec_mgr._category_names.append(node.category_name)

    t0 = time.time()
    success = vec_mgr._insert_api_vectors(remaining)
    print(f"向量重建完成: {success}条, 耗时{time.time()-t0:.1f}s", flush=True)

    vec_mgr.ensure_pgvector_ready()
else:
    print("所有向量已存在!", flush=True)
    vec_mgr.ensure_pgvector_ready()

db.close()
print("全部完成", flush=True)