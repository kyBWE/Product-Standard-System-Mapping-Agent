"""重新生成向量数据：用 WordPiece tokenizer 重新计算所有 embedding。"""
from __future__ import annotations
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.index.vector_index_manager import VectorIndexManager
from src.data.excel_reader import ExcelDataReader

config = ConfigManager("config.yaml")
db_config = config.get_db_config()
llm_config = config.get_llm_config()

db = DBConnectionManager(db_config)
db.initialize()

vec_mgr = VectorIndexManager(
    db,
    embedding_model=llm_config.embedding_model,
    embedding_dimension=llm_config.embedding_dimension,
    base_url=llm_config.base_url,
    api_key=llm_config.api_key,
)
vec_mgr.ensure_pgvector_ready()

reader = ExcelDataReader()
nodes, _ = reader.load_standard_system(config.get("data.standard_system_file", "产品标准体系.xlsx"))

print(f"开始重新生成 {len(nodes)} 条向量...")
success = vec_mgr.insert_category_vectors(nodes)
print(f"向量重新生成完成: 成功 {success}/{len(nodes)}")

vec_mgr.warmup()

query_vec = vec_mgr.embed_query("非抗生素类普药")
results = vec_mgr.search_by_vector(query_vec, top_k=5)
print("\n测试: 非抗生素类普药 Top-5:")
for r in results:
    print(f"  {r.category_name}({r.category_id}): sim={r.similarity:.4f}")

db.close()