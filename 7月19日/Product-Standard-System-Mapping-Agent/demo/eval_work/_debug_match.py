from src.index.vector_index_manager import VectorIndexManager
from src.index.trgm_index_manager import TrgmIndexManager
from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.config_manager import ConfigManager
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')


config = ConfigManager('config.yaml')
db_config = config.get_db_config()
llm_config = config.get_llm_config()
embedding_config = config.get_embedding_config()

db = DBConnectionManager(db_config)
llm = LLMAdapter(llm_config)
trgm_mgr = TrgmIndexManager(db)
vec_mgr = VectorIndexManager(
    db,
    embedding_model=llm_config.embedding_model,
    embedding_dimension=llm_config.embedding_dimension,
    base_url=llm_config.base_url,
    api_key=llm_config.api_key,
    embedding_config=embedding_config,
)

print(f"pgvector ready: {vec_mgr._use_pgvector}")
print(f"use_api: {vec_mgr._use_api}")

pg_ok = vec_mgr.ensure_pgvector_ready()
print(f"ensure_pgvector_ready: {pg_ok}")
vec_mgr.warmup()
print(f"vector count: {len(vec_mgr._category_ids) if hasattr(vec_mgr, '_category_ids') and vec_mgr._category_ids else 'N/A'}")

# Try embedding + search
query_vec = vec_mgr.embed_query("汽车")
print(f"query vector dim: {len(query_vec)}")
results = vec_mgr.search_by_vector(query_vec, top_k=5)
print(f"search results: {len(results)}")
for r in results:
    print(f"  {r.category_id} {r.category_name} sim={r.similarity:.4f}")

# Now try trgm
trgm_results = trgm_mgr.search_by_trgm("汽车", top_k=5, threshold=0.1)
print(f"trgm results: {len(trgm_results)}")
for r in trgm_results:
    print(f"  {r.category_id} {r.category_name} sim={r.similarity:.4f}")
