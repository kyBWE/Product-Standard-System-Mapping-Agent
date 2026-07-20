import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.index.trgm_index_manager import TrgmIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.index.vector_index_manager import VectorIndexManager

config = ConfigManager("config.yaml")
db = DBConnectionManager(config.get_db_config())
db.initialize()
trgm = TrgmIndexManager(db)

results = trgm.search_by_trgm("苹果汁", threshold=0.3, limit=10)
print("=== Trigram Top-10 ===")
for r in results:
    print(f"  {r.category_name}({r.category_id}): sim={r.similarity:.4f}")

llm_config = config.get_llm_config()
vec = VectorIndexManager(db, embedding_model=llm_config.embedding_model,
    embedding_dimension=llm_config.embedding_dimension,
    base_url=llm_config.base_url, api_key=llm_config.api_key)
vec.ensure_pgvector_ready()
vec.warmup()
qvec = vec.embed_query("苹果汁")
vresults = vec.search_by_vector(qvec, top_k=10)
print()
print("=== Vector Top-10 ===")
for r in vresults:
    print(f"  {r.category_name}({r.category_id}): sim={r.similarity:.4f}")

db.close()