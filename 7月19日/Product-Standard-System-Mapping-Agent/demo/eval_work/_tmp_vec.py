import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
c = ConfigManager("config.yaml")
db = DBConnectionManager(c.get_db_config())
db.initialize()
r = db.execute("SELECT COUNT(*) as cnt FROM category_vectors")
print(f"vectors: {r[0]['cnt']}")
try:
    r2 = db.execute("SELECT category_id, vec_search FROM category_vectors WHERE vec_search IS NOT NULL LIMIT 1")
    if r2:
        dim = len(str(r2[0]['vec_search']).strip('[]').split(','))
        print(f"vec_search dim: {dim}")
    else:
        print("vec_search: empty")
except Exception as e:
    print(f"vec_search: {e}")
db.close()