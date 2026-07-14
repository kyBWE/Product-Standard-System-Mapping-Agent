import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
c = ConfigManager("config.yaml")
db = DBConnectionManager(c.get_db_config())
db.initialize()
r = db.execute("SELECT COUNT(*) as cnt FROM category_texts WHERE array_length(syn_list,1) >= 3")
print(f"3+syns: {r[0]['cnt']}")
r2 = db.execute("SELECT COUNT(*) as cnt FROM category_texts WHERE array_length(syn_list,1) IS NULL OR array_length(syn_list,1) = 0")
print(f"0 syns: {r2[0]['cnt']}")
r3 = db.execute("SELECT COUNT(*) as cnt FROM category_texts")
print(f"total: {r3[0]['cnt']}")
db.close()