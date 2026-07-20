import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

print("1. 导入Flask...", flush=True)
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def t():
    return "ok"

print("2. 导入配置...", flush=True)
from src.infrastructure.config_manager import ConfigManager
config = ConfigManager("config.yaml")

print("3. 连接数据库...", flush=True)
from src.infrastructure.db_manager import DBConnectionManager
db = DBConnectionManager(config.get_db_config())
db.initialize()
r = db.execute("SELECT COUNT(*) as cnt FROM category_vectors")
print(f"   向量数: {r[0]['cnt']}", flush=True)

print("4. 初始化LLM...", flush=True)
from src.engine.llm_adapter import LLMAdapter
llm = LLMAdapter(config.get_llm_config())

print("5. 初始化向量索引...", flush=True)
from src.index.vector_index_manager import VectorIndexManager
vec_mgr = VectorIndexManager(
    db,
    embedding_model=config.get_llm_config().embedding_model,
    embedding_dimension=config.get_llm_config().embedding_dimension,
    base_url=config.get_llm_config().base_url,
    api_key=config.get_llm_config().api_key,
)
vec_mgr.ensure_pgvector_ready()
print("   预热向量矩阵...", flush=True)
vec_mgr.warmup()

print("6. 初始化PageIndex树...", flush=True)
from src.data.excel_reader import ExcelDataReader
from src.index.page_index_tree import PageIndexTree
reader = ExcelDataReader()
nodes, _ = reader.load_standard_system("产品标准体系.xlsx")
page_tree = PageIndexTree()
page_tree.build_tree(nodes)

print("7. 启动Flask...", flush=True)
app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)