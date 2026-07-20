import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.engine.llm_adapter import LLMAdapter
from src.index.page_index_tree import PageIndexTree
from src.data.excel_reader import ExcelDataReader
from src.data.taxonomy_utils import suggest_expansion_path

config = ConfigManager("config.yaml")
db_config = config.get_db_config()
llm_config = config.get_llm_config()

db = DBConnectionManager(db_config)
db.initialize()
llm = LLMAdapter(llm_config)

excel_reader = ExcelDataReader()
standard_file = r"C:\Users\11523\Downloads\产品标准体系.xlsx"
page_tree = PageIndexTree()
nodes, _ = excel_reader.load_standard_system(standard_file)
page_tree.build_tree(nodes)

print(f"树根节点数: {len(page_tree.get_root_nodes())}")
roots = page_tree.get_root_nodes()
for r in roots[:5]:
    print(f"  根: {r.category_name}(#{r.category_id}) 子节点数={len(r.children)}")

print()
print("测试 suggest_expansion_path('碳纤维')...")
result = suggest_expansion_path(llm, page_tree, db, "碳纤维")
print(f"  suggested_parent_id: {result['suggested_parent_id']}")
print(f"  suggested_parent_name: {result.get('suggested_parent_name', '')}")
print(f"  suggested_category_name: {result['suggested_category_name']}")
print(f"  confidence: {result['confidence']}")
print(f"  llm_reason: {result['llm_reason']}")
print(f"  path: {result['path']}")
print(f"  path_text: {result.get('path_text', '')}")
print(f"  suggested_path: {result.get('suggested_path', [])}")