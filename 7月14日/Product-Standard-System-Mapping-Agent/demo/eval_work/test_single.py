import json, time, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.engine.page_index_engine import PageIndexEngine
from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.rerank_adapter import RerankAdapter
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EngineType

from eval_test_set import init_engines

engines = init_engines()

with open("output/test_set_200_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

item = data[6]
print(f"Testing item 7: {item['product_name']}")

engine_map = {
    "rag": engines["rag"],
    "rag_rerank": engines["rag_rerank"],
    "page_index": engines["page_index"],
    "page_index_force": engines["page_index_force"],
}

for ek, engine in engine_map.items():
    print(f"\n--- Engine: {ek} ---")
    t0 = time.time()
    try:
        result = engine.match(item["product_name"])
        elapsed = time.time() - t0
        print(f"Result: {result.matched_category_id}, conf={result.confidence:.3f}, time={elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR after {elapsed:.1f}s: {e}")

engines["db"].close()
print("\nDone!")