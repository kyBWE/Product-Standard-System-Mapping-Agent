import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data.excel_reader import ExcelDataReader
from src.index.page_index_tree import PageIndexTree

reader = ExcelDataReader()
nodes, _ = reader.load_standard_system("产品标准体系.xlsx")
tree = PageIndexTree()
tree.build_tree(nodes)

hits = tree.lookup_index("苹果汁")
if hits:
    for i, h in enumerate(hits[:5]):
        print(f"  [{i}] {h.node.category_name}({h.node.category_id}): "
              f"score={h.score:.3f}, type={h.match_type}, "
              f"path={' > '.join(n.category_name for n in h.path)}")
else:
    print("  lookup_index: 无命中")