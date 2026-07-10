"""诊断树结构断裂：找出所有缺失父节点的节点，分析修复方案。"""
from __future__ import annotations
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data.excel_reader import ExcelDataReader
from src.models.category_node import CategoryNode

reader = ExcelDataReader()
nodes, _ = reader.load_standard_system("产品标准体系.xlsx")

node_map = {n.category_id: n for n in nodes}

orphan_count = 0
orphan_details = []
for node in nodes:
    if not node.category_pids:
        continue
    parent_id = node.category_pids[-1]
    if parent_id not in node_map:
        orphan_count += 1
        pids_str = " > ".join(node.category_pids)
        orphan_details.append({
            "category_id": node.category_id,
            "category_name": node.category_name,
            "missing_parent_id": parent_id,
            "pids": pids_str,
            "pids_list": node.category_pids,
        })

print(f"总节点数: {len(nodes)}")
print(f"缺失父节点的节点数: {orphan_count}")
print()

for i, d in enumerate(orphan_details):
    pids = d["pids_list"]
    existing_ancestor = None
    for pid in reversed(pids[:-1]):
        if pid in node_map:
            existing_ancestor = pid
            break
    ancestor_name = node_map[existing_ancestor].category_name if existing_ancestor else "无"
    print(f"{i+1}. {d['category_name']}({d['category_id']}) "
          f"缺失父={d['missing_parent_id']}, "
          f"最近存在祖先={ancestor_name}({existing_ancestor}), "
          f"pids={d['pids']}")