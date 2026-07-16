import requests
import json

BASE = "http://127.0.0.1:5000"

print("=" * 60)
print("1. 查看暂存池")
print("=" * 60)
r = requests.get(f"{BASE}/api/expansion/pool?limit=50")
d = r.json()
print(f"总数: {d['total']}")
for e in d["entries"]:
    pn = e["product_name"]
    parent = e.get("suggested_parent_name", "")
    pid = e.get("suggested_parent_id", "")
    cat = e.get("suggested_category_name", "")
    conf = e.get("confidence", 0)
    pt = e.get("path_text", "")
    print(f"  {pn} | 父={parent}(#{pid}) | 分类={cat} | 置信度={conf} | 路径={pt}")

print()
print("=" * 60)
print("2. 暂存几条新产品")
print("=" * 60)
new_products = ["碳纤维", "航空煤油", "喷气燃料", "无人机", "工业机器人",
                "碳酸锂", "3D打印机", "智能手表", "氢燃料电池", "光伏逆变器",
                "石墨烯", "激光雷达", "柔性显示屏", "锂电池隔膜", "质谱仪"]
for p in new_products:
    r = requests.post(f"{BASE}/api/expansion/stash", json={"product_name": p}, timeout=120)
    d = r.json()
    status = d.get("status", "")
    if status == "already_exists":
        print(f"  {p} -> 已存在")
    elif status == "ok":
        parent = d.get("suggested_parent_name", "")
        pid = d.get("suggested_parent_id", "")
        cat = d.get("suggested_category_name", "")
        conf = d.get("confidence", 0)
        pt = d.get("path_text", "")
        path = d.get("path", [])
        print(f"  {p} -> 父={parent}(#{pid}) 分类={cat} 置信度={conf}")
        print(f"    路径: {pt}")
    else:
        err = d.get("error", "")
        print(f"  {p} -> 错误: {err[:100]}")

print()
print("=" * 60)
print("3. 查看暂存池统计")
print("=" * 60)
r = requests.get(f"{BASE}/api/expansion/pool_stats")
d = r.json()
print(f"总数: {d['total']}")
print(f"上次聚类: {d.get('last_cluster_time', '无')}")
dist = d.get("parent_distribution", {})
for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k}: {v}条")

print()
print("=" * 60)
print("4. 执行聚类分析")
print("=" * 60)
r = requests.post(f"{BASE}/api/expansion/cluster", json={"threshold": 0.65}, timeout=120)
d = r.json()
print(f"状态: {d.get('status', '')}")
print(f"方法: {d.get('cluster_method', '')}")
print(f"总条目: {d.get('total_entries', 0)}")
print(f"簇数: {d.get('cluster_count', 0)}")
print(f"孤立条目: {d.get('outlier_count', 0)}")

for c in d.get("clusters", []):
    cid = c["cluster_id"]
    fp = c.get("full_path", "")
    pn = c.get("product_names", [])
    reason = c.get("llm_reason", "")
    llm = c.get("is_llm_clustered", False)
    star = c.get("star_rating", 0)
    print(f"\n  簇 {cid} [{'★' * star}] {'[LLM]' if llm else ''}")
    print(f"    完整路径: {fp}")
    print(f"    产品: {', '.join(pn)}")
    print(f"    理由: {reason[:100]}")

for o in d.get("outliers", []):
    print(f"\n  孤立: {o['product_name']} 原因={o.get('reason', '')}")

print()
print("=" * 60)
print("5. 测试完成")
print("=" * 60)