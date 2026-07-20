import requests
import json
import time

BASE = "http://127.0.0.1:5000"

print("等待服务启动...")
for i in range(30):
    try:
        r = requests.get(f"{BASE}/api/stats", timeout=5)
        if r.status_code == 200:
            print("服务已就绪")
            break
    except Exception:
        time.sleep(2)
else:
    print("服务启动超时")
    exit(1)

print()
print("=" * 60)
print("1. 暂存测试产品")
print("=" * 60)
products = ["碳纤维", "航空煤油", "喷气燃料", "无人机", "工业机器人",
            "碳酸锂", "3D打印机", "智能手表", "氢燃料电池", "光伏逆变器",
            "石墨烯", "激光雷达", "柔性显示屏", "锂电池隔膜", "质谱仪"]

for p in products:
    try:
        r = requests.post(f"{BASE}/api/expansion/stash", json={"product_name": p}, timeout=120)
        d = r.json()
        if d.get("status") == "already_exists":
            print(f"  {p} -> 已存在")
        elif d.get("error"):
            print(f"  {p} -> 错误: {d['error'][:80]}")
        else:
            pt = d.get("path_text", "")
            conf = d.get("confidence", 0)
            parent = d.get("suggested_parent_name", "")
            pid = d.get("suggested_parent_id", "")
            cat = d.get("suggested_category_name", "")
            print(f"  {p} -> 父={parent}(#{pid}) 分类={cat} 置信度={conf}")
            print(f"    路径: {pt}")
    except Exception as e:
        print(f"  {p} -> 异常: {str(e)[:80]}")

print()
print("=" * 60)
print("2. 暂存池统计")
print("=" * 60)
r = requests.get(f"{BASE}/api/expansion/pool_stats")
d = r.json()
print(f"总数: {d['total']}")

print()
print("=" * 60)
print("3. 执行聚类分析")
print("=" * 60)
r = requests.post(f"{BASE}/api/expansion/cluster", json={"threshold": 0.65}, timeout=180)
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
    if reason:
        print(f"    理由: {reason[:150]}")

for o in d.get("outliers", []):
    print(f"\n  孤立: {o['product_name']} 原因={o.get('reason', '')}")

print()
print("=" * 60)
print("4. 测试完成")
print("=" * 60)