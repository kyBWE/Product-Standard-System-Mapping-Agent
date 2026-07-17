import urllib.request, json

# Test log API
req = urllib.request.Request("http://localhost:5000/api/expansion/log?limit=5")
r = urllib.request.urlopen(req, timeout=10)
d = json.loads(r.read().decode("utf-8"))
print(f"Log entries: {d['total']}")
for e in d.get("entries", []):
    print(f"  {e['product_name']} -> {e['match_path']} ({e['created_at'][:16]})")

# Test node_stats with path
print()
req = urllib.request.Request("http://localhost:5000/api/expansion/node_stats")
r = urllib.request.urlopen(req, timeout=10)
d = json.loads(r.read().decode("utf-8"))
for n in d.get("nodes", [])[:3]:
    print(f"  #{n['category_id']} {n['category_name']}: path={n.get('match_path','')} syns={n['expansion_syns']}")