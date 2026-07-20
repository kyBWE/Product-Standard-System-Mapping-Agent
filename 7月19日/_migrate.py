import json
import urllib.request

with open(r"E:\Code\Projects\7月14日\Product-Standard-System-Mapping-Agent\demo\eval_work\data\pending_pool.json", "r", encoding="utf-8") as f:
    pool = json.load(f)

entries = pool.get("entries", [])
print(f"Migrating {len(entries)} entries...")

results = []
for e in entries:
    product_name = e["product_name"]
    try:
        match_data = json.dumps({"product_name": product_name, "engine": "rag_rerank"}).encode("utf-8")
        match_req = urllib.request.Request("http://localhost:5000/api/match", data=match_data, headers={"Content-Type": "application/json"})
        match_r = urllib.request.urlopen(match_req, timeout=60)
        match_result = json.loads(match_r.read().decode("utf-8"))

        category_id = ""
        category_name = ""
        if match_result.get("candidates") and len(match_result["candidates"]) > 0:
            category_id = match_result["candidates"][0]["category_id"]
            category_name = match_result["candidates"][0].get("category_name", "")

        stash_data = json.dumps({"product_name": product_name, "category_id": category_id}).encode("utf-8")
        stash_req = urllib.request.Request("http://localhost:5000/api/expansion/stash", data=stash_data, headers={"Content-Type": "application/json"})
        stash_r = urllib.request.urlopen(stash_req, timeout=30)
        stash_result = json.loads(stash_r.read().decode("utf-8"))

        status = stash_result.get("status", "?")
        msg = stash_result.get("message", stash_result.get("error", ""))
        results.append({"product_name": product_name, "category_id": category_id, "category_name": category_name, "status": status})
        print(f"  {product_name} -> #{category_id}({category_name}) [{status}]")
    except Exception as ex:
        results.append({"product_name": product_name, "category_id": "", "category_name": "", "status": "error", "error": str(ex)})
        print(f"  {product_name} -> FAILED: {ex}")

print(f"\nMigration complete: {sum(1 for r in results if r['status'] == 'ok')} ok, {sum(1 for r in results if r['status'] != 'ok')} other")

pool["entries"] = []
with open(r"E:\Code\Projects\7月14日\Product-Standard-System-Mapping-Agent\demo\eval_work\data\pending_pool.json", "w", encoding="utf-8") as f:
    json.dump(pool, f, ensure_ascii=False, indent=2)
print("Pending pool cleared.")