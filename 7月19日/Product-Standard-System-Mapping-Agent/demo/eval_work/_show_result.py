import json
d = json.load(open('output/eval_2engine_v3.json', 'r', encoding='utf-8'))
print(json.dumps(d['summary'], ensure_ascii=False, indent=2))
print(f"total: {d['total_items']}")