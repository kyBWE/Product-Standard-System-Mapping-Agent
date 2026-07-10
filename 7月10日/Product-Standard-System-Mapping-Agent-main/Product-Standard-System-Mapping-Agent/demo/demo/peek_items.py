import json
with open("output/test_set_200_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)
for i, d in enumerate(data[5:15], start=6):
    print(f"{i}: {d['product_name']}")