import json

d = json.load(open('output/eval_2engine_v3.json', 'r', encoding='utf-8'))
details = d['details']

rr_wrong = [x for x in details if not x['engines']['rag_rerank'].get('hit', False)]
pi_wrong = [x for x in details if not x['engines']['page_index'].get('hit', False)]
both_wrong = [x for x in details if not x['engines']['rag_rerank'].get('hit', False) and not x['engines']['page_index'].get('hit', False)]

print(f"RAG+Rerank wrong: {len(rr_wrong)}")
print(f"PageIndex wrong: {len(pi_wrong)}")
print(f"Both wrong: {len(both_wrong)}")

either = 159 - len(both_wrong)
print(f"Either correct: {either}/159 = {either/159:.1%}")

print("\n=== Both wrong (top 30) ===")
for x in both_wrong[:30]:
    rr_pred = x['engines']['rag_rerank'].get('predicted') or 'NONE'
    pi_pred = x['engines']['page_index'].get('predicted') or 'NONE'
    rr_conf = x['engines']['rag_rerank']['confidence']
    pname = x['product_name'][:22]
    print(f"  {pname:<22} GT={x['ground_truth']:<6} RR={rr_pred:<8}({rr_conf:.2f}) PI={pi_pred:<8}")

print("\n=== RAG+Rerank confidence distribution ===")
all_confs = sorted([x['engines']['rag_rerank']['confidence'] for x in details])
correct_confs = sorted([x['engines']['rag_rerank']['confidence'] for x in details if x['engines']['rag_rerank'].get('hit', False)])
wrong_confs = sorted([x['engines']['rag_rerank']['confidence'] for x in rr_wrong])
for label, confs in [("All", all_confs), ("Correct", correct_confs), ("Wrong", wrong_confs)]:
    if confs:
        p50 = confs[len(confs)//2]
        print(f"  {label}: median={p50:.3f} min={confs[0]:.3f} max={confs[-1]:.3f}")

print("\n=== RR wrong but close (GT sibling/parent) ===")
close_count = 0
for x in rr_wrong:
    rr_pred = x['engines']['rag_rerank'].get('predicted')
    if rr_pred and rr_pred != 'NONE':
        gt = x['ground_truth']
        if gt[:2] == rr_pred[:2] or gt[:3] == rr_pred[:3]:
            close_count += 1
print(f"Same prefix (2-3 chars): {close_count}/{len(rr_wrong)}")