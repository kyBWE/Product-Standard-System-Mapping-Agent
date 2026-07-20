import json

with open('output/eval_2engine_v3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

details = data['details']

rag_rerank_no_match = []
rag_rerank_wrong = []
page_index_wrong = []
both_wrong = []
either_correct = 0

for d in details:
    gt = d['ground_truth']
    rr = d['engines']['rag_rerank']
    pi = d['engines']['page_index']
    
    rr_hit = rr.get('hit', False)
    pi_hit = pi.get('hit', False)
    
    if rr_hit or pi_hit:
        either_correct += 1
    
    if rr['predicted'] is None:
        rag_rerank_no_match.append(d)
    elif not rr_hit:
        rag_rerank_wrong.append(d)
    
    if not pi_hit:
        page_index_wrong.append(d)
    
    if not rr_hit and not pi_hit:
        both_wrong.append(d)

print(f"RAG+Rerank: no_match={len(rag_rerank_no_match)}, wrong={len(rag_rerank_wrong)}, correct={159-len(rag_rerank_no_match)-len(rag_rerank_wrong)}")
print(f"PageIndex: wrong={len(page_index_wrong)}, correct={159-len(page_index_wrong)}")
print(f"Both wrong: {len(both_wrong)}")
print(f"Either correct (fusion upper bound): {either_correct}/{159} = {either_correct/159:.1%}")
print()

print("=== RAG+Rerank No-Match (top 20) ===")
for d in rag_rerank_no_match[:20]:
    rr = d['engines']['rag_rerank']
    pi = d['engines']['page_index']
    pi_hit = "✓" if pi.get('hit') else pi.get('predicted', 'NONE')[:15]
    print(f"  {d['product_name'][:25]:<25} GT={d['ground_truth']:<6} conf={rr['confidence']:.3f} PI={pi_hit}")

print()
print("=== Both Wrong (top 20) ===")
for d in both_wrong[:20]:
    rr = d['engines']['rag_rerank']
    pi = d['engines']['page_index']
    print(f"  {d['product_name'][:25]:<25} GT={d['ground_truth']:<6} RR={rr.get('predicted','NONE') or 'NONE':<8} PI={pi.get('predicted','NONE') or 'NONE':<8}")

print()
print("=== RAG+Rerank confidence distribution ===")
confs = [d['engines']['rag_rerank']['confidence'] for d in details]
confs_sorted = sorted(confs)
for p in [10, 25, 50, 75, 90]:
    idx = int(len(confs_sorted) * p / 100)
    print(f"  P{p}: {confs_sorted[idx]:.3f}")

print()
print("=== RAG+Rerank no_match confidence ===")
nm_confs = [d['engines']['rag_rerank']['confidence'] for d in rag_rerank_no_match]
if nm_confs:
    print(f"  min={min(nm_confs):.3f} max={max(nm_confs):.3f} avg={sum(nm_confs)/len(nm_confs):.3f}")
    low_conf = [c for c in nm_confs if c < 0.3]
    print(f"  <0.3: {len(low_conf)}/{len(nm_confs)}")