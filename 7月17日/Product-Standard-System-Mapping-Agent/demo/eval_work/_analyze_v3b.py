import json

d = json.load(open('output/eval_2engine_v3.json', 'r', encoding='utf-8'))
either = 0
for item in d['details']:
    rr = item['engines']['rag_rerank'].get('hit', False)
    pi = item['engines']['page_index'].get('hit', False)
    if rr or pi:
        either += 1
print(f'Either correct (fusion upper bound): {either}/159 = {either/159:.1%}')

nm = [x for x in d['details'] if x['engines']['rag_rerank']['predicted'] is None]
print(f'RAG+Rerank no_match: {len(nm)}')
for x in nm:
    pi_hit = 'Y' if x['engines']['page_index'].get('hit') else 'N'
    pname = x['product_name'][:25]
    print(f'  {pname:<25} GT={x["ground_truth"]:<6} PI_hit={pi_hit}')

print()
print('=== Confidence distribution for wrong predictions ===')
wrong_rr = [x for x in d['details'] if not x['engines']['rag_rerank'].get('hit', False) and x['engines']['rag_rerank']['predicted'] is not None]
confs = sorted([x['engines']['rag_rerank']['confidence'] for x in wrong_rr])
if confs:
    for p in [10, 25, 50, 75, 90]:
        idx = int(len(confs) * p / 100)
        print(f'  P{p}: {confs[idx]:.3f}')

print()
print('=== RAG+Rerank wrong but PI correct ===')
rr_wrong_pi_right = [x for x in d['details'] if not x['engines']['rag_rerank'].get('hit', False) and x['engines']['page_index'].get('hit', False)]
print(f'Count: {len(rr_wrong_pi_right)}')
for x in rr_wrong_pi_right[:10]:
    rr_pred = x['engines']['rag_rerank'].get('predicted', 'NONE') or 'NONE'
    pname = x['product_name'][:25]
    print(f'  {pname:<25} GT={x["ground_truth"]:<6} RR={rr_pred:<8} conf={x["engines"]["rag_rerank"]["confidence"]:.3f}')