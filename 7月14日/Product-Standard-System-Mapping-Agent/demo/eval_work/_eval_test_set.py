"""
评测脚本：用test_set_llm_200.json跑一遍RAG引擎，统计粗召回召回率与最终准确率
"""
from src.models.enums import EngineType
from src.engine.rerank_adapter import RerankAdapter
from src.engine.rag_match_engine import RAGMatchEngine
from src.index.vector_index_manager import VectorIndexManager
from src.index.trgm_index_manager import TrgmIndexManager
from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.config_manager import ConfigManager
import json
import time
import logging
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.WARNING,
                    format='%(name)s - %(levelname)s - %(message)s')


# ---- 初始化 ----
config = ConfigManager('config.yaml')
db_config = config.get_db_config()
llm_config = config.get_llm_config()
match_config = config.get_match_config()
embedding_config = config.get_embedding_config()
rerank_config = config.get_rerank_config()

db = DBConnectionManager(db_config)
db.initialize()

llm = LLMAdapter(llm_config)
trgm_mgr = TrgmIndexManager(db)
vec_mgr = VectorIndexManager(
    db,
    embedding_model=llm_config.embedding_model,
    embedding_dimension=llm_config.embedding_dimension,
    base_url=llm_config.base_url,
    api_key=llm_config.api_key,
    embedding_config=embedding_config,
)

pg_ok = vec_mgr.ensure_pgvector_ready()
vec_mgr.warmup()
print(
    f"pgvector: {pg_ok}, 向量数: {len(vec_mgr._category_ids) if vec_mgr._category_ids else 0}")

# ---- 创建两个引擎 ----
# 1. RAG纯向量引擎(无LLM精排) - 用于观察粗召回
rag_coarse = RAGMatchEngine(
    vec_mgr, trgm_mgr, llm, match_config,
    enable_llm=False,
    fine_match_mode="llm",
    engine_type=EngineType.RAG_VECTOR,
)

# 2. RAG+Rerank引擎(完整流程) - 用于观察最终准确率
rerank_adapter = RerankAdapter(
    rerank_config) if rerank_config.api_key else None
rag_full = RAGMatchEngine(
    vec_mgr, trgm_mgr, llm, match_config,
    enable_llm=match_config.enable_llm,
    rerank=rerank_adapter,
    fine_match_mode="rerank",
    engine_type=EngineType.RAG_RERANK,
)

# ---- 加载测试集 ----
test_file = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', '..', 'test_set_llm_200.json')
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"测试集: {len(test_data)} 条")

# ---- 评测 ----
results = []
coarse_hit = 0   # GT在粗召回Top-50中
final_hit = 0    # 最终匹配==GT
no_match = 0
llm_skip = 0
llm_used = 0
errors = 0

# 只跑前30条做快速评测(完整198条太慢)
MAX_ITEMS = 30
test_subset = test_data[:MAX_ITEMS]

for i, item in enumerate(test_subset):
    pn = item['product_name']
    gt_id = str(item['ground_truth'])
    gt_name = item['ground_truth_name']

    try:
        # 1. 粗召回评测(无LLM)
        coarse_result = rag_coarse.match(pn)
        coarse_ids = [c.category_id for c in coarse_result.candidates]
        coarse_hit_flag = gt_id in coarse_ids
        if coarse_hit_flag:
            coarse_hit += 1

        # 2. 完整流程评测(含Rerank+LLM)
        full_result = rag_full.match(pn)
        final_id = str(
            full_result.matched_category_id) if full_result.matched_category_id else None
        final_hit_flag = (final_id == gt_id)
        if final_hit_flag:
            final_hit += 1

        if full_result.match_status.value == 'NO_MATCH':
            no_match += 1
        if not full_result.llm_participated:
            llm_skip += 1
        else:
            llm_used += 1

        # GT在粗召回中的排名
        gt_rank = None
        if coarse_hit_flag:
            gt_rank = coarse_ids.index(gt_id) + 1

        results.append({
            'product_name': pn,
            'gt_id': gt_id,
            'gt_name': gt_name,
            'coarse_hit': coarse_hit_flag,
            'gt_rank_in_coarse': gt_rank,
            'coarse_top1_id': coarse_ids[0] if coarse_ids else None,
            'coarse_top1_name': coarse_result.candidates[0].category_name if coarse_result.candidates else None,
            'coarse_top1_score': coarse_result.candidates[0].coarse_score if coarse_result.candidates else None,
            'final_id': final_id,
            'final_name': full_result.candidates[0].category_name if full_result.candidates else None,
            'final_confidence': full_result.confidence,
            'final_hit': final_hit_flag,
            'match_status': full_result.match_status.value,
            'llm_participated': full_result.llm_participated,
        })

        status = "✓" if final_hit_flag else "✗"
        c_status = "C✓" if coarse_hit_flag else "C✗"
        print(f"[{i+1}/{MAX_ITEMS}] {status} {c_status} {pn} -> GT={gt_id}({gt_name}) | coarse_top1={coarse_ids[0] if coarse_ids else 'N/A'} | final={final_id} | rank={gt_rank}")

    except Exception as e:
        errors += 1
        print(f"[{i+1}/{MAX_ITEMS}] ERROR: {pn} -> {e}")
        results.append({
            'product_name': pn,
            'gt_id': gt_id,
            'gt_name': gt_name,
            'error': str(e),
        })

    time.sleep(0.3)  # 避免API限流

# ---- 输出统计 ----
total = len(test_subset)
print(f"\n{'='*60}")
print(f"评测统计 (前{total}条)")
print(f"{'='*60}")
print(
    f"粗召回召回率 (GT在Top-K中): {coarse_hit}/{total} = {100*coarse_hit/total:.1f}%")
print(
    f"最终准确率 (Top-1 == GT):   {final_hit}/{total} = {100*final_hit/total:.1f}%")
print(
    f"无匹配(NO_MATCH):           {no_match}/{total} = {100*no_match/total:.1f}%")
print(f"LLM参与精排:               {llm_used}/{total}")
print(f"LLM跳过(粗召回高置信):     {llm_skip}/{total}")
print(f"错误:                      {errors}/{total}")

# GT在粗召回中的排名分布
ranks = [r['gt_rank_in_coarse']
         for r in results if r.get('gt_rank_in_coarse') is not None]
if ranks:
    from collections import Counter
    rank_dist = Counter(ranks)
    print(f"\nGT在粗召回中的排名分布:")
    for rank in sorted(rank_dist.keys())[:10]:
        print(f"  Rank {rank}: {rank_dist[rank]}条")

# 保存详细结果
out_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', '..', '_eval_results.json')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n详细结果已保存到: {out_file}")
