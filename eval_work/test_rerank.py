import time
import dashscope
from dashscope import TextReRank
dashscope.api_key = "sk-3c3345cfa6bc4ca093724db842fbe350"
for i in range(5):
    t0 = time.time()
    try:
        resp = TextReRank.call(
            model="qwen3-rerank",
            query=f"测试产品{i}",
            documents=[f"分类A", f"分类B", f"分类C"],
            top_n=3,
            return_documents=False,
        )
        elapsed = time.time() - t0
        print(f"{i}: OK {elapsed:.1f}s status={resp.status_code}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{i}: FAIL {elapsed:.1f}s {e}")