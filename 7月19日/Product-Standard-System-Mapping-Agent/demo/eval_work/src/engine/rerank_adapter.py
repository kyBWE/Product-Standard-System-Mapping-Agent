from __future__ import annotations
import logging
import time
from http import HTTPStatus

import dashscope
from dashscope import TextReRank

from src.models.config_models import RerankConfig


logger = logging.getLogger("RerankAdapter")


class RerankAdapter:
    def __init__(self, config: RerankConfig):
        self._config = config
        if config.api_key:
            dashscope.api_key = config.api_key

    def rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if not self._config.api_key:
            logger.warning("DashScope API key未配置, 无法调用Rerank")
            return [0.0] * len(documents)

        top_n = min(self._config.top_n or len(documents), len(documents))
        last_error = None
        for attempt in range(self._config.max_retries):
            try:
                resp = TextReRank.call(
                    model=self._config.model,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=False,
                    instruct=self._config.instruct,
                )
                if resp.status_code == HTTPStatus.OK:
                    return self._parse_scores(resp, len(documents))
                last_error = f"status={resp.status_code}, message={getattr(resp, 'message', '')}"
                logger.warning(f"Rerank返回异常(第{attempt + 1}次): {last_error}")
            except Exception as e:
                last_error = e
                logger.warning(f"Rerank调用失败(第{attempt + 1}次): {e}")
            time.sleep(1)

        logger.error(f"Rerank调用{self._config.max_retries}次均失败: {last_error}")
        return [0.0] * len(documents)

    def _parse_scores(self, resp, count: int) -> list[float]:
        scores = [0.0] * count
        output = resp.output
        if output is None:
            return scores

        results = getattr(output, "results", None)
        if results is None and isinstance(output, dict):
            results = output.get("results", [])

        if not results:
            return scores

        for item in results:
            if hasattr(item, "index"):
                idx = int(item.index)
                score = float(item.relevance_score)
            elif isinstance(item, dict):
                idx = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0))
            else:
                continue
            if 0 <= idx < count:
                scores[idx] = max(0.0, min(1.0, score))
        return scores
