from __future__ import annotations
import logging
import time

import numpy as np
import requests

logger = logging.getLogger("ApiEmbedder")

BGE_M3_DIM = 1024


class ApiEmbedder:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.modelarts-maas.com/v1/embeddings",
        model: str = "bge-m3",
        embedding_dim: int = BGE_M3_DIM,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 16,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._embedding_dim = embedding_dim
        self._max_retries = max_retries
        self._timeout = timeout
        self._batch_size = batch_size

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "input": texts,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                embeddings = []
                for item in data.get("data", []):
                    embeddings.append(item.get("embedding", []))

                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"返回向量数({len(embeddings)})与输入数({len(texts)})不匹配"
                    )

                return embeddings

            except Exception as e:
                logger.warning(
                    f"Embedding API调用失败(第{attempt}/{self._max_retries}次): {e}"
                )
                if attempt < self._max_retries:
                    time.sleep(min(attempt * 2, 10))
                else:
                    raise

    def _normalize(self, vec: list[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def embed(self, text: str) -> np.ndarray:
        results = self._call_api([text])
        return self._normalize(results[0])

    def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[np.ndarray]:
        bs = batch_size or self._batch_size
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            try:
                raw_embeddings = self._call_api(chunk)
            except Exception as e:
                logger.warning(f"批量Embedding失败(起始索引{i}): {e}, 使用零向量填充")
                raw_embeddings = [[0.0] * self._embedding_dim] * len(chunk)

            for j, raw in enumerate(raw_embeddings):
                if not raw or len(raw) != self._embedding_dim:
                    logger.warning(
                        f"向量维度异常: 期望{self._embedding_dim}, 实际{len(raw)}, 使用零向量"
                    )
                    all_embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                else:
                    all_embeddings.append(self._normalize(raw))

            if (i + bs) % 1000 < bs:
                logger.info(f"Embedding进度: {min(i + bs, len(texts))}/{len(texts)}")

        return all_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim