from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "product_standard_mapping"
    user: str = ""
    password: str = ""
    pool_size: int = 5


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    max_retries: int = 3
    timeout: int = 30
    thinking_enabled: bool = False


@dataclass
class RerankConfig:
    api_key: str = ""
    model: str = "qwen3-rerank"
    instruct: str = (
        "给定企业产品名称，判断其与标准分类名称及同义词的语义相关程度，"
        "用于产品到标准分类体系节点的精确映射匹配。"
    )
    top_n: int = 10
    max_retries: int = 2
    timeout: int = 30


@dataclass
class MatchConfig:
    vector_weight: float = 0.6
    trgm_weight: float = 0.4
    coarse_weight: float = 0.4
    llm_weight: float = 0.6
    coarse_top_k: int = 20
    trgm_threshold: float = 0.3
    syn_confidence_threshold: float = 0.95
    syn_trgm_threshold: float = 0.3
    expand_confidence_threshold: float = 0.3
    low_confidence_threshold: float = 0.5
    enable_llm: bool = False
    enable_rerank: bool = True
    rerank_weight: float = 0.6
    page_index_force_llm_layers: bool = False
