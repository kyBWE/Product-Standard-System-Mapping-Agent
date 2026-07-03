from __future__ import annotations
import os
from typing import Any, Optional

import yaml

from src.models.config_models import DBConfig, LLMConfig, MatchConfig, RerankConfig


class ConfigManager:
    _instance: Optional["ConfigManager"] = None

    def __init__(self, config_path: str = "config.yaml"):
        self._config: dict = {}
        self.load(config_path)

    def load(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_db_config(self) -> DBConfig:
        db = self._config.get("database", {})
        return DBConfig(
            host=db.get("host", "localhost"),
            port=db.get("port", 5432),
            database=db.get("database", "product_standard_mapping"),
            user=os.environ.get("DB_USER", db.get("user") or "postgres"),
            password=os.environ.get("DB_PASSWORD", db.get("password") or ""),
            pool_size=db.get("pool_size", 5),
        )

    def get_llm_config(self) -> LLMConfig:
        llm = self._config.get("llm", {})
        return LLMConfig(
            api_key=os.environ.get("LLM_API_KEY", llm.get("api_key", "")),
            base_url=llm.get("base_url", "https://api.openai.com/v1"),
            model=llm.get("model", "gpt-4o"),
            embedding_model=llm.get("embedding_model", "text-embedding-3-small"),
            embedding_dimension=llm.get("embedding_dimension", 1536),
            max_retries=llm.get("max_retries", 3),
            timeout=llm.get("timeout", 30),
            thinking_enabled=llm.get("thinking_enabled", False),
        )

    def get_rerank_config(self) -> RerankConfig:
        rerank = self._config.get("rerank", {})
        return RerankConfig(
            api_key=os.environ.get("DASHSCOPE_API_KEY", rerank.get("api_key", "")),
            model=rerank.get("model", "qwen3-rerank"),
            instruct=rerank.get(
                "instruct",
                "给定企业产品名称，判断其与标准分类名称及同义词的语义相关程度，"
                "用于产品到标准分类体系节点的精确映射匹配。",
            ),
            top_n=rerank.get("top_n", 10),
            max_retries=rerank.get("max_retries", 2),
            timeout=rerank.get("timeout", 30),
        )

    def get_match_config(self) -> MatchConfig:
        match = self._config.get("match", {})
        return MatchConfig(
            vector_weight=match.get("vector_weight", 0.6),
            trgm_weight=match.get("trgm_weight", 0.4),
            coarse_weight=match.get("coarse_weight", 0.4),
            llm_weight=match.get("llm_weight", 0.6),
            coarse_top_k=match.get("coarse_top_k", 20),
            trgm_threshold=match.get("trgm_threshold", 0.3),
            syn_confidence_threshold=match.get("syn_confidence_threshold", 0.95),
            syn_trgm_threshold=match.get("syn_trgm_threshold", 0.3),
            expand_confidence_threshold=match.get("expand_confidence_threshold", 0.3),
            low_confidence_threshold=match.get("low_confidence_threshold", 0.5),
            enable_llm=match.get("enable_llm", False),
            enable_rerank=match.get("enable_rerank", True),
            rerank_weight=match.get("rerank_weight", match.get("llm_weight", 0.6)),
            page_index_force_llm_layers=match.get("page_index_force_llm_layers", False),
        )
