from __future__ import annotations
import json
import logging
import time
from typing import Any

from openai import OpenAI

from src.infrastructure.llm_cache import LLMResponseCache
from src.models.config_models import LLMConfig
from src.models.evolve_models import SynonymVerifyResult, CategoryAnalysisResult


logger = logging.getLogger("LLMAdapter")


class LLMAdapter:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = None
        if config.api_key:
            self._client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        self._max_retries = config.max_retries
        self._cache_enabled = config.cache_enabled
        self._cache = LLMResponseCache(config.cache_size) if config.cache_enabled else None

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str = "你是一个专业的产品分类分析助手。",
        method: str = "chat",
    ) -> str:
        if self._cache_enabled and self._cache is not None:
            cached = self._cache.get(method, prompt, system_prompt)
            if cached is not None:
                logger.debug(f"LLM缓存命中: method={method}")
                return cached

        if self._client is None:
            raise RuntimeError("LLM API key未配置, 无法调用LLM服务")
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    **self._chat_completion_kwargs(prompt, system_prompt)
                )
                content = response.choices[0].message.content
                if content:
                    result = content.strip()
                    if self._cache_enabled and self._cache is not None:
                        self._cache.set(method, result, prompt, system_prompt)
                    return result
                raise ValueError("LLM返回内容为空")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败(第{attempt + 1}次): {e}")
                time.sleep(1)
        raise RuntimeError(f"LLM调用{self._max_retries}次均失败: {last_error}")

    def _chat_completion_kwargs(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        if "deepseek.com" in self._config.base_url:
            thinking_type = "enabled" if self._config.thinking_enabled else "disabled"
            kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
        return kwargs

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"LLM响应JSON解析失败: {e}, response={response[:200]}")

    def semantic_scoring(self, product_name: str, category_name: str, syn_list: list[str]) -> float:
        scores = self.multi_candidate_semantic_scoring(
            product_name,
            [(category_name, category_name, syn_list)],
        )
        return scores[0] if scores else 0.0

    def multi_candidate_semantic_scoring(
        self,
        product_name: str,
        candidates: list[tuple[str, str, list[str]]],
        coarse_hints: list[float] | None = None,
    ) -> list[float]:
        """一次 LLM 调用，对全部候选在同一上下文中打分。tuple: (category_id, category_name, syn_list)。"""
        if not candidates:
            return []

        if len(candidates) == 1:
            _, cat_name, syn_list = candidates[0]
            syn_text = "、".join(syn_list[:8]) if syn_list else "无"
            hint = ""
            if coarse_hints:
                hint = f"\n粗召回融合分：{coarse_hints[0]:.3f}"
            prompt = f"""请判断以下产品名称与标准分类的语义相关程度，返回0到1之间的分数。

产品名称：{product_name}
标准分类名称：{cat_name}
同义词列表：{syn_text}{hint}

请以JSON格式返回：{{"scores": [{{"index": 1, "score": 0.85}}]}}"""
        else:
            lines = []
            for i, (_, cat_name, syn_list) in enumerate(candidates, start=1):
                syn_text = "、".join(syn_list[:8]) if syn_list else "无"
                hint = ""
                if coarse_hints and i - 1 < len(coarse_hints):
                    hint = f"，粗召回分={coarse_hints[i - 1]:.3f}"
                lines.append(f"{i}. 分类名称：{cat_name} | 同义词：{syn_text}{hint}")
            candidates_text = "\n".join(lines)
            prompt = f"""请判断以下产品名称与各个标准分类候选的语义相关程度。
请在同一标准下为每个候选打分（0到1），分数越高表示越匹配；若均不合适，所有分数应偏低。

产品名称：{product_name}

候选列表：
{candidates_text}

请以JSON格式返回：
{{"scores": [{{"index": 1, "score": 0.85}}, {{"index": 2, "score": 0.42}}]}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="multi_candidate_scoring")
                result = self._parse_json_response(response)
                return self._parse_multi_candidate_scores(result, len(candidates))
            except Exception as e:
                logger.warning(f"多候选语义打分失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return [0.0] * len(candidates)
        return [0.0] * len(candidates)

    def _parse_multi_candidate_scores(self, result: dict[str, Any], count: int) -> list[float]:
        scores = [0.0] * count
        raw_scores = result.get("scores")
        if not isinstance(raw_scores, list):
            single = result.get("score")
            if single is not None and count == 1:
                scores[0] = max(0.0, min(1.0, float(single)))
            return scores

        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("index", 0)) - 1
            if 0 <= idx < count:
                scores[idx] = max(0.0, min(1.0, float(item.get("score", 0))))
        return scores

    def select_best_category(
        self,
        product_name: str,
        candidates: list[tuple[str, str, list[str]]],
    ) -> tuple[int | None, float, str]:
        """从粗召回候选中直接择一最匹配分类。返回 (0-based索引或None, 置信度, 原因)。"""
        if not candidates:
            return None, 0.0, "无候选"

        lines = []
        for i, (_, cat_name, syn_list) in enumerate(candidates, start=1):
            syn_text = "、".join(syn_list[:12]) if syn_list else "无"
            lines.append(f"{i}. 分类名称：{cat_name} | 同义词：{syn_text}")
        candidates_text = "\n".join(lines)

        prompt = f"""请将以下企业产品名称映射到最合适的一项标准分类。

企业产品名称：{product_name}

候选标准分类（含同义词）：
{candidates_text}

要求：
1. 综合产品名称、分类名称与同义词判断语义是否指代同一类产品；
2. 只选择一个最匹配的候选；若均不合适，selected_index 填 0；
3. confidence 为你对该映射的确信程度（0到1），无匹配时应偏低。

请以JSON格式返回：
{{"selected_index": 1, "confidence": 0.85, "reason": "简要说明"}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(
                    prompt,
                    system_prompt="你是产品标准分类映射专家，擅长根据产品名称与同义词选择最准确的标准分类节点。",
                    method="rag_category_selection",
                )
                result = self._parse_json_response(response)
                idx_raw = int(result.get("selected_index", 0))
                confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
                reason = str(result.get("reason", ""))
                if idx_raw <= 0:
                    return None, confidence, reason
                idx = idx_raw - 1
                if 0 <= idx < len(candidates):
                    return idx, confidence, reason
                logger.warning(f"LLM返回越界索引: {idx_raw}, 候选数={len(candidates)}")
                return None, 0.0, reason or "索引越界"
            except Exception as e:
                logger.warning(f"分类择一失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return None, 0.0, f"择一失败: {e}"
        return None, 0.0, "未知错误"

    def batch_semantic_scoring(
        self,
        product_name: str,
        candidates: list[tuple[str, str, list[str]]],
    ) -> list[float]:
        if not candidates:
            return []
        payload = [(name, name, syns) for name, _, syns in candidates]
        return self.multi_candidate_semantic_scoring(product_name, payload)

    def keyword_extraction(self, text: str) -> list[str]:
        prompt = f"""请从以下文本中提取关键词，返回JSON格式。

文本：{text}

请以JSON格式返回：{{"keywords": ["关键词1", "关键词2"]}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="keyword_extraction")
                result = self._parse_json_response(response)
                return result.get("keywords", [])
            except Exception as e:
                logger.warning(f"关键词提取失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return []
        return []

    def synonym_verification(self, product_name: str, category_name: str) -> SynonymVerifyResult:
        prompt = f"""请判断以下产品名称与标准分类名称是否属于同义表述（即指代同一类产品）。

产品名称：{product_name}
标准分类名称：{category_name}

请以JSON格式返回：{{"is_synonym": true/false, "confidence": 0.9, "reason": "说明原因"}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="synonym_verification")
                result = self._parse_json_response(response)
                return SynonymVerifyResult(
                    is_synonym=bool(result.get("is_synonym", False)),
                    confidence=float(result.get("confidence", 0)),
                    reason=str(result.get("reason", "")),
                )
            except Exception as e:
                logger.warning(f"同义校验失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return SynonymVerifyResult(is_synonym=False, confidence=0, reason=f"校验失败: {e}")
        return SynonymVerifyResult(is_synonym=False, confidence=0, reason="未知错误")

    def category_analysis(self, product_name: str, categories: list[str] | None = None) -> CategoryAnalysisResult:
        categories_text = "、".join(categories[:30]) if categories else ""
        prompt = f"""请分析以下产品名称所属的品类和属性，给出标准分类体系中的建议位置。

产品名称：{product_name}
{'现有一级分类参考：' + categories_text if categories_text else ''}

请以JSON格式返回：
{{"category_name": "建议分类名称", "parent_category": "建议父分类", "level_position": "建议层级位置", "attributes": {{"属性1": "值1"}}}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="category_analysis")
                result = self._parse_json_response(response)
                return CategoryAnalysisResult(
                    category_name=str(result.get("category_name", "")),
                    parent_category=str(result.get("parent_category", "")),
                    level_position=str(result.get("level_position", "")),
                    attributes=result.get("attributes", {}),
                )
            except Exception as e:
                logger.warning(f"品类分析失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return CategoryAnalysisResult()
        return CategoryAnalysisResult()

    def detailed_category_analysis(
        self, product_name: str, root_nodes: list[tuple[str, str]]
    ) -> dict:
        root_lines = [f"{i+1}. #{rid} {rname}" for i, (rid, rname) in enumerate(root_nodes)]
        root_text = "\n".join(root_lines)
        prompt = f"""请分析以下产品在标准分类体系中应归属的位置。

产品名称：{product_name}

标准体系一级分类：
{root_text}

请先选择最相关的一级分类，然后说明该产品应作为什么新分类插入。

请以JSON格式返回：
{{"root_category_id": "一级分类ID", "root_category_name": "一级分类名称", "suggested_category_name": "建议新增的分类名称", "reason": "归类理由", "confidence": 0.8}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="detailed_category_analysis")
                result = self._parse_json_response(response)
                return {
                    "root_category_id": str(result.get("root_category_id", "")),
                    "root_category_name": str(result.get("root_category_name", "")),
                    "suggested_category_name": str(result.get("suggested_category_name", product_name)),
                    "reason": str(result.get("reason", "")),
                    "confidence": float(result.get("confidence", 0)),
                }
            except Exception as e:
                logger.warning(f"详细品类分析失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return {"root_category_id": "", "root_category_name": "", "suggested_category_name": product_name, "reason": f"分析失败: {e}", "confidence": 0}
        return {"root_category_id": "", "root_category_name": "", "suggested_category_name": product_name, "reason": "未知错误", "confidence": 0}

    def layer_disambiguation(
        self,
        product_name: str,
        candidates: list[str],
        path_hints: list[str] | None = None,
    ) -> str | None:
        if path_hints and len(path_hints) == len(candidates):
            candidates_text = "\n".join(
                [f"{i + 1}. {c}（路径: {p}）" for i, (c, p) in enumerate(zip(candidates, path_hints))]
            )
        else:
            candidates_text = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])
        prompt = f"""在标准分类体系逐层匹配中，以下产品名称在当前层有多个候选子节点，请选择最相关的一个。

产品名称：{product_name}
候选子节点：
{candidates_text}

请以JSON格式返回：{{"selected_index": 1, "reason": "说明原因"}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="layer_disambiguation")
                result = self._parse_json_response(response)
                idx = int(result.get("selected_index", 0)) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
                return None
            except Exception as e:
                logger.warning(f"逐层消歧失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return None
        return None
