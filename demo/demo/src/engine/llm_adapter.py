from __future__ import annotations
import json
import logging
import time
from typing import Any

from openai import OpenAI

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

    def _call_llm(self, prompt: str, system_prompt: str = "你是一个专业的产品分类分析助手。") -> str:
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
                    return content.strip()
                raise ValueError("LLM返回内容为空")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败(第{attempt + 1}次): {e}")
                time.sleep(1)
        raise RuntimeError(f"LLM调用{self._max_retries}次均失败: {last_error}")

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
                response = self._call_llm(prompt)
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
                response = self._call_llm(prompt)
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
                response = self._call_llm(prompt)
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
                response = self._call_llm(prompt)
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
                response = self._call_llm(prompt)
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
