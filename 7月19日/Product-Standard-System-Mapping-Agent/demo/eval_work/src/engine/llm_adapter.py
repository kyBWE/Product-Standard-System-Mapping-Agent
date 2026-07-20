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
        self,
        product_name: str,
        root_nodes: list[tuple[str, str]],
        *,
        level_hint: str = "一级分类",
        current_path: str = "",
    ) -> dict:
        root_lines = [f"{i+1}. #{rid} {rname}" for i, (rid, rname) in enumerate(root_nodes)]
        root_text = "\n".join(root_lines)

        path_context = ""
        if current_path:
            path_context = f"\n当前已确定的路径：{current_path}\n"

        prompt = f"""请分析以下产品在标准分类体系中应归属的位置。

产品名称：{product_name}
{path_context}
标准体系{level_hint}候选：
{root_text}

请从候选中选择最相关的分类节点，继续深入定位。
规则：
1. 若候选中有语义精确匹配的节点，选择它继续下钻
2. 若候选中只有粗略匹配，选择最接近的节点继续下钻
3. 若候选中没有合适的，说明该产品需要在当前层级下新增分类

然后规划从当前层级到该产品的完整新增路径。路径深度根据产品特性决定：
- 宽泛产品（如"汽油"）：路径较浅，1-2层即可
- 专业产品（如"碳纤维增强复合材料"）：路径较深，需要3-5层逐级细化
- 每一层新增分类名应是对上一层的合理细分

请以JSON格式返回：
{{"root_category_id": "所选候选节点ID", "root_category_name": "所选候选节点名称", "suggested_category_name": "建议新增的分类名称", "suggested_path": ["第一层新增名", "第二层新增名", "最终分类名"], "reason": "归类理由及路径规划理由", "confidence": 0.8}}

注意：suggested_path 是从所选候选节点往下需要新增的完整层级路径，suggested_category_name 是 suggested_path 的最后一项。如果不需要新增中间层级，suggested_path 只有一项。"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="detailed_category_analysis")
                result = self._parse_json_response(response)

                suggested_path = result.get("suggested_path", [])
                if isinstance(suggested_path, str):
                    suggested_path = [s.strip() for s in suggested_path.split(">") if s.strip()]
                if not suggested_path:
                    suggested_path = [str(result.get("suggested_category_name", product_name))]

                return {
                    "root_category_id": str(result.get("root_category_id", "")),
                    "root_category_name": str(result.get("root_category_name", "")),
                    "suggested_category_name": suggested_path[-1] if suggested_path else str(result.get("suggested_category_name", product_name)),
                    "suggested_path": suggested_path,
                    "reason": str(result.get("reason", "")),
                    "confidence": float(result.get("confidence", 0)),
                }
            except Exception as e:
                logger.warning(f"详细品类分析失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return {"root_category_id": "", "root_category_name": "", "suggested_category_name": product_name, "suggested_path": [product_name], "reason": f"分析失败: {e}", "confidence": 0}
        return {"root_category_id": "", "root_category_name": "", "suggested_category_name": product_name, "suggested_path": [product_name], "reason": "未知错误", "confidence": 0}

    def layer_pick_or_stop(
        self,
        product_name: str,
        candidates: list[str],
        ancestry: list[str] | None = None,
    ) -> tuple[str | None, float, str]:
        """扩展定位专用：在当前层选子节点，或判定都不合适则停在本层。

        返回 (选中的候选名或 None 表示停止, confidence, reason)。
        """
        if not candidates:
            return None, 0.0, "无子节点"
        ancestry = ancestry or []
        path_text = " > ".join(ancestry) if ancestry else "（根层）"
        candidates_text = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])
        prompt = f"""你在做标准分类体系的「扩展挂载」：产品尚无精确匹配，需要自上而下找到合适的已有父节点。

产品名称：{product_name}
当前已走过的路径：{path_text}
当前层候选子节点：
{candidates_text}

规则：
1. 只能从候选列表中选择；返回的 selected_name 必须与列表中某一项**完全一致**。
2. 若某个子节点领域/用途能覆盖该产品，选最合适的一个。
3. 若所有候选领域都不对，必须停止：selected_index=0，selected_name=""。
4. 不要被字面「雾/烟/喷/饼/盐」误导，优先看行业真实用途：
   - 工地/矿山降尘雾炮 → 不要进农业植保，也不要仅因「喷水」就进水利机械
   - 手用丝锥扳手 → 优先通用手工具/钻孔攻丝工具，不要进风动电动工具零件
   - 核工业黄饼(重铀酸铵) → 矿产品/铀相关，绝不要进食品
5. 宁可停在较宽的正确父节点，也不要钻进错误细类或「其他…」兜底类。
6. selected_index 与 selected_name 必须指向同一项（1..N 对应列表序号）。

请以JSON返回：
{{"selected_index": 0, "selected_name": "", "confidence": 0.8, "reason": "说明"}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="layer_pick_or_stop")
                result = self._parse_json_response(response)
                idx = int(result.get("selected_index", 0) or 0)
                conf = float(result.get("confidence", 0.6) or 0.6)
                reason = str(result.get("reason", ""))
                selected_name = str(result.get("selected_name") or "").strip()

                # 优先用精确名称对齐
                if selected_name in candidates:
                    return selected_name, conf, reason
                if idx == 0:
                    return None, conf, reason or "候选均不适合，停在当前层"
                if 1 <= idx <= len(candidates):
                    return candidates[idx - 1], conf, reason
                return None, conf, reason or "无效序号，停在当前层"
            except Exception as e:
                logger.warning(f"扩展逐层选择失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return None, 0.0, f"选择失败: {e}"
        return None, 0.0, "未知错误"

    def suggest_formal_category_name(
        self,
        product_name: str,
        parent_name: str,
        parent_path: str,
        sibling_names: list[str] | None = None,
    ) -> dict:
        """在已确定的父节点下，建议正式标准子分类名（禁止产品级细叶子）。"""
        siblings = "、".join((sibling_names or [])[:20]) or "（无）"
        prompt = f"""在标准产品分类体系中，已确定父节点，请为产品建议一个正式的标准子分类名。

产品名称：{product_name}
父节点：{parent_name}
父节点路径：{parent_path}
同级已有子类参考：{siblings}

规则：
1. 新分类名必须正式、通用，能覆盖一类产品，禁止使用产品名原样或型号级名称（如禁止「T型丝锥扳手」「雾炮机」这种SKU名）。
2. 应与同级子类风格相近（…机械/…设备/…工具/…制品 等）。
3. 若父节点下已有足够合适的子类名，可返回 already_exists=true 并给出该已有名。

JSON返回：
{{"new_node_name": "正式分类名", "already_exists": false, "confidence": 0.75, "reason": "..."}}"""
        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="suggest_formal_category_name")
                result = self._parse_json_response(response)
                return {
                    "new_node_name": str(result.get("new_node_name") or "").strip(),
                    "already_exists": bool(result.get("already_exists", False)),
                    "confidence": float(result.get("confidence") or 0.7),
                    "reason": str(result.get("reason") or ""),
                }
            except Exception as e:
                logger.warning(f"正式类名建议失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return {
                        "new_node_name": "",
                        "already_exists": False,
                        "confidence": 0.0,
                        "reason": f"失败: {e}",
                    }
        return {"new_node_name": "", "already_exists": False, "confidence": 0.0, "reason": "未知错误"}

    def judge_mapping_reasonable(
        self,
        product_name: str,
        predicted_name: str,
        predicted_path: str,
        gt_name: str = "",
        gt_path: str = "",
    ) -> tuple[bool, float, str]:
        """评测兜底：判断「预测分类路径」对企业产品名是否合理（可与 GT 不同支/同义）。

        返回 (reasonable, confidence, reason)。
        """
        gt_block = ""
        if gt_name or gt_path:
            gt_block = f"""
参考标准答案（仅供对照，预测走另一合理分支也可判合理）：
- 标准分类：{gt_name or "未知"}
- 分类路径：{gt_path or "未知"}
"""
        prompt = f"""请判断：将企业产品映射到「预测分类」是否合理。

企业产品名称：{product_name}

系统预测：
- 分类名称：{predicted_name}
- 分类路径：{predicted_path or "未知"}
{gt_block}
判定标准：
1. 若预测分类在业务上可表示该产品（同义、材料/制品双视角、设备树与产业链双树、父子/近邻），判合理；
2. 仅名称碰巧相似但品类明显错误（如黄饼→黄酒、弯头→牙科手机），判不合理；
3. 缩写/俗称无法对上预测语义时，判不合理。

请以JSON格式返回：
{{"reasonable": true, "confidence": 0.85, "reason": "一句话说明"}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(
                    prompt,
                    system_prompt=(
                        "你是产品标准分类评测员。关注预测映射是否业务合理，"
                        "不要求与参考答案路径完全一致。"
                    ),
                    method="eval_mapping_reasonable",
                )
                result = self._parse_json_response(response)
                reasonable = bool(result.get("reasonable", False))
                confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
                reason = str(result.get("reason", ""))
                return reasonable, confidence, reason
            except Exception as e:
                logger.warning(f"映射合理性判定失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return False, 0.0, f"判定失败: {e}"
        return False, 0.0, "未知错误"

    def batch_judge_mapping_reasonable(
        self,
        items: list[dict],
    ) -> list[tuple[bool, float, str]]:
        """批量评测兜底：一次给多条，LLM 一次性返回所有判定结果。

        items: [{"product_name", "predicted_name", "predicted_path", "gt_name", "gt_path"}, ...]
        返回: [(reasonable, confidence, reason), ...]
        """
        if not items:
            return []

        lines = []
        for i, item in enumerate(items, start=1):
            gt_line = ""
            if item.get("gt_name") or item.get("gt_path"):
                gt_line = f"  参考答案：{item.get('gt_name', '')} ({item.get('gt_path', '')})"
            lines.append(
                f'{i}. 产品：{item["product_name"]} → 预测：{item["predicted_name"]} '
                f'({item.get("predicted_path", "")}){gt_line}'
            )
        items_text = "\n".join(lines)

        prompt = f"""请批量判断以下企业产品映射到「预测分类」是否合理。

{items_text}

判定标准：
1. 若预测分类在业务上可表示该产品（同义、材料/制品双视角、设备树与产业链双树、父子/近邻），判合理；
2. 仅名称碰巧相似但品类明显错误（如黄饼→黄酒、弯头→牙科手机），判不合理；
3. 缩写/俗称无法对上预测语义时，判不合理。

请以JSON数组格式返回，每项对应一条：
[{{"index": 1, "reasonable": true, "confidence": 0.85, "reason": "一句话说明"}}, ...]"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(
                    prompt,
                    system_prompt=(
                        "你是产品标准分类评测员。关注预测映射是否业务合理，"
                        "不要求与参考答案路径完全一致。严格按JSON数组格式返回。"
                    ),
                    method="batch_eval_mapping_reasonable",
                )
                result = self._parse_json_response(response)
                if isinstance(result, dict):
                    result = result.get("results", result.get("items", [result]))
                if not isinstance(result, list):
                    result = [result]

                results_map: dict[int, tuple[bool, float, str]] = {}
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index", 0)) - 1
                    if idx < 0:
                        continue
                    reasonable = bool(item.get("reasonable", False))
                    confidence = max(0.0, min(1.0, float(item.get("confidence", 0))))
                    reason = str(item.get("reason", ""))
                    results_map[idx] = (reasonable, confidence, reason)

                output: list[tuple[bool, float, str]] = []
                for i in range(len(items)):
                    if i in results_map:
                        output.append(results_map[i])
                    else:
                        output.append((False, 0.0, "批量判定未返回"))
                return output
            except Exception as e:
                logger.warning(f"批量映射合理性判定失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return [(False, 0.0, f"批量判定失败: {e}")] * len(items)
        return [(False, 0.0, "未知错误")] * len(items)

    def suggest_free_path(
        self,
        product_name: str,
        taxonomy_overview: str = "",
    ) -> dict:
        taxonomy_hint = ""
        if taxonomy_overview:
            taxonomy_hint = f"\n当前标准体系一级分类概览：\n{taxonomy_overview}\n"

        prompt = f"""你是一个标准分类体系专家。以下产品无法匹配到现有标准分类，请为它规划一条分类路径。

规则：
1. 从最大的产品大类开始，逐级细分到该产品应归属的**分类节点**，而非产品名本身
2. 路径的最后一层是一个"分类"，该分类下可以包含此产品及同类产品，而不是只包含这一个产品
3. 路径深度根据产品特性决定：宽泛产品2-3层，专业产品4-6层
4. 每一层是对上一层的合理细分，层级之间用 > 分隔
5. 路径不受现有标准体系约束，可以自由规划
6. 不要把产品名本身作为路径的最后一层，最后一层应该是能涵盖该产品的分类名
{taxonomy_hint}
产品名称：{product_name}

请以JSON格式返回：
{{"full_path": "大类 > 中类 > 小类 > ... > 分类名", "reason": "路径规划理由", "confidence": 0.8}}

示例：
- 碳纤维 → {{"full_path": "石油、化工、医药产品 > 化学原料及化学制品 > 专项化学用品 > 高功能化工产品", "reason": "碳纤维属于高性能纤维材料，归入高功能化工产品分类", "confidence": 0.72}}
- 航空煤油 → {{"full_path": "石油、化工、医药产品 > 石油产品 > 燃料油 > 航空燃料", "reason": "航空煤油归入航空燃料分类", "confidence": 0.78}}
- 工业机器人 → {{"full_path": "机械、设备产品 > 通用设备 > 自动化设备", "reason": "工业机器人归入自动化设备分类", "confidence": 0.70}}
- 智能手表 → {{"full_path": "电子信息、仪器仪表产品 > 电子信息产品 > 智能终端设备 > 可穿戴智能设备", "reason": "智能手表归入可穿戴智能设备分类", "confidence": 0.66}}"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="suggest_free_path")
                result = self._parse_json_response(response)

                full_path = str(result.get("full_path", "")).strip()
                if not full_path:
                    full_path = product_name

                path_parts = [p.strip() for p in full_path.split(">") if p.strip()]
                if not path_parts:
                    path_parts = [product_name]

                return {
                    "full_path": full_path,
                    "path_parts": path_parts,
                    "suggested_category_name": path_parts[-1] if path_parts else product_name,
                    "reason": str(result.get("reason", "")),
                    "confidence": float(result.get("confidence", 0)),
                }
            except Exception as e:
                logger.warning(f"自由路径规划失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return {
                        "full_path": product_name,
                        "path_parts": [product_name],
                        "suggested_category_name": product_name,
                        "reason": f"规划失败: {e}",
                        "confidence": 0.0,
                    }
        return {"full_path": product_name, "path_parts": [product_name], "suggested_category_name": product_name, "reason": "未知错误", "confidence": 0.0}

    def cluster_products(
        self,
        product_entries: list[dict],
        taxonomy_overview: str = "",
    ) -> list[dict]:
        if not product_entries:
            return []

        lines = []
        for i, e in enumerate(product_entries):
            pid = e.get("suggested_parent_id", "") or "未确定"
            pname = e.get("suggested_parent_name", "") or "未知"
            cat = e.get("suggested_category_name", "") or "未确定"
            path_text = e.get("path_text", "") or ""
            path_hint = f"，推理路径: {path_text}" if path_text else ""
            lines.append(f"{i+1}. {e['product_name']}（建议父节点: {pname}(#{pid}), 建议分类: {cat}{path_hint}）")

        product_list = "\n".join(lines)

        taxonomy_hint = ""
        if taxonomy_overview:
            taxonomy_hint = f"\n当前标准体系一级分类概览：\n{taxonomy_overview}\n"

        prompt = f"""你是一个标准分类体系专家。以下是一批未能匹配到现有标准分类的产品，每条都附带了之前LLM推理的建议路径。

请将这些产品进行**精细**分组聚类。核心目标：宁可多出孤立条目，也不要把不相关产品硬塞进同一个宽泛大类。

强制规则：
1. 只有「产业用途相近、可共用同一细分类名」的产品才能进同一簇（例如同属「中央空调设备」或同属「工控主板」）
2. 禁止仅因同属一级大类（如「机械、设备类产品」）就合并；一级大类相同但细分领域不同的必须拆开或列为 outliers
3. 反例（绝对不要并成一簇）：燃气轮机、电磁灶、列车制动配件、中央空调、工控主板、飞机光电球罩 —— 这些虽都可能挂在机械设备大类下，但细分完全不同
4. 每个簇的 full_path 至少 3 级（大类 > 中类 > 细类），鼓励 4 级；禁止 full_path 只有「一级大类」或「一级大类 > 模糊总称」
5. group_name 必须是可落地的细分类名，禁止使用「其他设备」「机械设备」「综合产品」等空泛名称
6. 同一簇产品应能回答：它们是否可被同一采购目录/同一质检标准覆盖？不能则拆分
7. 拿不准就放 outliers，不要凑簇

{taxonomy_hint}
待聚类产品列表：
{product_list}

请以JSON格式返回：
{{
  "clusters": [
    {{
      "group_name": "细分类名（具体、可落地）",
      "parent_id": "挂载父节点ID（尽量用已有较深节点，不要用一级根）",
      "parent_name": "挂载父节点名称",
      "full_path": "一级 > 二级 > 三级 > 细类（至少3级）",
      "product_indices": [1, 3],
      "reason": "说明为何这些产品同属该细类，以及为何路径足够细"
    }}
  ],
  "outliers": [2]
}}

注意：
- product_indices / outliers 序号从1开始
- 若本批产品彼此都不相近，clusters 可以为空，全部放 outliers
- full_path 必须逐级细化，末级名称应与 group_name 一致或接近"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="cluster_products")
                result = self._parse_json_response(response)

                clusters_out = []
                for c in result.get("clusters", []):
                    indices = c.get("product_indices", [])
                    valid_indices = [i - 1 for i in indices if 1 <= i <= len(product_entries)]
                    if not valid_indices:
                        continue

                    matched_entries = [product_entries[i] for i in valid_indices]
                    product_names = [e["product_name"] for e in matched_entries]
                    entry_ids = [e["id"] for e in matched_entries]

                    parent_id = str(c.get("parent_id", "")).strip().lstrip("#")
                    if not parent_id or not parent_id.isdigit():
                        parent_id = matched_entries[0].get("suggested_parent_id", "") if matched_entries else ""

                    parent_name = str(c.get("parent_name", "")).strip()
                    if not parent_name:
                        pn = matched_entries[0].get("suggested_parent_name", "") if matched_entries else ""
                        parent_name = pn

                    confidences = [e.get("confidence", 0.0) for e in matched_entries]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

                    clusters_out.append({
                        "suggested_parent_id": parent_id,
                        "suggested_parent_name": parent_name,
                        "suggested_category_name": str(c.get("group_name", "")),
                        "merged_category_name": str(c.get("group_name", "")),
                        "full_path": str(c.get("full_path", "")),
                        "entry_count": len(matched_entries),
                        "avg_confidence": round(avg_conf, 4),
                        "confidence_variance": 0.0,
                        "star_rating": 3 if len(matched_entries) >= 5 else (2 if len(matched_entries) >= 3 else 1),
                        "has_divergence": False,
                        "entries": entry_ids,
                        "product_names": product_names,
                        "llm_reason": str(c.get("reason", "")),
                        "is_llm_clustered": True,
                    })

                outlier_indices = [i - 1 for i in result.get("outliers", []) if 1 <= i <= len(product_entries)]
                outliers_out = []
                for i in outlier_indices:
                    e = product_entries[i]
                    outliers_out.append({
                        "entry_id": e["id"],
                        "product_name": e["product_name"],
                        "suggested_parent_id": e.get("suggested_parent_id", ""),
                        "suggested_category_name": e.get("suggested_category_name", ""),
                        "reason": "LLM判定无法归类",
                    })

                return clusters_out, outliers_out

            except Exception as e:
                logger.warning(f"LLM聚类失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return [], []
        
        return [], []

    def batch_path_reasoning(
        self,
        products: list[dict],
        taxonomy_overview: str = "",
    ) -> list[dict]:
        """批量路径推理：一次LLM调用分析多个产品，提升性能
        
        Args:
            products: 产品列表，每个元素包含 {product_name, vector_candidates}
            taxonomy_overview: 标准体系一级分类概览
            
        Returns:
            推理结果列表，每个元素对应一个产品
        """
        if not products or not self._client:
            return [None] * len(products)
        
        products_info = ""
        for idx, p in enumerate(products, 1):
            product_name = p.get("product_name", "")
            vector_candidates = p.get("vector_candidates", [])
            
            candidate_info = ""
            for i, c in enumerate(vector_candidates[:5], 1):
                path = c.get("path", "")
                path_display = path.replace(",", " > ") if path else c.get("category_name", "")
                candidate_info += f"  {i}. #{c.get('category_id', '')} {c.get('category_name', '')} (相似度{c.get('similarity', 0):.2f})\n     路径: {path_display}\n"
            
            products_info += f"""
【产品{idx}】{product_name}
向量语义匹配找到的相似节点:
{candidate_info if candidate_info else "  无相似节点"}
"""
        
        prompt = f"""你是一个标准分类体系专家。现在有多个产品无法匹配到现有标准分类，需要你批量推理它们应该放在什么位置。

标准体系一级分类概览:
{taxonomy_overview}

待分析产品列表:
{products_info}

请对每个产品进行分析，推理它在标准体系中的合理位置。要求:
1. 分析产品的本质属性和用途
2. 判断它属于哪个一级分类
3. 在该一级分类下，找到最合适的父节点
4. 如果是新兴产品或现有分类不够精确，判断是否需要创建新的子分类
5. 给出推理理由

**重要提示**：
- suggested_parent_id必须从每个产品的向量候选中选择一个实际存在的节点ID（如#21447）
- 如果你认为应该放在一个不存在的分类下（如"其他有机化学原料"），请设置should_create_new_node=true
- 此时suggested_parent_id应该是新节点的父节点ID（从向量候选中选择），new_node_name填写要创建的新节点名称
- 不要返回节点名称作为suggested_parent_id，必须是数字ID或留空

请以JSON格式返回:
{{
  "results": [
    {{
      "product_name": "产品名称",
      "product_analysis": "对产品特征的分析",
      "primary_category": "产品属于的一级分类名称",
      "suggested_parent_id": "推荐的父节点ID（必须从向量候选中选择，如#21447）",
      "suggested_parent_name": "推荐的父节点名称",
      "should_create_new_node": true/false,
      "new_node_name": "如果需要新建，新节点的名称（如'其他有机化学原料'）",
      "full_path": "完整路径",
      "reasoning": "推理理由",
      "confidence": 0.0-1.0
    }}
  ]
}}

注意：results数组的顺序必须与输入产品列表的顺序一致！"""

        for attempt in range(self._max_retries):
            try:
                response = self._call_llm(prompt, method="batch_path_reasoning")
                result = self._parse_json_response(response)
                results = result.get("results", [])
                
                if len(results) != len(products):
                    logger.warning(f"批量推理结果数量不匹配: 期望{len(products)}, 实际{len(results)}")
                    if attempt == self._max_retries - 1:
                        return [None] * len(products)
                    continue
                
                return results
                
            except Exception as e:
                logger.warning(f"批量路径推理失败(第{attempt + 1}次): {e}")
                if attempt == self._max_retries - 1:
                    return [None] * len(products)
        
        return [None] * len(products)

        return [], []
