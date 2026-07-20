from __future__ import annotations
import logging
from datetime import datetime

from src.data.excel_reader import ExcelDataReader
from src.data.synonym_sanitizer import sanitize_syn_list
from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EvolveActionType, ExpansionStatus, MatchStatus
from src.models.evolve_models import (
    EvolveAction,
    ExpansionAction,
    ExpansionSuggestion,
    SynonymUpdateAction,
    SynonymVerifyResult,
)
from src.models.match_result import MatchResult
from src.models.config_models import MatchConfig


logger = logging.getLogger("SelfEvolveScheduler")


class SelfEvolveScheduler:
    def __init__(
        self,
        llm: LLMAdapter,
        db: DBConnectionManager,
        excel_reader: ExcelDataReader,
        match_config: MatchConfig,
        standard_file_path: str = "",
    ):
        self._llm = llm
        self._db = db
        self._excel_reader = excel_reader
        self._config = match_config
        self._standard_file_path = standard_file_path

    def process_match_result(
        self, result: MatchResult, trgm_similarity: float = 0.0
    ) -> EvolveAction | None:
        syn_action = self.check_synonym_update(result, trgm_similarity)
        if syn_action is not None:
            success = self.execute_synonym_update(syn_action)
            if success:
                return EvolveAction(
                    action_type=EvolveActionType.SYNONYM_UPDATE,
                    trigger_reason=f"置信度={result.confidence:.4f}, 文本相似度={trgm_similarity:.4f}",
                    timestamp=datetime.now().isoformat(),
                )

        expand_action = self.check_taxonomy_expansion(result)
        if expand_action is not None:
            suggestion = self.execute_taxonomy_expansion(expand_action)
            if suggestion:
                return EvolveAction(
                    action_type=EvolveActionType.TAXONOMY_EXPANSION,
                    trigger_reason=f"置信度={result.confidence:.4f}",
                    timestamp=datetime.now().isoformat(),
                )

        return None

    def check_synonym_update(
        self, result: MatchResult, trgm_similarity: float
    ) -> SynonymUpdateAction | None:
        if result.matched_category_id is None:
            return None
        if result.confidence < self._config.syn_confidence_threshold:
            return None

        category_name = self._get_category_name(result.matched_category_id)
        if not category_name:
            return None

        # pg_trgm 对纯中文常恒为 0；优先用中文字面相似度判断「是否近重复」
        from src.data.text_similarity import chinese_text_similarity
        text_sim = chinese_text_similarity(result.product_name, category_name)
        # 兼容调用方传入的 trgm：取更大者（ASCII 场景 trgm 仍可用）
        effective_text_sim = max(float(trgm_similarity or 0), text_sim)
        if effective_text_sim >= self._config.syn_trgm_threshold:
            return None

        logger.info(
            f"触发同义词发现: product_name={result.product_name}, "
            f"category_id={result.matched_category_id}, "
            f"confidence={result.confidence:.4f}, "
            f"text_sim={text_sim:.4f}, trgm={trgm_similarity:.4f}"
        )

        verify = self._llm.synonym_verification(result.product_name, category_name)

        if verify.is_synonym and verify.confidence >= 0.7:
            return SynonymUpdateAction(
                category_id=result.matched_category_id,
                product_name=result.product_name,
                llm_verified=True,
                timestamp=datetime.now().isoformat(),
            )
        elif verify.confidence >= 0.5:
            logger.warning(
                f"LLM同义校验不确定, 标记待人工确认: "
                f"product_name={result.product_name}, reason={verify.reason}"
            )
            return SynonymUpdateAction(
                category_id=result.matched_category_id,
                product_name=result.product_name,
                llm_verified=False,
                timestamp=datetime.now().isoformat(),
            )

        return None

    def execute_synonym_update(self, action: SynonymUpdateAction) -> bool:
        if not action.llm_verified:
            logger.warning(f"同义词未经LLM确认, 跳过自动追加: {action.product_name}")
            self._record_synonym_update(action, status="PENDING_MANUAL_REVIEW")
            return False

        existing = self._get_existing_synonyms(action.category_id)
        if action.product_name in existing:
            logger.info(f"同义词已存在, 跳过: category_id={action.category_id}")
            return False

        cat_name_row = self._db.execute_one(
            "SELECT category_name FROM category_texts WHERE category_id = %s",
            (action.category_id,),
        )
        cat_name = cat_name_row["category_name"] if cat_name_row else ""
        cleaned, removed = sanitize_syn_list([action.product_name], cat_name)
        if removed or not cleaned:
            logger.info(
                f"同义词被清洗规则拒绝: category_id={action.category_id}, "
                f"synonym={action.product_name}"
            )
            return False

        try:
            self._db.execute(
                """UPDATE category_texts
                   SET syn_list = array_append(syn_list, %s),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE category_id = %s""",
                (action.product_name, action.category_id),
            )

            self._db.execute(
                """UPDATE category_vectors
                   SET syn_list = array_append(syn_list, %s),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE category_id = %s""",
                (action.product_name, action.category_id),
            )

            if self._standard_file_path:
                self._excel_reader.write_back_synonyms(
                    self._standard_file_path, [action]
                )

            self._record_synonym_update(action, status="COMPLETED")
            logger.info(
                f"同义词更新成功: category_id={action.category_id}, "
                f"new_synonym={action.product_name}"
            )
            return True
        except Exception as e:
            logger.error(f"同义词更新失败: {e}")
            self._record_synonym_update(action, status="FAILED")
            return False

    def check_taxonomy_expansion(self, result: MatchResult) -> ExpansionAction | None:
        if result.confidence >= self._config.expand_confidence_threshold:
            return None
        if result.match_status not in (MatchStatus.NO_MATCH, MatchStatus.LOW_CONFIDENCE):
            return None

        logger.info(
            f"触发体系扩展: product_name={result.product_name}, "
            f"confidence={result.confidence:.4f}"
        )

        root_names = self._get_root_category_names()
        analysis = self._llm.category_analysis(result.product_name, root_names)

        return ExpansionAction(
            product_name=result.product_name,
            category_analysis=str(analysis),
            suggested_parent_id=None,
            suggested_name=analysis.category_name or None,
            suggested_level=analysis.level_position or None,
            status=ExpansionStatus.PENDING_REVIEW,
            timestamp=datetime.now().isoformat(),
        )

    def execute_taxonomy_expansion(self, action: ExpansionAction) -> ExpansionSuggestion | None:
        suggestion = ExpansionSuggestion(
            product_name=action.product_name,
            suggested_parent_id=action.suggested_parent_id or "",
            suggested_category_name=action.suggested_name or "",
            suggested_level_position=action.suggested_level or "",
            llm_analysis=action.category_analysis,
            status=ExpansionStatus.PENDING_REVIEW,
            timestamp=action.timestamp,
        )

        try:
            self._db.execute(
                """INSERT INTO expansion_suggestions
                   (product_name, suggested_parent_id, suggested_category_name,
                    suggested_level_position, llm_analysis, status)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    suggestion.product_name,
                    suggestion.suggested_parent_id or None,
                    suggestion.suggested_category_name or None,
                    suggestion.suggested_level_position or None,
                    suggestion.llm_analysis,
                    suggestion.status.value,
                ),
            )
            logger.info(f"扩展建议已记录: product_name={suggestion.product_name}")
            return suggestion
        except Exception as e:
            logger.error(f"扩展建议写入失败: {e}")
            return suggestion

    def _get_category_name(self, category_id: str) -> str:
        try:
            result = self._db.execute_one(
                "SELECT category_name FROM category_texts WHERE category_id = %s",
                (category_id,),
            )
            return result["category_name"] if result else ""
        except Exception:
            return ""

    def _get_existing_synonyms(self, category_id: str) -> list[str]:
        try:
            result = self._db.execute_one(
                "SELECT syn_list FROM category_texts WHERE category_id = %s",
                (category_id,),
            )
            return result["syn_list"] if result else []
        except Exception:
            return []

    def _get_root_category_names(self) -> list[str]:
        try:
            rows = self._db.execute(
                "SELECT category_name FROM category_texts WHERE category_pids = '{}'"
            )
            return [r["category_name"] for r in rows]
        except Exception:
            return []

    def _record_synonym_update(self, action: SynonymUpdateAction, status: str) -> None:
        try:
            self._db.execute(
                """INSERT INTO synonym_updates
                   (category_id, new_synonym, llm_verified, trigger_reason, status)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    action.category_id,
                    action.product_name,
                    action.llm_verified,
                    f"自动发现-{status}",
                    status,
                ),
            )
        except Exception as e:
            logger.error(f"同义词更新记录写入失败: {e}")
