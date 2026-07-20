from __future__ import annotations
import logging
import sys
from datetime import datetime

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger
from src.models.enums import ExpansionStatus, MatchStatus
from src.models.evolve_models import ExpansionSuggestion
from src.orchestration.result_exporter import ResultExporter
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler


def generate_taxonomy(config_path: str = "config.yaml") -> list[ExpansionSuggestion]:
    """
    体系生成入口函数
    基于低置信度匹配结果生成标准体系扩展建议，供管理员审核
    """
    logger = StructuredLogger()
    suggestions: list[ExpansionSuggestion] = []

    try:
        config = ConfigManager(config_path)
    except FileNotFoundError as e:
        logger.error("TaxonomyGenerator", f"配置文件加载失败: {e}")
        return suggestions

    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    output_dir = config.get("data.output_dir", "./output")
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")

    db = DBConnectionManager(db_config)
    try:
        db.initialize()
    except Exception as e:
        logger.error("TaxonomyGenerator", f"数据库连接失败: {e}")
        return suggestions

    llm = LLMAdapter(llm_config)
    excel_reader = ExcelDataReader()
    scheduler = SelfEvolveScheduler(llm, db, excel_reader, match_config, standard_file)
    exporter = ResultExporter(output_dir)

    try:
        rows = db.execute(
            """SELECT DISTINCT product_name FROM match_results
               WHERE match_status IN ('NO_MATCH', 'LOW_CONFIDENCE')
                 AND product_name NOT IN (
                     SELECT product_name FROM expansion_suggestions
                 )
               LIMIT 200"""
        )
    except Exception as e:
        logger.error("TaxonomyGenerator", f"查询低置信度结果失败: {e}")
        db.close()
        return suggestions

    if not rows:
        logger.info("TaxonomyGenerator", "无低置信度匹配结果, 无需生成扩展建议")
        db.close()
        return suggestions

    root_names = []
    try:
        root_rows = db.execute(
            "SELECT category_name FROM category_texts WHERE category_pids = '{}'"
        )
        root_names = [r["category_name"] for r in root_rows]
    except Exception:
        pass

    logger.info("TaxonomyGenerator", f"开始分析{len(rows)}条低置信度产品")

    for row in rows:
        pn = row["product_name"]
        try:
            analysis = llm.category_analysis(pn, root_names)
            suggestion = ExpansionSuggestion(
                product_name=pn,
                suggested_parent_id="",
                suggested_category_name=analysis.category_name,
                suggested_level_position=analysis.level_position,
                llm_analysis=str(analysis),
                status=ExpansionStatus.PENDING_REVIEW,
                timestamp=datetime.now().isoformat(),
            )

            try:
                db.execute(
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
            except Exception as e:
                logger.warning(f"扩展建议写入失败: {e}")

            suggestions.append(suggestion)
        except Exception as e:
            logger.warning(f"品类分析失败: product_name={pn}, error={e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exporter.export_expansion_suggestions(suggestions, f"expansion_suggestions_{timestamp}.csv")

    db.close()
    logger.info("TaxonomyGenerator", f"扩展建议生成完成: 共{len(suggestions)}条")
    return suggestions


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    r = generate_taxonomy(cfg)
    for s in r:
        print(f"{s.product_name} -> {s.suggested_category_name} ({s.status.value})")
