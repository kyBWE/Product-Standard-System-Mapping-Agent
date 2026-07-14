from __future__ import annotations
import logging
import sys

from src.data.excel_reader import ExcelDataReader
from src.engine.llm_adapter import LLMAdapter
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.logger import StructuredLogger
from src.models.evolve_models import SynonymUpdateAction
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler


def update_synonyms(
    config_path: str = "config.yaml",
    category_id: str | None = None,
    new_synonym: str | None = None,
) -> list[SynonymUpdateAction]:
    """
    同义词更新入口函数
    支持手动追加或基于历史匹配结果批量自动发现
    """
    logger = StructuredLogger()
    actions: list[SynonymUpdateAction] = []

    try:
        config = ConfigManager(config_path)
    except FileNotFoundError as e:
        logger.error("SynonymUpdater", f"配置文件加载失败: {e}")
        return actions

    db_config = config.get_db_config()
    llm_config = config.get_llm_config()
    match_config = config.get_match_config()
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")

    db = DBConnectionManager(db_config)
    try:
        db.initialize()
    except Exception as e:
        logger.error("SynonymUpdater", f"数据库连接失败: {e}")
        return actions

    llm = LLMAdapter(llm_config)
    excel_reader = ExcelDataReader()
    scheduler = SelfEvolveScheduler(llm, db, excel_reader, match_config, standard_file)

    if category_id and new_synonym:
        logger.info("SynonymUpdater", f"手动追加同义词: category_id={category_id}, synonym={new_synonym}")

        cat_result = db.execute_one(
            "SELECT category_name FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        if not cat_result:
            logger.error("SynonymUpdater", f"未找到category_id={category_id}")
            db.close()
            return actions

        verify = llm.synonym_verification(new_synonym, cat_result["category_name"])
        action = SynonymUpdateAction(
            category_id=category_id,
            product_name=new_synonym,
            llm_verified=verify.is_synonym,
            timestamp=__import__("datetime").datetime.now().isoformat(),
        )
        actions.append(action)

        if verify.is_synonym:
            success = scheduler.execute_synonym_update(action)
            logger.info("SynonymUpdater", f"同义词追加{'成功' if success else '失败'}")
        else:
            logger.warning("SynonymUpdater", f"LLM判定不同义, 跳过: reason={verify.reason}")
    else:
        logger.info("SynonymUpdater", "批量自动发现同义词")

        rows = db.execute(
            """SELECT mr.product_name, mr.matched_category_id, mr.confidence
               FROM match_results mr
               WHERE mr.match_status = 'MATCHED'
                 AND mr.confidence >= %s
               ORDER BY mr.confidence DESC
               LIMIT 100""",
            (match_config.syn_confidence_threshold,),
        )

        for row in rows:
            cat_id = row["matched_category_id"]
            pn = row["product_name"]

            cat_result = db.execute_one(
                "SELECT category_name FROM category_texts WHERE category_id = %s",
                (cat_id,),
            )
            if not cat_result:
                continue

            trgm_sim_result = db.execute_one(
                "SELECT similarity(%s, %s) AS sim",
                (pn, cat_result["category_name"]),
            )
            trgm_sim = float(trgm_sim_result["sim"]) if trgm_sim_result else 0.0

            if trgm_sim >= match_config.syn_trgm_threshold:
                continue

            verify = llm.synonym_verification(pn, cat_result["category_name"])
            action = SynonymUpdateAction(
                category_id=cat_id,
                product_name=pn,
                llm_verified=verify.is_synonym,
                timestamp=__import__("datetime").datetime.now().isoformat(),
            )
            actions.append(action)

            if verify.is_synonym:
                scheduler.execute_synonym_update(action)

    db.close()
    logger.info("SynonymUpdater", f"同义词更新完成: 共{len(actions)}条操作")
    return actions


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cid = sys.argv[2] if len(sys.argv) > 2 else None
    syn = sys.argv[3] if len(sys.argv) > 3 else None
    r = update_synonyms(cfg, cid, syn)
    for a in r:
        print(f"category_id={a.category_id}, synonym={a.product_name}, verified={a.llm_verified}")
