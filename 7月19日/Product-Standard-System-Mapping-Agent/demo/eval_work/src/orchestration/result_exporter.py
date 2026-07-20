from __future__ import annotations
import csv
import logging
import os
from datetime import datetime

from src.models.enums import MatchStatus
from src.models.match_result import MatchResult
from src.models.evolve_models import ExpansionSuggestion


logger = logging.getLogger("ResultExporter")


class ResultExporter:
    def __init__(self, output_dir: str = "./output"):
        self._output_dir = output_dir

    def export_csv(self, results: list[MatchResult], file_name: str) -> str:
        os.makedirs(self._output_dir, exist_ok=True)
        file_path = os.path.join(self._output_dir, file_name)

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["product_name", "matched_category_id", "confidence", "match_status"])
                for r in results:
                    writer.writerow([
                        r.product_name,
                        r.matched_category_id or "",
                        f"{r.confidence:.4f}",
                        r.match_status.value,
                    ])
            logger.info(f"匹配结果CSV导出成功: {file_path}")
            return file_path
        except PermissionError:
            fallback_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "product_mapping_output")
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_path = os.path.join(fallback_dir, file_name)
            try:
                with open(fallback_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["product_name", "matched_category_id", "confidence", "match_status"])
                    for r in results:
                        writer.writerow([
                            r.product_name,
                            r.matched_category_id or "",
                            f"{r.confidence:.4f}",
                            r.match_status.value,
                        ])
                logger.info(f"匹配结果CSV导出至备用目录: {fallback_path}")
                return fallback_path
            except Exception as e:
                logger.error(f"CSV导出失败: {e}, 输出至控制台")
                for r in results:
                    print(f"{r.product_name},{r.matched_category_id or ''},{r.confidence:.4f},{r.match_status.value}")
                return ""
        except Exception as e:
            logger.error(f"CSV导出失败: {e}")
            return ""

    def export_expansion_suggestions(self, suggestions: list[ExpansionSuggestion], file_name: str) -> str:
        os.makedirs(self._output_dir, exist_ok=True)
        file_path = os.path.join(self._output_dir, file_name)

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "product_name", "suggested_parent_id", "suggested_category_name",
                    "suggested_level_position", "llm_analysis", "status", "timestamp",
                ])
                for s in suggestions:
                    writer.writerow([
                        s.product_name,
                        s.suggested_parent_id,
                        s.suggested_category_name,
                        s.suggested_level_position,
                        s.llm_analysis,
                        s.status.value,
                        s.timestamp,
                    ])
            logger.info(f"扩展建议CSV导出成功: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"扩展建议CSV导出失败: {e}")
            return ""
