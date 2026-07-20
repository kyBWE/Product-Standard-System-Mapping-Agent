from __future__ import annotations
import logging

from src.models.category_node import CategoryNode


logger = logging.getLogger("SynonymSanitizer")

# ≤2 字且语义过宽、易造成误匹配的同义词
GENERIC_SHORT_SYNONYMS: frozenset[str] = frozenset({
    "系统", "设备", "产品", "装置", "机器", "机械", "材料", "部件", "组件",
    "配件", "用品", "制品", "物品", "工具", "仪器", "仪表", "软件", "硬件",
    "服务", "技术", "工程", "制造", "生产", "加工", "处理", "管理", "其他",
    "类型", "品种", "类别", "系列", "通用", "专用", "标准", "新型", "高级",
})


def _is_category_related(short: str, category_name: str) -> bool:
    """短同义词与分类名有明确关联时保留（如 水泵→水泵设备）。"""
    if not short or not category_name:
        return False
    if short == category_name:
        return True
    if short in category_name:
        return True
    if category_name in short:
        return True
    return False


def sanitize_syn_list(
    syn_list: list[str],
    category_name: str = "",
) -> tuple[list[str], list[str]]:
    """清洗同义词列表，返回 (保留列表, 移除列表)。"""
    kept: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()

    for syn in syn_list:
        s = (syn or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)

        if len(s) <= 2:
            if _is_category_related(s, category_name):
                kept.append(s)
            elif s in GENERIC_SHORT_SYNONYMS:
                removed.append(s)
            else:
                removed.append(s)
        else:
            kept.append(s)

    return kept, removed


def sanitize_nodes(nodes: list[CategoryNode]) -> int:
    """就地清洗节点同义词，返回移除的词条总数。"""
    total_removed = 0
    for node in nodes:
        cleaned, removed = sanitize_syn_list(node.syn_list, node.category_name)
        if removed:
            total_removed += len(removed)
            logger.debug(
                f"同义词清洗 category_id={node.category_id}: "
                f"移除={removed}, 保留={len(cleaned)}"
            )
        node.syn_list = cleaned
    if total_removed:
        logger.info(f"同义词清洗完成: 共移除 {total_removed} 条泛词同义词")
    return total_removed
