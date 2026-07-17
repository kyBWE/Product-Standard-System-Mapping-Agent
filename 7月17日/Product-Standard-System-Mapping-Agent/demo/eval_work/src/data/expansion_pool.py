from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from uuid import uuid4

import logging

logger = logging.getLogger("ExpansionPool")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)

PENDING_POOL_PATH = os.path.join(DATA_DIR, "pending_pool.json")
CLUSTER_REPORT_PATH = os.path.join(DATA_DIR, "cluster_report.json")

_LOCK = threading.Lock()


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _init_pool() -> dict:
    return {
        "version": 1,
        "last_cluster_time": None,
        "entries": [],
    }


def _init_report() -> dict:
    return {
        "version": 1,
        "cluster_time": None,
        "total_entries": 0,
        "clusters": [],
        "outliers": [],
    }


def load_pool(path: str | None = None) -> dict:
    p = path or PENDING_POOL_PATH
    if not os.path.exists(p):
        return _init_pool()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "entries" not in data:
            data["entries"] = []
        return data
    except Exception as e:
        logger.warning(f"暂存池读取失败，重建: {e}")
        return _init_pool()


def save_pool(pool: dict, path: str | None = None) -> None:
    p = path or PENDING_POOL_PATH
    _ensure_data_dir()
    with _LOCK:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)


def add_entry(
    product_name: str,
    suggested_parent_id: str,
    suggested_parent_name: str,
    suggested_category_name: str,
    path: list[dict],
    confidence: float,
    llm_reason: str,
    sibling_nodes: list[dict],
    source: str = "web",
    path_text: str | None = None,
) -> dict:
    pool = load_pool()

    for e in pool["entries"]:
        if e["product_name"] == product_name:
            logger.info(f"产品已存在于暂存池: {product_name}")
            return {"status": "already_exists", "entry_id": e["id"]}

    entry_id = f"e_{uuid4().hex[:6]}"
    entry = {
        "id": entry_id,
        "product_name": product_name,
        "suggested_parent_id": suggested_parent_id,
        "suggested_parent_name": suggested_parent_name,
        "suggested_category_name": suggested_category_name,
        "path": path,
        "path_text": path_text or "",
        "confidence": round(float(confidence), 4),
        "llm_reason": llm_reason,
        "sibling_nodes": sibling_nodes[:20],
        "source": source,
        "created_at": datetime.now().isoformat(),
    }

    pool["entries"].append(entry)
    save_pool(pool)

    logger.info(f"暂存条目已添加: {entry_id} - {product_name}")
    return {"status": "ok", "entry_id": entry_id}


def remove_entry(entry_id: str) -> dict:
    pool = load_pool()
    before = len(pool["entries"])
    pool["entries"] = [e for e in pool["entries"] if e["id"] != entry_id]
    after = len(pool["entries"])

    if before == after:
        return {"status": "not_found", "entry_id": entry_id}

    save_pool(pool)
    return {"status": "ok", "removed": entry_id}


def remove_entries(entry_ids: list[str]) -> dict:
    pool = load_pool()
    id_set = set(entry_ids)
    before = len(pool["entries"])
    pool["entries"] = [e for e in pool["entries"] if e["id"] not in id_set]
    after = len(pool["entries"])

    save_pool(pool)
    return {"status": "ok", "removed_count": before - after}


def get_pool_stats() -> dict:
    pool = load_pool()
    entries = pool["entries"]
    total = len(entries)

    parent_dist: dict[str, int] = {}
    for e in entries:
        pid = e.get("suggested_parent_id", "") or "未确定"
        parent_dist[pid] = parent_dist.get(pid, 0) + 1

    last_cluster = pool.get("last_cluster_time")
    days_since = None
    if last_cluster:
        try:
            dt = datetime.fromisoformat(last_cluster)
            days_since = (datetime.now() - dt).days
        except Exception:
            pass

    return {
        "total": total,
        "parent_distribution": parent_dist,
        "last_cluster_time": last_cluster,
        "days_since_last_cluster": days_since,
    }


def load_report(path: str | None = None) -> dict:
    p = path or CLUSTER_REPORT_PATH
    if not os.path.exists(p):
        return _init_report()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"聚类报告读取失败: {e}")
        return _init_report()


def save_report(report: dict, path: str | None = None) -> None:
    p = path or CLUSTER_REPORT_PATH
    _ensure_data_dir()
    with _LOCK:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)