from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.data.expansion_pool import load_pool, save_pool, save_report

logger = logging.getLogger("ClusterEngine")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _single_linkage_cluster(
    embeddings: list[np.ndarray],
    threshold: float = 0.75,
) -> list[list[int]]:
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return list(groups.values())


def _pick_best_name(entries: list[dict]) -> str:
    name_freq: dict[str, int] = defaultdict(int)
    name_conf: dict[str, float] = defaultdict(float)
    for e in entries:
        name = e.get("suggested_category_name", "").strip()
        if not name:
            continue
        name_freq[name] += 1
        name_conf[name] += e.get("confidence", 0.0)

    if not name_freq:
        return entries[0].get("suggested_category_name", "") if entries else ""

    best_name = ""
    best_score = -1.0
    for name, freq in name_freq.items():
        avg_conf = name_conf[name] / freq
        score = freq * 0.6 + avg_conf * 0.4
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def _pick_parent_name(entries: list[dict]) -> str:
    for e in entries:
        name = e.get("suggested_parent_name", "").strip()
        if name:
            return name
    return ""


def _build_taxonomy_overview(page_tree=None) -> str:
    if page_tree is None:
        return ""
    try:
        roots = page_tree.get_root_nodes()
        lines = []
        for r in roots[:30]:
            child_names = [c.category_name for c in r.children[:8]]
            children_str = "、".join(child_names) if child_names else "无子分类"
            lines.append(f"- {r.category_name}(#{r.category_id}): {children_str}")
        return "\n".join(lines)
    except Exception:
        return ""


def _extract_root_category(entry: dict) -> str:
    path_text = entry.get("path_text", "")
    if path_text and ">" in path_text:
        return path_text.split(">")[0].strip()
    path = entry.get("path", [])
    if path:
        return path[0].get("category_name", "UNKNOWN")
    return entry.get("suggested_parent_id", "") or "UNKNOWN"


def _path_depth(full_path: str) -> int:
    if not full_path:
        return 0
    return len([p for p in full_path.replace("›", ">").split(">") if p.strip()])


_BROAD_LEAF_NAMES = frozenset({
    "其他", "其它", "综合", "通用", "相关产品", "相关设备", "机械设备",
    "设备", "产品", "装置", "机械、设备类产品", "其他设备", "其他产品",
    "机械及设备", "设备类产品", "未分类",
})


def _is_overly_broad_cluster(cluster: dict, root_cat: str = "") -> tuple[bool, str]:
    """路径过浅或类名空泛的簇视为不合格，应降级为孤立条目。"""
    full_path = (cluster.get("full_path") or "").strip()
    group_name = (
        cluster.get("merged_category_name")
        or cluster.get("suggested_category_name")
        or ""
    ).strip()
    parent_name = (cluster.get("suggested_parent_name") or "").strip()
    depth = _path_depth(full_path)

    if depth < 3:
        return True, f"路径过浅({depth}级，要求≥3级): {full_path or '(空)'}"

    leaf = full_path.split(">")[-1].strip() if full_path else group_name
    if leaf in _BROAD_LEAF_NAMES or group_name in _BROAD_LEAF_NAMES:
        return True, f"分类名过于宽泛: {leaf or group_name}"

    if root_cat and (full_path == root_cat or (full_path.startswith(root_cat) and depth <= 2)):
        return True, f"仅落在一级大类「{root_cat}」下，过宽泛"

    return False, ""


def _entries_to_outliers(entries: list[dict], reason: str) -> list[dict]:
    out = []
    for e in entries:
        out.append({
            "entry_id": e.get("id") or e.get("entry_id", ""),
            "product_name": e.get("product_name", ""),
            "suggested_parent_id": e.get("suggested_parent_id", ""),
            "suggested_category_name": e.get("suggested_category_name", ""),
            "path_text": e.get("path_text", ""),
            "reason": reason,
        })
    return out


def _precluster_by_embedding(
    group_entries: list[dict],
    embed_func,
    threshold: float,
) -> list[list[dict]]:
    """在同一大类内先用向量相似度预分簇，避免把不相关产品塞进同一次 LLM。"""
    if embed_func is None or len(group_entries) <= 2:
        return [group_entries]

    names = [e["product_name"] for e in group_entries]
    try:
        embeddings = embed_func(names)
    except Exception as ex:
        logger.warning(f"预聚类 embedding 失败: {ex}")
        return [group_entries]

    if not embeddings or len(embeddings) != len(group_entries):
        return [group_entries]

    arrays = [
        np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32)
        for v in embeddings
    ]
    index_groups = _single_linkage_cluster(arrays, threshold=threshold)
    return [[group_entries[i] for i in idxs] for idxs in index_groups]


def run_cluster(
    llm=None,
    page_tree=None,
    embed_func=None,
    similarity_threshold: float = 0.72,
    min_cluster_size: int = 2,
    min_entries: int = 10,
    batch_size: int = 12,
    precluster_threshold: float = 0.70,
) -> dict:
    pool = load_pool()
    entries = pool.get("entries", [])

    if len(entries) < min_entries:
        return {
            "status": "insufficient",
            "message": f"暂存条目不足{min_entries}条(当前{len(entries)}条)，建议继续积累",
            "total_entries": len(entries),
        }

    clusters = []
    outliers = []
    cluster_seq = 0
    used_llm = False

    if llm is not None:
        try:
            taxonomy_overview = _build_taxonomy_overview(page_tree)

            parent_groups: dict[str, list[dict]] = defaultdict(list)
            for e in entries:
                root_cat = _extract_root_category(e)
                parent_groups[root_cat].append(e)

            all_llm_clusters = []
            all_llm_outliers = []

            for root_cat, group in parent_groups.items():
                if len(group) <= 1:
                    all_llm_outliers.extend(
                        _entries_to_outliers(group, f"该大类({root_cat})下仅1条，无法聚类")
                    )
                    continue

                # 先按产品向量预分簇，再送 LLM，避免「同属机械设备」就被并成一坨
                pre_groups = _precluster_by_embedding(
                    group, embed_func, threshold=precluster_threshold
                )
                logger.info(
                    f"大类「{root_cat}」{len(group)}条 → 向量预分 {len(pre_groups)} 组"
                )

                for pre in pre_groups:
                    if len(pre) < min_cluster_size:
                        all_llm_outliers.extend(
                            _entries_to_outliers(
                                pre,
                                "向量预聚类后组内产品不够相近，归入孤立条目",
                            )
                        )
                        continue

                    for batch_start in range(0, len(pre), batch_size):
                        batch = pre[batch_start : batch_start + batch_size]
                        logger.info(
                            f"LLM聚类批次: root={root_cat}, 预分簇内 {len(batch)}条"
                        )
                        try:
                            bc, bo = llm.cluster_products(batch, taxonomy_overview)
                            # 过宽泛的簇打散为孤立
                            entry_by_id = {e["id"]: e for e in batch}
                            for lc in bc:
                                broad, why = _is_overly_broad_cluster(lc, root_cat)
                                if broad:
                                    demoted = [
                                        entry_by_id[eid]
                                        for eid in lc.get("entries", [])
                                        if eid in entry_by_id
                                    ]
                                    if not demoted:
                                        demoted = [
                                            e for e in batch
                                            if e["product_name"] in set(lc.get("product_names") or [])
                                        ]
                                    all_llm_outliers.extend(
                                        _entries_to_outliers(
                                            demoted,
                                            f"簇过宽泛已打散: {why}",
                                        )
                                    )
                                    logger.info(f"打散过宽泛簇: {why}; products={lc.get('product_names')}")
                                else:
                                    all_llm_clusters.append(lc)
                            all_llm_outliers.extend(bo)
                        except Exception as ex:
                            logger.warning(
                                f"LLM聚类批次失败(root={root_cat}): {ex}"
                            )
                            all_llm_outliers.extend(
                                _entries_to_outliers(
                                    batch, f"LLM聚类失败: {str(ex)[:50]}"
                                )
                            )

            used_llm = True

            for lc in all_llm_clusters:
                cluster_seq += 1
                depth = _path_depth(lc.get("full_path", ""))
                # 路径越深星级略高，避免「大杂烩大簇」显得可信
                base_star = 3 if len(lc.get("product_names") or []) >= 5 else (
                    2 if len(lc.get("product_names") or []) >= 3 else 1
                )
                if depth >= 4:
                    star = min(3, base_star + 1)
                elif depth < 3:
                    star = 1
                else:
                    star = base_star

                clusters.append({
                    "cluster_id": f"c_{cluster_seq:03d}",
                    "suggested_parent_id": lc.get("suggested_parent_id", ""),
                    "suggested_parent_name": lc.get("suggested_parent_name", ""),
                    "suggested_category_name": lc.get("suggested_category_name", ""),
                    "merged_category_name": lc.get("merged_category_name", ""),
                    "full_path": lc.get("full_path", ""),
                    "entry_count": lc.get("entry_count", 0),
                    "avg_confidence": lc.get("avg_confidence", 0.0),
                    "confidence_variance": lc.get("confidence_variance", 0.0),
                    "star_rating": star,
                    "has_divergence": lc.get("has_divergence", False),
                    "entries": lc.get("entries", []),
                    "product_names": lc.get("product_names", []),
                    "llm_reason": lc.get("llm_reason", ""),
                    "status": "PENDING_REVIEW",
                    "review_note": "",
                    "category_id_created": "",
                    "is_llm_clustered": True,
                })

            outliers = all_llm_outliers
            logger.info(
                f"LLM聚类完成: {len(clusters)}个有效簇 + {len(outliers)}个孤立条目"
            )

        except Exception as e:
            logger.warning(f"LLM聚类失败，退化为embedding聚类: {e}")
            used_llm = False

    if not used_llm:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            root_cat = _extract_root_category(e)
            groups[root_cat].append(e)

        for root_cat, group_entries in groups.items():
            if len(group_entries) == 1:
                e = group_entries[0]
                outliers.append({
                    "entry_id": e["id"],
                    "product_name": e["product_name"],
                    "suggested_parent_id": e.get("suggested_parent_id", ""),
                    "suggested_category_name": e.get("suggested_category_name", ""),
                    "path_text": e.get("path_text", ""),
                    "reason": f"该大类({root_cat})下仅1条，无法聚类",
                })
                continue

            if embed_func is not None:
                product_names = [e["product_name"] for e in group_entries]
                category_names = [
                    e.get("suggested_category_name", "") or e["product_name"]
                    for e in group_entries
                ]
                product_embeddings = None
                category_embeddings = None
                try:
                    product_embeddings = embed_func(product_names)
                except Exception:
                    pass
                try:
                    category_embeddings = embed_func(category_names)
                except Exception:
                    pass

                if product_embeddings is not None and category_embeddings is not None:
                    n = len(group_entries)
                    combined_sim = np.zeros((n, n), dtype=np.float32)
                    for i in range(n):
                        for j in range(i + 1, n):
                            ps = _cosine_similarity(product_embeddings[i], product_embeddings[j])
                            cs = _cosine_similarity(category_embeddings[i], category_embeddings[j])
                            combined_sim[i][j] = 0.6 * ps + 0.4 * cs
                            combined_sim[j][i] = combined_sim[i][j]
                    sub_groups = _single_linkage_cluster(product_embeddings, threshold=similarity_threshold)
                elif product_embeddings is not None:
                    sub_groups = _single_linkage_cluster(product_embeddings, threshold=similarity_threshold)
                else:
                    sub_groups = None
            else:
                sub_groups = None

            if sub_groups is None:
                name_groups: dict[str, list[int]] = defaultdict(list)
                for i, e in enumerate(group_entries):
                    key = e.get("suggested_category_name", "").strip()
                    name_groups[key].append(i)
                sub_groups = list(name_groups.values())

            for indices in sub_groups:
                sub_entries = [group_entries[i] for i in indices]

                if len(sub_entries) < min_cluster_size:
                    for e in sub_entries:
                        outliers.append({
                            "entry_id": e["id"],
                            "product_name": e["product_name"],
                            "suggested_parent_id": e.get("suggested_parent_id", ""),
                            "suggested_category_name": e.get("suggested_category_name", ""),
                            "path_text": e.get("path_text", ""),
                            "reason": "簇大小不足，归入孤立条目",
                        })
                    continue

                cluster_seq += 1
                merged_name = _pick_best_name(sub_entries)
                parent_name = _pick_parent_name(sub_entries)
                confidences = [e.get("confidence", 0.0) for e in sub_entries]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                conf_variance = float(np.var(confidences)) if len(confidences) > 1 else 0.0
                star_rating = 3 if len(sub_entries) >= 5 else (2 if len(sub_entries) >= 3 else 1)

                clusters.append({
                    "cluster_id": f"c_{cluster_seq:03d}",
                    "suggested_parent_id": "",
                    "suggested_parent_name": root_cat,
                    "suggested_category_name": merged_name,
                    "merged_category_name": merged_name,
                    "full_path": f"{root_cat} > {merged_name}",
                    "entry_count": len(sub_entries),
                    "avg_confidence": round(avg_conf, 4),
                    "confidence_variance": round(conf_variance, 4),
                    "star_rating": star_rating,
                    "has_divergence": conf_variance > 0.05,
                    "entries": [e["id"] for e in sub_entries],
                    "product_names": [e["product_name"] for e in sub_entries],
                    "status": "PENDING_REVIEW",
                    "review_note": "",
                    "category_id_created": "",
                    "is_llm_clustered": False,
                })

    report = {
        "version": 1,
        "cluster_time": datetime.now().isoformat(),
        "cluster_method": "llm" if used_llm else "embedding",
        "total_entries": len(entries),
        "cluster_count": len(clusters),
        "outlier_count": len(outliers),
        "clusters": clusters,
        "outliers": outliers,
    }

    save_report(report)

    pool["last_cluster_time"] = report["cluster_time"]
    save_pool(pool)

    logger.info(
        f"聚类完成({('LLM' if used_llm else 'embedding')}): "
        f"{len(entries)}条 → {len(clusters)}个簇 + {len(outliers)}个孤立条目"
    )

    return {"status": "ok", **report}
