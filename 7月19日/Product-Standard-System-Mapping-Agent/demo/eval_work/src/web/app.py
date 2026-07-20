from __future__ import annotations
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler
from src.models.enums import EngineType, MatchStatus
from src.infrastructure.db_manager import DBConnectionManager
from src.infrastructure.config_manager import ConfigManager
from src.index.vector_index_manager import VectorIndexManager
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.page_index_tree import PageIndexTree
from src.engine.rerank_adapter import RerankAdapter
from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.page_index_engine import PageIndexEngine
from src.engine.llm_adapter import LLMAdapter
from src.data.taxonomy_utils import (
    allocate_next_category_id,
    build_category_path_fields,
    format_category_path,
    locate_expansion_parent,
)
from src.data.excel_reader import ExcelDataReader
import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_from_directory

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "web")


app = Flask(__name__, static_folder=os.path.join(
    WEB_DIR, "static"), static_url_path="/static")

CONFIG_PATH = "config.yaml"

_config = None
_db = None
_llm = None
_trgm_mgr = None
_vec_mgr = None
_rag_engine = None
_rag_rerank_engine = None
_page_engine = None
_page_engine_force_llm = None
_page_tree = None
_excel_reader = None
_evolve_scheduler = None
_initialized = False


def _load_category_nodes_from_db(db=None) -> list:
    """从 category_texts 加载分类节点，供 PageIndex 建树（Excel 不可用时的兜底）。"""
    from src.models.category_node import CategoryNode
    conn = db or _db
    if conn is None:
        return []
    rows = conn.execute(
        """
        SELECT category_id, category_name, category_pids, category_group_name, syn_list
        FROM category_texts
        """
    )
    nodes = []
    for r in rows or []:
        pids = r.get("category_pids") or []
        if isinstance(pids, str):
            pids = [p for p in pids.strip("{}").split(",") if p]
        syn = r.get("syn_list") or []
        if isinstance(syn, str):
            syn = [s for s in syn.strip("{}").split(",") if s]
        nodes.append(
            CategoryNode(
                category_id=str(r["category_id"]),
                category_name=r["category_name"] or "",
                category_pids=[str(p) for p in pids if str(p) not in ("", "-1")],
                category_group_name=r.get("category_group_name") or "",
                syn_list=list(syn),
            )
        )
    return nodes


def _ensure_page_tree_ready() -> bool:
    """确保 PageIndex 树可用；空树则从 DB 重建。"""
    global _page_tree, _page_engine, _page_engine_force_llm
    if _page_tree is not None and getattr(_page_tree, "_node_map", None) and _page_tree.get_root_nodes():
        return True
    if _db is None or _page_tree is None:
        return False
    nodes = _load_category_nodes_from_db(_db)
    if not nodes:
        return False
    _page_tree.build_tree(nodes)
    logging.getLogger("WebAPI").info(
        f"PageIndex树已从DB重建: 根={len(_page_tree.get_root_nodes())}, 总={len(_page_tree._node_map)}"
    )
    return bool(_page_tree.get_root_nodes())


def _pageindex_locate_expansion_parent(product_name: str) -> dict:
    """用 PageIndex 自上而下定位扩展父节点：某层无合适子类则停在该层。"""
    if not _ensure_page_tree_ready() or not _llm:
        return {"ok": False, "error": "PageIndex树或LLM不可用"}

    roots = list(_page_tree.get_root_nodes())
    if not roots:
        return {"ok": False, "error": "无根节点"}

    # 根层必须选一个大类（不允许 stop）
    root_names = [r.category_name for r in roots]
    picked_name, conf, reason = _llm.layer_pick_or_stop(product_name, root_names, [])
    if picked_name is None:
        # 根层强制选置信最高的表述：再问一次仅选择
        picked_name = _llm.layer_disambiguation(product_name, root_names)
        conf = max(conf, 0.5)
        reason = reason or "根层强制选择大类"
    current = next((r for r in roots if r.category_name == picked_name), roots[0])
    path = [current]
    steps = [{"level": 0, "choice": current.category_name, "action": "enter", "reason": reason, "confidence": conf}]

    for depth in range(1, 10):
        children = list(current.children or [])
        if not children:
            steps.append({"level": depth, "action": "leaf", "choice": current.category_name})
            break
        # 子节点过多时只送名称；若>40则截断并提示（避免 prompt 爆炸）
        child_names = [c.category_name for c in children]
        if len(child_names) > 40:
            # 优先用向量/名称粗筛 Top40 再交给 LLM
            child_names = _rank_children_for_expansion(product_name, children)[:40]
            children = [c for c in children if c.category_name in set(child_names)]
            # 保持 child_names 顺序
            name_to_child = {c.category_name: c for c in children}
            children = [name_to_child[n] for n in child_names if n in name_to_child]

        ancestry = [n.category_name for n in path]
        picked_name, conf, reason = _llm.layer_pick_or_stop(
            product_name, [c.category_name for c in children], ancestry
        )
        if picked_name is None:
            steps.append({
                "level": depth,
                "action": "stop",
                "choice": current.category_name,
                "reason": reason,
                "confidence": conf,
            })
            break
        nxt = next((c for c in children if c.category_name == picked_name), None)
        if nxt is None:
            steps.append({"level": depth, "action": "stop", "choice": current.category_name, "reason": "未匹配到子节点"})
            break
        current = nxt
        path.append(current)
        steps.append({
            "level": depth,
            "action": "enter",
            "choice": current.category_name,
            "reason": reason,
            "confidence": conf,
        })
        # 「其他…」兜底叶子 / 与产品近同名：上提或停止
        from src.data.text_similarity import chinese_text_similarity
        if current.category_name.startswith("其他") and current.parent:
            current = current.parent
            path = path[:-1]
            steps.append({"level": depth, "action": "promote", "choice": current.category_name, "reason": "避开「其他」兜底类"})
            break
        if chinese_text_similarity(current.category_name, product_name) >= 0.85 and current.parent:
            current = current.parent
            path = path[:-1]
            steps.append({"level": depth, "action": "promote", "choice": current.category_name, "reason": "近同名叶子上提"})
            break

    path_text = " > ".join(n.category_name for n in path)
    sibling_names = [c.category_name for c in (current.children or [])][:30]
    return {
        "ok": True,
        "parent_id": str(current.category_id),
        "parent_name": current.category_name,
        "path_nodes": path,
        "path_text": path_text,
        "sibling_names": sibling_names,
        "steps": steps,
        "confidence": float(steps[-1].get("confidence") or conf or 0.6),
    }


def _rank_children_for_expansion(product_name: str, children: list) -> list[str]:
    """子节点过多时按中文字面相关度粗排，供扩展层 LLM 选择。"""
    try:
        from src.data.text_similarity import chinese_text_similarity
        scored = []
        for c in children:
            name = c.category_name or ""
            score = chinese_text_similarity(product_name, name)
            if name and (name in product_name or product_name in name):
                score += 0.2
            scored.append((score, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored if n]
    except Exception:
        return [c.category_name for c in children if c.category_name]


def _init_components():
    global _config, _db, _llm, _trgm_mgr, _vec_mgr, _rag_engine, _rag_rerank_engine, _page_engine, _page_engine_force_llm, _page_tree, _excel_reader, _evolve_scheduler, _initialized
    if _initialized:
        return

    config = ConfigManager(CONFIG_PATH)
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()

    db = DBConnectionManager(db_config)
    db.initialize()
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_box (
                id SERIAL PRIMARY KEY,
                product_name TEXT NOT NULL UNIQUE,
                status TEXT DEFAULT 'pending',
                source TEXT DEFAULT 'web',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS expansion_log (
                id SERIAL PRIMARY KEY,
                product_name TEXT NOT NULL,
                category_id TEXT,
                category_name TEXT,
                match_path TEXT,
                match_status TEXT,
                source TEXT DEFAULT 'web',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute(
            "ALTER TABLE category_texts ADD COLUMN IF NOT EXISTS expansion_syn_list TEXT[] DEFAULT '{}'"
        )
        db.execute(
            "ALTER TABLE category_vectors ADD COLUMN IF NOT EXISTS expansion_syn_list TEXT[] DEFAULT '{}'"
        )
        try:
            db.execute(
                "ALTER TABLE category_vectors ADD COLUMN IF NOT EXISTS vec_bgem3 vector(1024)"
            )
        except Exception as vec_e:
            logging.getLogger("WebAPI").warning(f"确保 vec_bgem3 列失败(可忽略): {vec_e}")
    except Exception as e:
        logging.getLogger("WebAPI").warning(f"确保扩展相关表/列失败: {e}")
    llm = LLMAdapter(llm_config)

    embedding_config = config.get_embedding_config()

    trgm_mgr = TrgmIndexManager(db)
    vec_mgr = VectorIndexManager(
        db,
        embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        embedding_config=embedding_config,
    )
    try:
        pg_ok = vec_mgr.ensure_pgvector_ready()
        vec_mgr.warmup()
        logging.getLogger("WebAPI").info(
            f"向量索引初始化完成: pgvector={'是' if pg_ok else '否(内存矩阵)'}"
        )
    except Exception as e:
        logging.getLogger("WebAPI").warning(f"向量索引预热失败: {e}")

    match_config = config.get_match_config()
    rerank_config = config.get_rerank_config()

    rag_engine = RAGMatchEngine(
        vec_mgr, trgm_mgr, llm, match_config,
        enable_llm=match_config.enable_llm,
        fine_match_mode="llm",
        engine_type=EngineType.RAG_VECTOR,
    )
    rag_rerank_engine = RAGMatchEngine(
        vec_mgr, trgm_mgr, llm, match_config,
        enable_llm=match_config.enable_rerank,
        rerank=RerankAdapter(rerank_config),
        fine_match_mode="rerank",
        engine_type=EngineType.RAG_RERANK,
    )

    excel_reader = ExcelDataReader()
    standard_file = config.get("data.standard_system_file", "产品标准体系.xlsx")
    page_tree = PageIndexTree()
    try:
        nodes, _ = excel_reader.load_standard_system(standard_file)
        page_tree.build_tree(nodes)
        logging.getLogger("WebAPI").info(
            f"PageIndex树构建完成(Excel): {len(page_tree.get_root_nodes())}个根节点, "
            f"共{len(page_tree._node_map)}个节点"
        )
    except Exception as e:
        logging.getLogger("WebAPI").warning(f"Excel构建PageIndex失败，尝试从数据库加载: {e}")
        try:
            nodes = _load_category_nodes_from_db(db)
            if nodes:
                page_tree.build_tree(nodes)
                logging.getLogger("WebAPI").info(
                    f"PageIndex树构建完成(DB): {len(page_tree.get_root_nodes())}个根节点, "
                    f"共{len(page_tree._node_map)}个节点"
                )
            else:
                logging.getLogger("WebAPI").error("数据库无分类节点，PageIndex树为空")
        except Exception as e2:
            logging.getLogger("WebAPI").error(f"DB构建PageIndex失败: {e2}")
            import traceback
            traceback.print_exc()

    rerank_adapter = RerankAdapter(
        rerank_config) if rerank_config.api_key else None

    page_engine = PageIndexEngine(page_tree, llm, force_llm_each_layer=False,
                                  vec_mgr=vec_mgr, rerank=rerank_adapter, trgm_mgr=trgm_mgr)
    page_engine_force_llm = PageIndexEngine(
        page_tree, llm, force_llm_each_layer=True, vec_mgr=vec_mgr, rerank=rerank_adapter, trgm_mgr=trgm_mgr)

    _config = config
    _db = db
    _llm = llm
    _trgm_mgr = trgm_mgr
    _vec_mgr = vec_mgr
    _rag_engine = rag_engine
    _rag_rerank_engine = rag_rerank_engine
    _page_engine = page_engine
    _page_engine_force_llm = page_engine_force_llm
    _page_tree = page_tree
    _excel_reader = excel_reader
    standard_file_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), standard_file)
    _evolve_scheduler = SelfEvolveScheduler(
        llm, db, excel_reader, match_config, standard_file_path)
    _initialized = True


def _trigram_fallback_match(product_name: str, top_k: int = 3, trgm_threshold: float = 0.3, vector_threshold: float = 0.5):
    """Trigram降级匹配，返回top-k候选并计算向量相似度"""
    trgm_rows = _db.execute(
        f"""SELECT category_id, category_name, similarity(category_name, %s) AS sim
           FROM category_texts
           ORDER BY sim DESC LIMIT {top_k}""",
        (product_name,),
    )
    if not trgm_rows:
        return []

    candidates = []
    query_vec = None
    if _vec_mgr:
        try:
            query_vec = _vec_mgr.embed_query(product_name)
        except Exception:
            pass

    for row in trgm_rows:
        trgm_sim = row["sim"]
        if trgm_sim < trgm_threshold:
            continue

        vec_sim = 0.0
        if not _vec_is_empty(query_vec):
            try:
                cat_vec_row = _db.execute_one(
                    "SELECT embedding FROM category_vectors WHERE category_id = %s",
                    (row["category_id"],),
                )
                if cat_vec_row:
                    cat_vec = _normalize_embedding(cat_vec_row.get("embedding"))
                    if cat_vec is not None and len(query_vec) == len(cat_vec):
                        import numpy as np
                        q = np.array(query_vec)
                        c = np.array(cat_vec)
                        norm_q = np.linalg.norm(q)
                        norm_c = np.linalg.norm(c)
                        if norm_q > 0 and norm_c > 0:
                            vec_sim = float(np.dot(q, c) / (norm_q * norm_c))
            except Exception:
                pass

        if vec_sim < vector_threshold:
            continue

        candidates.append({
            "category_id": row["category_id"],
            "category_name": row["category_name"],
            "trgm_similarity": round(trgm_sim, 4),
            "vector_similarity": round(vec_sim, 4),
        })

    return candidates


def _vector_semantic_locate(product_name: str, top_k: int = 10):
    """向量语义粗定位：计算产品向量与所有节点的相似度，返回top-k候选及路径分析"""
    if not _vec_mgr:
        return {"candidates": [], "path_analysis": {}}

    try:
        query_vec = _vec_mgr.embed_query(product_name)
    except Exception:
        return {"candidates": [], "path_analysis": {}}

    import pickle
    vec_col = "embedding"
    rows = _db.execute(
        f"""SELECT cv.category_id, ct.category_name, ct.category_group_name, cv.{vec_col} AS embedding
           FROM category_vectors cv
           JOIN category_texts ct ON cv.category_id = ct.category_id
           WHERE cv.{vec_col} IS NOT NULL"""
    )

    if not rows:
        return {"candidates": [], "path_analysis": {}}

    import numpy as np
    q = np.array(query_vec)
    norm_q = np.linalg.norm(q)
    if norm_q == 0:
        return {"candidates": [], "path_analysis": {}}

    scored = []
    for row in rows:
        vec = _normalize_embedding(row.get("embedding"))
        if vec is None:
            continue

        if len(vec) != len(query_vec):
            continue
        c = np.array(vec)
        norm_c = np.linalg.norm(c)
        if norm_c == 0:
            continue
        sim = float(np.dot(q, c) / (norm_q * norm_c))
        scored.append({
            "category_id": row["category_id"],
            "category_name": row["category_name"],
            "category_group_name": row.get("category_group_name", ""),
            "similarity": sim,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_candidates = scored[:top_k]

    path_depth_count = {}
    for c in top_candidates:
        group_name = c.get("category_group_name", "")
        if group_name:
            depth = len([p for p in group_name.split(",") if p.strip()])
            path_depth_count[depth] = path_depth_count.get(depth, 0) + 1

    common_ancestor = None
    if top_candidates:
        paths = []
        for c in top_candidates[:5]:
            group_name = c.get("category_group_name", "")
            if group_name:
                parts = [p.strip() for p in group_name.split(",") if p.strip()]
                paths.append(parts)

        if paths:
            min_len = min(len(p) for p in paths)
            for i in range(min_len):
                candidates_at_i = [p[i] for p in paths if len(p) > i]
                if len(set(candidates_at_i)) == 1:
                    common_ancestor = candidates_at_i[0]
                else:
                    break

    return {
        "candidates": [
            {
                "category_id": c["category_id"],
                "category_name": c["category_name"],
                "similarity": round(c["similarity"], 4),
                "path": c.get("category_group_name", ""),
            }
            for c in top_candidates
        ],
        "path_analysis": {
            "common_ancestor": common_ancestor,
            "path_depth_distribution": path_depth_count,
        },
    }


# 批量扩展用的向量缓存，避免每个产品扫一遍全库
_locate_matrix_cache: dict | None = None


def _invalidate_locate_matrix_cache() -> None:
    global _locate_matrix_cache
    _locate_matrix_cache = None


def _ensure_locate_matrix_cache() -> dict:
    """一次性加载 category embedding 矩阵（供批量扩展复用）。"""
    global _locate_matrix_cache
    if _locate_matrix_cache is not None:
        return _locate_matrix_cache

    import pickle
    import numpy as np

    rows = _db.execute(
        """
        SELECT cv.category_id, ct.category_name, ct.category_group_name, cv.embedding
        FROM category_vectors cv
        JOIN category_texts ct ON cv.category_id = ct.category_id
        WHERE cv.embedding IS NOT NULL
        """
    )
    ids: list[str] = []
    names: list[str] = []
    paths: list[str] = []
    vecs: list[list[float]] = []
    for row in rows:
        vec = _normalize_embedding(row.get("embedding"))
        if vec is None:
            continue
        ids.append(str(row["category_id"]))
        names.append(row["category_name"])
        paths.append(row.get("category_group_name") or "")
        vecs.append(vec)

    if not vecs:
        _locate_matrix_cache = {
            "ids": [],
            "names": [],
            "paths": [],
            "matrix": None,
            "norms": None,
        }
        return _locate_matrix_cache

    matrix = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    _locate_matrix_cache = {
        "ids": ids,
        "names": names,
        "paths": paths,
        "matrix": matrix,
        "norms": norms,
    }
    return _locate_matrix_cache


def _top_k_from_query_vec(query_vec: list[float], top_k: int = 5) -> dict:
    """用缓存矩阵做 TopK，不再每次扫库。"""
    import numpy as np

    cache = _ensure_locate_matrix_cache()
    matrix = cache.get("matrix")
    if matrix is None or len(cache["ids"]) == 0:
        return {"candidates": [], "path_analysis": {}}

    q = np.asarray(query_vec, dtype=np.float32)
    if q.ndim != 1 or q.shape[0] != matrix.shape[1]:
        return {"candidates": [], "path_analysis": {}}
    qn = float(np.linalg.norm(q))
    if qn == 0:
        return {"candidates": [], "path_analysis": {}}

    sims = (matrix @ q) / (cache["norms"] * qn)
    k = min(top_k, len(sims))
    if k <= 0:
        return {"candidates": [], "path_analysis": {}}
    # argpartition 比全排序更快
    if k < len(sims):
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    else:
        idx = np.argsort(-sims)

    top_candidates = []
    for i in idx:
        top_candidates.append({
            "category_id": cache["ids"][int(i)],
            "category_name": cache["names"][int(i)],
            "category_group_name": cache["paths"][int(i)],
            "similarity": float(sims[int(i)]),
        })

    path_depth_count: dict[int, int] = {}
    for c in top_candidates:
        group_name = c.get("category_group_name", "")
        if group_name:
            depth = len([p for p in group_name.split(",") if p.strip()])
            path_depth_count[depth] = path_depth_count.get(depth, 0) + 1

    common_ancestor = None
    paths = []
    for c in top_candidates[:5]:
        group_name = c.get("category_group_name", "")
        if group_name:
            paths.append([p.strip() for p in group_name.split(",") if p.strip()])
    if paths:
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            at_i = [p[i] for p in paths if len(p) > i]
            if len(set(at_i)) == 1:
                common_ancestor = at_i[0]
            else:
                break

    return {
        "candidates": [
            {
                "category_id": c["category_id"],
                "category_name": c["category_name"],
                "similarity": round(c["similarity"], 4),
                "path": c.get("category_group_name", ""),
            }
            for c in top_candidates
        ],
        "path_analysis": {
            "common_ancestor": common_ancestor,
            "path_depth_distribution": path_depth_count,
        },
    }


def _embed_queries_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    def _to_list(a) -> list[float]:
        if a is None:
            return []
        if hasattr(a, "tolist"):
            a = a.tolist()
        try:
            return [float(x) for x in a]
        except Exception:
            return []

    if _vec_mgr and getattr(_vec_mgr, "_api_embedder", None):
        arrs = _vec_mgr._api_embedder.embed_batch(texts)
        return [_to_list(a) for a in arrs]
    if _vec_mgr and getattr(_vec_mgr, "_onnx_embedder", None) and hasattr(_vec_mgr._onnx_embedder, "embed_batch"):
        arrs = _vec_mgr._onnx_embedder.embed_batch(texts)
        return [_to_list(a) for a in arrs]
    if _vec_mgr:
        return [_to_list(_vec_mgr.embed_query(t)) for t in texts]
    return [[] for _ in texts]


def _vec_is_empty(qvec) -> bool:
    """安全判断向量是否为空（避免 numpy 数组触发 truth-value 歧义）。"""
    if qvec is None:
        return True
    try:
        import numpy as np
        if isinstance(qvec, np.ndarray):
            return qvec.size == 0
    except Exception:
        pass
    try:
        return len(qvec) == 0
    except Exception:
        return True


def _normalize_embedding(emb) -> list[float] | None:
    """把 DB/pickle/ndarray 等形态的 embedding 统一成 list[float]；失败返回 None。"""
    if emb is None:
        return None
    import pickle
    import numpy as np

    vec = None
    try:
        if isinstance(emb, memoryview):
            vec = pickle.loads(bytes(emb))
        elif isinstance(emb, (bytes, bytearray)):
            vec = pickle.loads(bytes(emb))
        elif isinstance(emb, str):
            s = emb.strip()
            if s.startswith("["):
                vec = [float(v) for v in s.strip("[]").split(",") if v.strip()]
            else:
                return None
        elif isinstance(emb, np.ndarray):
            vec = emb.tolist()
        elif isinstance(emb, list):
            vec = emb
        else:
            # 某些驱动可能直接给出 array-like
            if hasattr(emb, "tolist"):
                vec = emb.tolist()
            else:
                return None
    except Exception:
        return None

    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    if _vec_is_empty(vec):
        return None
    try:
        return [float(x) for x in vec]
    except Exception:
        return None


def _build_expansion_suggestion(product_name: str, vector_candidates: list, path_analysis: dict, llm_result) -> dict:
    llm_path_validation = None
    if llm_result:
        llm_full_path = llm_result.get("full_path", "")
        if llm_full_path:
            llm_path_validation = _validate_path_nodes(llm_full_path)

    recommendation = None
    decision_reason = ""
    candidate_ids = {
        str(c.get("category_id", "")) for c in vector_candidates if c.get("category_id")
    }

    def _names_compatible(a: str, b: str) -> bool:
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b:
            return True
        return a == b or a in b or b in a

    if llm_result:
        llm_confidence = float(llm_result.get("confidence") or 0.0)
        llm_parent_id = str(llm_result.get("suggested_parent_id") or "")
        llm_parent_name = str(llm_result.get("suggested_parent_name") or "")
        llm_should_create = bool(llm_result.get("should_create_new_node", False))
        decision_reason = f"LLM置信度{llm_confidence:.2f}"
        if llm_confidence >= 0.85:
            decision_reason = f"LLM置信度{llm_confidence:.2f}≥0.85，采纳LLM建议"
        elif llm_should_create:
            decision_reason = "LLM建议创建新节点，优先采纳"
        elif llm_confidence >= 0.7:
            decision_reason = f"LLM置信度{llm_confidence:.2f}≥0.7，采纳LLM建议"

        pid = llm_parent_id[1:] if llm_parent_id.startswith("#") else llm_parent_id
        if pid and pid in candidate_ids:
            recommendation = _lookup_category_recommendation(
                parent_id=pid,
                parent_name=llm_parent_name,
                should_create_new_node=llm_should_create,
                new_node_name=llm_result.get("new_node_name", ""),
                confidence=llm_confidence,
                reason=llm_result.get("reasoning", ""),
                decision_reason=decision_reason + "（父节点来自向量候选）",
                trust_llm=True,
                llm_weight="high" if llm_confidence >= 0.85 else "medium",
            )
        elif llm_parent_name:
            by_name = _lookup_category_recommendation(
                parent_id="",
                parent_name=llm_parent_name,
                should_create_new_node=llm_should_create,
                new_node_name=llm_result.get("new_node_name", ""),
                confidence=llm_confidence,
                reason=llm_result.get("reasoning", ""),
                decision_reason=decision_reason + "（按父节点名称解析）",
                trust_llm=True,
                llm_weight="medium",
            )
            if by_name:
                recommendation = by_name
            elif pid:
                by_id = _lookup_category_recommendation(
                    parent_id=pid,
                    parent_name="",
                    should_create_new_node=llm_should_create,
                    new_node_name=llm_result.get("new_node_name", ""),
                    confidence=llm_confidence,
                    reason=llm_result.get("reasoning", ""),
                    decision_reason=decision_reason,
                    trust_llm=True,
                    llm_weight="medium",
                )
                if by_id and _names_compatible(llm_parent_name, by_id["parent_name"]):
                    recommendation = by_id

        if not recommendation and llm_path_validation:
            for seg in reversed(llm_path_validation.get("path_segments") or []):
                if seg.get("exists") and seg.get("node_id"):
                    recommendation = _lookup_category_recommendation(
                        parent_id=str(seg["node_id"]),
                        parent_name=seg.get("name", ""),
                        should_create_new_node=True,
                        new_node_name=llm_result.get("new_node_name") or product_name,
                        confidence=llm_confidence,
                        reason=llm_result.get("reasoning", ""),
                        decision_reason="LLM父节点无效，挂到路径中最近的已存在节点",
                        trust_llm=True,
                        llm_weight="medium",
                    )
                    break

    if not recommendation and vector_candidates:
        recommendation = _recommendation_from_vector(
            vector_candidates, decision_reason=decision_reason
        )

    return {
        "product_name": product_name,
        "status": "no_match",
        "vector_candidates": vector_candidates,
        "path_analysis": path_analysis,
        "llm_reasoning": llm_result,
        "llm_path_validation": llm_path_validation,
        "recommendation": recommendation,
    }


def _validate_path_nodes(full_path: str):
    """验证路径中的节点是否存在，返回带标记的路径信息"""
    if not full_path:
        return {"path_segments": [], "has_missing_nodes": False}
    
    path_nodes = [node.strip() for node in full_path.split(">")]
    path_segments = []
    has_missing_nodes = False
    
    for i, node_name in enumerate(path_nodes):
        if not node_name:
            continue
        
        exists = False
        node_id = None
        
        row = _db.execute_one(
            "SELECT category_id FROM category_texts WHERE category_name = %s LIMIT 1",
            (node_name,)
        )
        
        if row:
            exists = True
            node_id = row["category_id"]
        else:
            has_missing_nodes = True
        
        path_segments.append({
            "name": node_name,
            "exists": exists,
            "node_id": node_id,
            "level": i + 1,
            "status": "existing" if exists else "missing"
        })
    
    return {
        "path_segments": path_segments,
        "has_missing_nodes": has_missing_nodes,
        "original_path": full_path
    }


def _lookup_category_recommendation(
    parent_id: str = "",
    parent_name: str = "",
    *,
    should_create_new_node: bool = False,
    new_node_name: str = "",
    confidence: float = 0.0,
    reason: str = "",
    decision_reason: str = "",
    trust_llm: bool = False,
    llm_weight: str = "low",
) -> dict | None:
    """将父节点 id/名称解析为可执行的 recommendation；节点不存在则返回 None。"""
    pid = str(parent_id or "").strip()
    if pid.startswith("#"):
        pid = pid[1:]
    pname = (parent_name or "").strip()

    row = None
    if pid and pid.isdigit():
        row = _db.execute_one(
            "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_id = %s",
            (pid,),
        )
    if not row and pname:
        row = _db.execute_one(
            "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_name = %s LIMIT 1",
            (pname,),
        )
    if not row and pid and not pid.isdigit():
        row = _db.execute_one(
            "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_name = %s LIMIT 1",
            (pid,),
        )
    if not row:
        return None

    path = row.get("category_group_name") or ""
    full_path = (
        path.replace(",", " > ") + " > " + row["category_name"]
        if path
        else row["category_name"]
    )
    return {
        "parent_id": str(row["category_id"]),
        "parent_name": row["category_name"],
        "full_path": full_path,
        "should_create_new_node": should_create_new_node,
        "new_node_name": new_node_name,
        "confidence": confidence,
        "reason": reason,
        "decision_reason": decision_reason,
        "trust_llm": trust_llm,
        "llm_weight": llm_weight,
    }


def _recommendation_from_vector(
    vector_candidates: list,
    decision_reason: str = "",
) -> dict | None:
    if not vector_candidates:
        return None
    top = vector_candidates[0]
    return _lookup_category_recommendation(
        parent_id=str(top.get("category_id", "")),
        parent_name=top.get("category_name", ""),
        confidence=float(top.get("similarity") or 0),
        reason="基于向量相似度定位（LLM未给出可用父节点）",
        decision_reason=decision_reason or "回退到向量Top1",
        trust_llm=False,
        llm_weight="low",
    )


def _llm_path_reasoning(product_name: str, vector_candidates: list):
    """LLM路径推理：必须锚定向量召回的真实节点，禁止自由编造体系路径。"""
    if not _llm:
        return None

    root_rows = _db.execute("SELECT category_id, category_name FROM category_texts WHERE category_pids = '{}' LIMIT 30")
    taxonomy_overview = ", ".join([r["category_name"] for r in root_rows]) if root_rows else ""

    candidate_info = ""
    for i, c in enumerate(vector_candidates[:8], 1):
        path = c.get("path", "")
        path_display = path.replace(",", " > ") if path else c["category_name"]
        candidate_info += (
            f"{i}. #{c['category_id']} {c['category_name']} "
            f"(向量相似度{c['similarity']:.3f})\n   真实路径: {path_display}\n"
        )

    prompt = f"""你是标准产品分类体系的编纂助手。产品未能精确匹配现有分类，请建议「挂到哪个已有父节点下、新建一个正式的标准子分类名」。

产品名称: {product_name}

体系一级分类（对照用）:
{taxonomy_overview}

向量召回的真实节点（唯一可信证据）:
{candidate_info}

命名与挂载硬性规则：
1. suggested_parent_id 必须来自候选 ID（可写 #数字），禁止编造。
2. new_node_name 必须是正式、通用的标准分类名，能覆盖一类产品，而不是单个SKU/型号/款式。
3. 严禁把产品名原样、或仅加前缀/后缀的产品级名称当作 new_node_name。
   反例：产品「T型丝锥扳手」→ 禁止新建「T型丝锥扳手」；应新建「丝锥扳手」，父节点选「钻孔或攻丝工具」等更上位已有节点。
   反例：产品「304不锈钢弯头」→ 禁止「304不锈钢弯头」；宜「管件弯头」或「不锈钢管件」。
4. 去掉型号/规格/材质牌号/形状代号后再抽象：T型/L型/U型、十字/梅花、内六角、数字规格、牌号等都不进入分类名。
5. 若候选里已有足够合适的正式类名（如已有「丝锥扳手」），不要再在其下建更细的产品叶子；父节点应选其上一级，new_node_name 用同级正式类名；若该类已存在则 should_create_new_node=false。
6. 对黑话/多义词禁止望文生义，以向量近邻行业域为准；近邻冲突时 confidence≤0.55。
7. full_path = 父节点真实路径 > new_node_name（仅末级允许新建）。

请以JSON返回:
{{
  "product_analysis": "产品行业属性；若名称歧义请写明",
  "primary_category": "一级分类",
  "suggested_parent_id": "#候选ID",
  "suggested_parent_name": "父节点名",
  "should_create_new_node": true,
  "new_node_name": "正式标准分类名（禁止等于产品名）",
  "full_path": "真实父路径 > 正式分类名",
  "reasoning": "为何该正式类名可覆盖同类产品",
  "confidence": 0.0-1.0
}}"""

    try:
        response = _llm._call_llm(prompt, method="path_reasoning")
        result = _llm._parse_json_response(response)
        return result
    except Exception as e:
        logging.getLogger("app").warning(f"LLM路径推理失败: {e}")
        return None


def _strip_product_modifiers(name: str) -> str:
    """去掉型号/形状/规格等产品级修饰，得到更接近标准类名的主干。"""
    import re
    s = (name or "").strip()
    if not s:
        return ""
    # 先去牌号/数字规格前缀（如 304不锈钢…）
    s = re.sub(r"^\d{2,4}(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(
        r"^(?:[A-Za-z0-9]{1,4}型|十字|梅花|内六角|外六角|双向|单向|电动|手动|气动|液压|"
        r"不锈钢|镀锌|碳钢|合金|精密|微型|小型|大型|重型|轻型|便携式|固定式|移动式)+",
        "",
        s,
    )
    s = re.sub(r"(?:^|[\-_/])\d+(?:\.\d+)?(?:mm|cm|MM|CM|号|#)?(?=$|[\-_/])", "", s)
    s = re.sub(r"^[\d\-_.]+", "", s)
    s = re.sub(r"[\d\-_.]+$", "", s)
    return re.sub(r"\s+", "", s).strip("-_/ ")


def _formalize_new_category_name(
    product_name: str,
    proposed: str,
    parent_name: str,
) -> str:
    """将拟新建名规范为正式分类名：不得等于产品名，避免过细叶子。"""
    from src.data.text_similarity import chinese_text_similarity
    import re

    product_name = (product_name or "").strip()
    parent_name = (parent_name or "").strip()
    name = (proposed or "").strip()

    def _too_specific(n: str) -> bool:
        if not n:
            return True
        if n == product_name:
            return True
        if re.match(r"^[A-Za-z0-9]{1,4}型", n):
            return True
        if chinese_text_similarity(n, product_name) >= 0.92:
            return True
        # 与产品几乎同长且高度相似 → 产品级叶子
        if (
            product_name
            and abs(len(n) - len(product_name)) <= 1
            and chinese_text_similarity(n, product_name) >= 0.85
        ):
            return True
        return False

    for candidate in (name, _strip_product_modifiers(name), _strip_product_modifiers(product_name)):
        c = (candidate or "").strip()
        if not c or c == parent_name:
            continue
        if _too_specific(c):
            c2 = _strip_product_modifiers(c)
            if c2 and c2 != parent_name and not _too_specific(c2):
                return c2
            continue
        return c

    if parent_name:
        fallback = f"其他{parent_name}"
        if fallback != product_name:
            return fallback
    return "其他专用制品"


def _choose_expansion_parent(
    product_name: str,
    vector_candidates: list,
    preferred_parent_id: str = "",
) -> tuple[str, str]:
    """选择挂载父节点：若近邻已是产品细类/近同名，则上提到更上位正式节点。"""
    from src.data.text_similarity import chinese_text_similarity

    def _lookup(pid: str) -> tuple[str, str] | None:
        pid = str(pid or "").strip().lstrip("#")
        if not pid:
            return None
        row = _db.execute_one(
            "SELECT category_id, category_name FROM category_texts WHERE category_id = %s",
            (pid,),
        )
        if not row:
            return None
        return str(row["category_id"]), row["category_name"] or ""

    def _should_promote(cat_name: str) -> bool:
        if not cat_name or not product_name:
            return False
        if cat_name == product_name:
            return True
        if cat_name in product_name or product_name in cat_name:
            return True
        return chinese_text_similarity(cat_name, product_name) >= 0.75

    def _parent_from_group(category_id: str, group_name: str, self_name: str) -> tuple[str, str] | None:
        segs = [p.strip() for p in (group_name or "").split(",") if p.strip()]
        while segs and segs[-1] == self_name:
            segs.pop()
        if not segs:
            return None
        up_name = segs[-1]
        row = _db.execute_one(
            "SELECT category_id, category_name FROM category_texts WHERE category_name = %s LIMIT 1",
            (up_name,),
        )
        if row and str(row["category_id"]) != str(category_id):
            return str(row["category_id"]), row["category_name"]
        return None

    hit = _lookup(preferred_parent_id)
    cands = vector_candidates or []
    top = cands[0] if cands else None

    if hit and _should_promote(hit[1]):
        row = _db.execute_one(
            "SELECT category_group_name FROM category_texts WHERE category_id = %s",
            (hit[0],),
        )
        promoted = _parent_from_group(hit[0], (row or {}).get("category_group_name") or "", hit[1])
        if promoted:
            return promoted
    if hit:
        return hit

    if top:
        tid, tname = str(top["category_id"]), top.get("category_name") or ""
        if _should_promote(tname):
            promoted = _parent_from_group(tid, top.get("path") or "", tname)
            if promoted:
                return promoted
        looked = _lookup(tid)
        if looked:
            return looked
    return "", ""


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/match", methods=["POST"])
def api_match():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        engine = data.get("engine", "rag")
        pageindex_mode = data.get("pageindex_mode", "default")

        if not product_name:
            return jsonify({"error": "product_name is required"}), 400

        if engine == "rag_rerank":
            engine_obj = _rag_rerank_engine
        elif engine == "rag":
            engine_obj = _rag_engine
        elif engine in ("pageindex", "page_index"):
            engine_obj = (
                _page_engine_force_llm
                if pageindex_mode == "force_llm"
                else _page_engine
            )
        elif pageindex_mode == "force_llm":
            engine_obj = _page_engine_force_llm
        else:
            engine_obj = _page_engine

        if engine_obj is None:
            return jsonify({"error": "匹配引擎未初始化，请检查数据库连接并重启服务"}), 500

        start = time.perf_counter()
        result = engine_obj.match(product_name)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        candidates = []
        for c in result.candidates:
            candidates.append({
                "category_id": c.category_id,
                "category_name": c.category_name,
                "coarse_score": round(c.coarse_score, 4),
                "llm_score": round(c.llm_score, 4),
                "final_confidence": round(c.final_confidence, 4),
                "path_depth": c.path_depth,
                "path_total": c.path_total,
                "is_match_target": c.is_match_target,
            })

        return jsonify({
            "product_name": result.product_name,
            "matched_category_id": result.matched_category_id,
            "confidence": round(result.confidence, 4),
            "match_status": result.match_status.value,
            "engine_type": result.engine_type.value,
            "pageindex_mode": pageindex_mode if result.engine_type == EngineType.PAGE_INDEX else None,
            "llm_participated": result.llm_participated,
            "elapsed_ms": elapsed_ms,
            "candidates": candidates,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        category_id = data.get("category_id", "").strip()
        if not product_name or not category_id:
            return jsonify({"error": "product_name and category_id required"}), 400

        existing = _db.execute_one(
            "SELECT syn_list FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        if not existing:
            return jsonify({"error": f"category_id={category_id} not found"}), 404

        if product_name in existing["syn_list"]:
            return jsonify({"status": "already_exists", "message": f"{product_name} already in syn_list"})

        cat_row = _db.execute_one(
            "SELECT category_name FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        cat_name = cat_row["category_name"] if cat_row else ""

        from src.data.synonym_sanitizer import sanitize_syn_list
        cleaned, removed = sanitize_syn_list([product_name], cat_name)
        if removed or not cleaned:
            return jsonify({"status": "rejected", "message": "synonym rejected by sanitizer"})

        _db.execute(
            "UPDATE category_texts SET syn_list = array_append(syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, category_id),
        )
        _db.execute(
            "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, category_id),
        )
        _db.execute(
            "INSERT INTO synonym_updates (category_id, new_synonym, llm_verified, trigger_reason, status) VALUES (%s, %s, %s, %s, %s)",
            (category_id, product_name, True, "用户确认", "COMPLETED"),
        )

        return jsonify({"status": "ok", "message": f"已将 '{product_name}' 添加为 #{category_id} 的同义词"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correct", methods=["POST"])
def api_correct():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        correct_category_id = data.get("correct_category_id", "").strip()
        if not product_name or not correct_category_id:
            return jsonify({"error": "product_name and correct_category_id required"}), 400

        existing = _db.execute_one(
            "SELECT syn_list FROM category_texts WHERE category_id = %s",
            (correct_category_id,),
        )
        if not existing:
            return jsonify({"error": f"category_id={correct_category_id} not found"}), 404

        if product_name in existing["syn_list"]:
            return jsonify({"status": "already_exists", "message": f"{product_name} already in syn_list"})

        cat_row = _db.execute_one(
            "SELECT category_name FROM category_texts WHERE category_id = %s",
            (correct_category_id,),
        )
        cat_name = cat_row["category_name"] if cat_row else ""

        from src.data.synonym_sanitizer import sanitize_syn_list
        cleaned, removed = sanitize_syn_list([product_name], cat_name)
        if removed or not cleaned:
            return jsonify({"status": "rejected", "message": "synonym rejected by sanitizer"})

        _db.execute(
            "UPDATE category_texts SET syn_list = array_append(syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, correct_category_id),
        )
        _db.execute(
            "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, correct_category_id),
        )
        _db.execute(
            "INSERT INTO synonym_updates (category_id, new_synonym, llm_verified, trigger_reason, status) VALUES (%s, %s, %s, %s, %s)",
            (correct_category_id, product_name, True, "用户纠正", "COMPLETED"),
        )

        return jsonify({"status": "ok", "message": f"已纠正: '{product_name}' -> #{correct_category_id}, 同义词已追加"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search_category", methods=["GET"])
def api_search_category():
    try:
        _init_components()
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify([])
        rows = _db.execute(
            """SELECT category_id, category_name FROM category_texts
               WHERE category_name ILIKE %s OR %s = ANY(syn_list)
               LIMIT 20""",
            (f"%{query}%", query),
        )
        return jsonify([{"category_id": r["category_id"], "category_name": r["category_name"]} for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        _init_components()
        rows = _db.execute("SELECT COUNT(*) as cnt FROM category_vectors")
        vec_count = rows[0]["cnt"] if rows else 0

        rows2 = _db.execute("SELECT COUNT(*) as cnt FROM category_texts")
        txt_count = rows2[0]["cnt"] if rows2 else 0

        rows3 = _db.execute("SELECT COUNT(*) as cnt FROM match_results")
        match_count = rows3[0]["cnt"] if rows3 else 0

        rows4 = _db.execute("SELECT COUNT(*) as cnt FROM synonym_updates")
        syn_count = rows4[0]["cnt"] if rows4 else 0

        rows5 = _db.execute(
            "SELECT COUNT(*) as cnt FROM expansion_suggestions")
        exp_count = rows5[0]["cnt"] if rows5 else 0

        status_rows = _db.execute(
            "SELECT match_status, COUNT(*) as cnt FROM match_results GROUP BY match_status"
        )
        status_dist = {r["match_status"]: r["cnt"] for r in status_rows}

        return jsonify({
            "vector_count": vec_count,
            "text_count": txt_count,
            "match_count": match_count,
            "synonym_count": syn_count,
            "expansion_count": exp_count,
            "status_distribution": status_dist,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/match_history", methods=["GET"])
def api_match_history():
    try:
        _init_components()
        limit = request.args.get("limit", 50, type=int)
        rows = _db.execute(
            """SELECT m.product_name, m.matched_category_id, m.confidence, m.match_status, m.engine_type, m.llm_participated, m.created_at,
                      c.category_name as matched_category_name
               FROM match_results m
               LEFT JOIN category_texts c ON m.matched_category_id = c.category_id
               ORDER BY m.created_at DESC LIMIT %s""",
            (limit,),
        )
        results = []
        for r in rows:
            results.append({
                "product_name": r["product_name"],
                "matched_category_id": r["matched_category_id"],
                "matched_category_name": r.get("matched_category_name", ""),
                "confidence": float(r["confidence"]) if r["confidence"] else 0,
                "match_status": r["match_status"],
                "engine_type": r["engine_type"],
                "llm_participated": r["llm_participated"],
                "created_at": str(r["created_at"]),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/synonym_history", methods=["GET"])
def api_synonym_history():
    try:
        _init_components()
        limit = request.args.get("limit", 30, type=int)
        rows = _db.execute(
            """SELECT category_id, new_synonym, llm_verified, trigger_reason, trgm_similarity, match_confidence, created_at
               FROM synonym_updates ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        results = []
        for r in rows:
            results.append({
                "category_id": r["category_id"],
                "new_synonym": r["new_synonym"],
                "llm_verified": r["llm_verified"],
                "trigger_reason": r["trigger_reason"],
                "trgm_similarity": float(r["trgm_similarity"]) if r["trgm_similarity"] else None,
                "match_confidence": float(r["match_confidence"]) if r["match_confidence"] else None,
                "created_at": str(r["created_at"]),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/synonym/suggest_from_matches", methods=["POST"])
def api_synonym_suggest_from_matches():
    """从批量 MATCHED 结果筛同义词候选（对齐 SelfEvolveScheduler 意图）。

    准入条件：
    1. MATCHED 且置信度 >= syn_confidence_threshold
    2. 产品名 ≠ 分类名、不在 syn_list、通过清洗
    3. 字面相似度 < syn_trgm_threshold（中文 SequenceMatcher/bigram，
       非 pg_trgm——后者对纯中文恒为 0）
    4. 可选：产品名↔分类名向量余弦作参考（不过度用于硬过滤，
       因真正同义词向量也往往很高）
    5. LLM 同义校验：≥0.7 强候选；≥0.5 进人工；否则丢弃
    """
    try:
        _init_components()
        data = request.get_json(force=True) or {}
        items = data.get("items") or []
        if not items:
            return jsonify({"error": "items required"}), 400

        from src.data.synonym_sanitizer import sanitize_syn_list, GENERIC_SHORT_SYNONYMS
        from src.data.text_similarity import chinese_text_similarity, cosine_similarity

        match_cfg = _config.get_match_config()
        # 人工复核默认用 MATCHED 线（low_confidence）；auto_strict 才用 0.95
        auto_strict = bool(data.get("auto_strict", False))
        if data.get("min_confidence") is not None:
            conf_th = float(data.get("min_confidence"))
        elif auto_strict:
            conf_th = float(getattr(match_cfg, "syn_confidence_threshold", 0.95) or 0.95)
        else:
            # 测试/复核：只要 MATCHED 即可进列表（默认 0）
            conf_th = 0.0
        # 字面近重复上限；复核模式默认放宽到 0.92，便于测试集看到候选
        if data.get("text_threshold") is not None:
            text_th = float(data.get("text_threshold"))
        elif auto_strict:
            text_th = float(getattr(match_cfg, "syn_trgm_threshold", 0.65) or 0.65)
        else:
            text_th = float(data.get("text_threshold", 0.92))
        # 批量导入待复核默认不做 LLM（快、且不会被 LLM 全滤掉）；需要时传 use_llm=true
        use_llm = bool(data.get("use_llm", False)) and _llm is not None
        use_vec = bool(data.get("use_vec", False)) and _vec_mgr is not None

        suggestions = []
        skipped = []
        llm_candidates = []
        score_samples = []  # 便于前端/调试看分数分布

        for item in items:
            product_name = (item.get("product_name") or item.get("input") or "").strip()
            category_id = str(item.get("matched_category_id") or item.get("category_id") or "").strip()
            confidence = float(item.get("confidence") or 0)
            match_status = (item.get("match_status") or "MATCHED").strip().upper()
            if not product_name or not category_id:
                continue

            if match_status != "MATCHED":
                skipped.append({"product_name": product_name, "reason": f"状态={match_status}（仅 MATCHED）"})
                continue

            if confidence < conf_th:
                skipped.append({
                    "product_name": product_name,
                    "reason": f"置信度 {confidence:.4f} < 阈值 {conf_th}",
                })
                continue

            row = _db.execute_one(
                "SELECT category_id, category_name, category_group_name, syn_list "
                "FROM category_texts WHERE category_id = %s",
                (category_id,),
            )
            if not row:
                skipped.append({"product_name": product_name, "reason": f"分类#{category_id}不存在"})
                continue

            cat_name = row["category_name"] or ""
            syn_list = row.get("syn_list") or []

            if product_name == cat_name:
                skipped.append({"product_name": product_name, "reason": "与分类名相同"})
                continue
            if product_name in syn_list:
                skipped.append({"product_name": product_name, "reason": "已是同义词"})
                continue
            if product_name in GENERIC_SHORT_SYNONYMS:
                skipped.append({"product_name": product_name, "reason": "泛词短同义词"})
                continue

            # pg_trgm 对纯中文无效（show_trgm=[] → sim=0）；改用中文字面相似度
            text_sim = chinese_text_similarity(product_name, cat_name)
            pg_trgm = 0.0
            if _trgm_mgr is not None:
                try:
                    pg_trgm = float(_trgm_mgr.get_trgm_similarity(product_name, cat_name) or 0)
                except Exception:
                    pg_trgm = 0.0

            score_samples.append({
                "product_name": product_name,
                "category_name": cat_name,
                "text_similarity": round(text_sim, 4),
                "pg_trgm": round(pg_trgm, 4),
                "match_confidence": round(confidence, 4),
            })

            if text_sim >= text_th:
                skipped.append({
                    "product_name": product_name,
                    "reason": (
                        f"与分类名字面过像 text_sim={text_sim:.3f}≥{text_th}"
                        f"（pg_trgm={pg_trgm:.3f}，中文通常为0）"
                    ),
                })
                continue

            cleaned, removed = sanitize_syn_list([product_name], cat_name)
            if removed or not cleaned:
                skipped.append({"product_name": product_name, "reason": "被清洗规则拒绝"})
                continue

            path = row.get("category_group_name") or ""
            full_path = (
                path.replace(",", " > ") + " > " + cat_name if path else cat_name
            )
            llm_candidates.append({
                "product_name": product_name,
                "category_id": category_id,
                "cat_name": cat_name,
                "full_path": full_path,
                "confidence": confidence,
                "text_sim": text_sim,
                "pg_trgm": pg_trgm,
                "vec_sim": None,
            })

        # 批量算产品名↔分类名向量余弦（参考分，不作为硬过滤）
        if use_vec and llm_candidates:
            try:
                flat = []
                for c in llm_candidates:
                    flat.append(c["product_name"])
                    flat.append(c["cat_name"])
                vecs = _embed_queries_batch(flat)
                for i, c in enumerate(llm_candidates):
                    v1 = vecs[2 * i] if 2 * i < len(vecs) else None
                    v2 = vecs[2 * i + 1] if 2 * i + 1 < len(vecs) else None
                    c["vec_sim"] = cosine_similarity(v1, v2)
                    if i < len(score_samples):
                        # 对齐到仍在 hard-pass 的样本较难，直接挂在 candidate 上即可
                        pass
            except Exception as ve:
                logging.getLogger("WebAPI").warning(f"同义词向量相似度计算失败: {ve}")

        def _pack_suggestion(c: dict, llm_ok: bool, llm_conf: float, llm_reason: str) -> dict:
            vec_s = c.get("vec_sim")
            vec_part = f", vec={vec_s:.3f}" if isinstance(vec_s, (int, float)) else ""
            default_reason = (
                f"高置信匹配「{c['cat_name']}」，字面差异大"
                f"(text_sim={c['text_sim']:.3f}{vec_part})，适合作同义词候选"
            )
            return {
                "product_name": c["product_name"],
                "status": "synonym_candidate",
                "source": "matched",
                "matched_category_id": c["category_id"],
                "matched_category_name": c["cat_name"],
                "confidence": round(c["confidence"], 4),
                "text_similarity": round(c["text_sim"], 4),
                "trgm_similarity": round(c["pg_trgm"], 4),  # 兼容旧字段；中文多为 0
                "vec_similarity": round(c["vec_sim"], 4) if c.get("vec_sim") is not None else None,
                "vector_candidates": [],
                "path_analysis": {},
                "llm_reasoning": {
                    "product_analysis": llm_reason or default_reason,
                    "full_path": c["full_path"],
                    "confidence": llm_conf if llm_conf else c["confidence"],
                    "is_synonym": llm_ok,
                    "text_similarity": c["text_sim"],
                    "pg_trgm": c["pg_trgm"],
                    "vec_similarity": c.get("vec_sim"),
                },
                "llm_path_validation": None,
                "recommendation": {
                    "parent_id": str(c["category_id"]),
                    "parent_name": c["cat_name"],
                    "full_path": c["full_path"],
                    "should_create_new_node": False,
                    "new_node_name": "",
                    "confidence": float(llm_conf) if llm_conf else float(c["confidence"]),
                    "reason": llm_reason or "高置信+低字面相似度 → 同义词候选",
                    "decision_reason": (
                        f"MATCHED conf≥{conf_th}, text_sim<{text_th}"
                        + (", LLM通过" if llm_ok else (", LLM待人工" if use_llm else ""))
                    ),
                    "trust_llm": bool(llm_ok),
                    "llm_weight": "high" if llm_ok else "medium",
                },
            }

        if not use_llm:
            for c in llm_candidates:
                suggestions.append(_pack_suggestion(c, False, 0.0, ""))
        else:
            def _verify_one(c: dict):
                try:
                    verify = _llm.synonym_verification(c["product_name"], c["cat_name"])
                    return c, verify
                except Exception as e:
                    from src.models.evolve_models import SynonymVerifyResult
                    return c, SynonymVerifyResult(
                        is_synonym=False, confidence=0, reason=f"校验失败: {e}"
                    )

            workers = min(8, max(len(llm_candidates), 1))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_verify_one, c) for c in llm_candidates]
                for fut in as_completed(futures):
                    c, verify = fut.result()
                    if verify.is_synonym and verify.confidence >= 0.7:
                        suggestions.append(
                            _pack_suggestion(c, True, verify.confidence, verify.reason)
                        )
                    elif verify.confidence >= 0.5:
                        suggestions.append(
                            _pack_suggestion(
                                c, False, verify.confidence,
                                verify.reason or "LLM不确定，需人工确认",
                            )
                        )
                    else:
                        skipped.append({
                            "product_name": c["product_name"],
                            "reason": (
                                f"LLM判定非同义 conf={verify.confidence:.2f}"
                                + (f"：{verify.reason}" if verify.reason else "")
                            ),
                        })

        return jsonify({
            "status": "ok",
            "total_input": len(items),
            "suggestion_count": len(suggestions),
            "skipped_count": len(skipped),
            "hard_pass_count": len(llm_candidates),
            "rules": {
                "syn_confidence_threshold": conf_th,
                "syn_text_threshold": text_th,
                "syn_trgm_threshold": text_th,
                "text_metric": "chinese_text_similarity",
                "pg_trgm_note": "纯中文 pg_trgm 恒为0，已弃用为过滤依据",
                "auto_strict": auto_strict,
                "use_llm": use_llm,
                "use_vec": use_vec,
            },
            "score_samples": score_samples[:40],
            "suggestions": suggestions,
            "skipped": skipped[:80],
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion_history", methods=["GET"])
def api_expansion_history():
    try:
        _init_components()
        limit = request.args.get("limit", 30, type=int)
        rows = _db.execute(
            """SELECT product_name, suggested_parent_id, suggested_category_name, llm_analysis, status, created_at
               FROM expansion_suggestions ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        results = []
        for r in rows:
            results.append({
                "product_name": r["product_name"],
                "suggested_parent_id": r["suggested_parent_id"],
                "suggested_category_name": r["suggested_category_name"],
                "llm_analysis": r["llm_analysis"],
                "status": r["status"],
                "created_at": str(r["created_at"]),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/suggest_expansion", methods=["POST"])
def api_suggest_expansion():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400

        existing = _db.execute(
            "SELECT id FROM expansion_suggestions WHERE product_name = %s AND status = 'PENDING_REVIEW'",
            (product_name,),
        )
        if existing:
            return jsonify({"status": "already_pending", "product_name": product_name, "message": f"'{product_name}' 已有待审核建议"})

        from src.data.taxonomy_utils import suggest_expansion_path
        result = suggest_expansion_path(_llm, _page_tree, _db, product_name)

        path_text = " > ".join(
            f"★{n['category_name']}" if n["is_new"] else n["category_name"]
            for n in result["path"]
        )

        _db.execute(
            """INSERT INTO expansion_suggestions
               (product_name, suggested_parent_id, suggested_category_name, suggested_level_position, llm_analysis, status)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (product_name, result["suggested_parent_id"] or None, result["suggested_category_name"],
             path_text or None,
             f"置信度={result['confidence']:.2f} | {result['llm_reason']}",
             "PENDING_REVIEW"),
        )

        return jsonify({
            "status": "ok",
            "product_name": result["product_name"],
            "path": result["path"],
            "suggested_parent_id": result["suggested_parent_id"],
            "suggested_category_name": result["suggested_category_name"],
            "confidence": result["confidence"],
            "llm_reason": result["llm_reason"],
            "sibling_nodes": result["sibling_nodes"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/approve_expansion", methods=["POST"])
def api_approve_expansion():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        parent_id = data.get("parent_id", "").strip()
        category_name = data.get("category_name", product_name).strip()
        if not product_name or not parent_id:
            return jsonify({"error": "product_name and parent_id required"}), 400

        parent_node = _page_tree.get_node(parent_id)
        if not parent_node:
            return jsonify({"error": f"parent_id={parent_id} not found in tree"}), 404

        new_id = allocate_next_category_id(_db)
        if _page_tree.get_node(new_id) or _db.execute_one(
            "SELECT 1 FROM category_texts WHERE category_id = %s", (new_id,)
        ):
            return jsonify({"error": f"分配的新 category_id={new_id} 已存在，请检查 id 分配逻辑"}), 409

        category_pids, category_group_name = build_category_path_fields(
            _page_tree, parent_id)
        mount_path = format_category_path(_page_tree, parent_id)

        _db.execute(
            """INSERT INTO category_texts (category_id, category_name, category_pids, syn_list, category_group_name)
               VALUES (%s, %s, %s, %s, %s)""",
            (new_id, category_name, category_pids,
             [product_name], category_group_name),
        )

        from src.index.api_embedder import ApiEmbedder
        embedder = ApiEmbedder(
            api_key=_config.get_embedding_config().api_key,
            base_url=_config.get_embedding_config().base_url,
            model=_config.get_embedding_config().model,
            embedding_dim=_config.get_embedding_config().dimension,
        )
        text_parts = [category_name, product_name]
        embedding = embedder.embed(" ".join(text_parts))
        import numpy as np
        emb_list = embedding.tolist() if isinstance(
            embedding, np.ndarray) else list(embedding)

        import pickle
        embedding_bytes = pickle.dumps(embedding)

        _db.execute(
            """INSERT INTO category_vectors (category_id, category_name, embedding, syn_list)
               VALUES (%s, %s, %s, %s)""",
            (new_id, category_name, embedding_bytes, [product_name]),
        )

        vec_str = "[" + ",".join(str(float(v)) for v in emb_list) + "]"
        try:
            _db.execute(
                "UPDATE category_vectors SET vec_bgem3 = %s::vector WHERE category_id = %s",
                (vec_str, new_id),
            )
        except Exception as vec_err:
            import logging as _log
            _log.getLogger("app").warning(f"vec_bgem3写入失败(非致命): {vec_err}")

        _page_tree.add_node(new_id, category_name, parent_id, [product_name])

        try:
            _vec_mgr.invalidate_matrix()
        except Exception:
            pass

        verify_ok = False
        try:
            verify_result = _rag_rerank_engine.match(product_name)
            if verify_result.matched_category_id == new_id and verify_result.confidence >= 0.3:
                verify_ok = True
        except Exception:
            pass

        if not verify_ok:
            try:
                _db.execute(
                    "DELETE FROM category_texts WHERE category_id = %s", (new_id,))
                _db.execute(
                    "DELETE FROM category_vectors WHERE category_id = %s", (new_id,))
                node = _page_tree.get_node(new_id)
                if node and node.parent:
                    node.parent.children = [
                        c for c in node.parent.children if c.category_id != new_id]
                if new_id in _page_tree._node_map:
                    del _page_tree._node_map[new_id]
                return jsonify({"status": "verify_failed", "error": f"验证失败: 匹配'{product_name}'未能命中新分类#{new_id}，已自动回滚"})
            except Exception as rollback_err:
                return jsonify({"status": "verify_failed_rollback_error", "error": f"验证失败且回滚出错: {rollback_err}"})

        _db.execute(
            """UPDATE expansion_suggestions SET status = 'APPROVED'
               WHERE product_name = %s AND status = 'PENDING_REVIEW'""",
            (product_name,),
        )

        return jsonify({
            "status": "ok",
            "new_category_id": new_id,
            "category_name": category_name,
            "parent_id": parent_id,
            "mount_path": mount_path,
            "category_pids": category_pids,
            "verified": True,
            "message": f"已新增分类 #{new_id} '{category_name}'，挂载于 {mount_path}，验证通过",
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/reject_expansion", methods=["POST"])
def api_reject_expansion():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400

        _db.execute(
            """UPDATE expansion_suggestions SET status = 'REJECTED'
               WHERE product_name = %s AND status = 'PENDING_REVIEW'""",
            (product_name,),
        )

        return jsonify({"status": "ok", "message": f"已拒绝 '{product_name}' 的扩展建议"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pending_expansions", methods=["GET"])
def api_pending_expansions():
    try:
        _init_components()
        rows = _db.execute(
            """SELECT product_name, suggested_parent_id, suggested_category_name,
                      suggested_level_position, llm_analysis, status, created_at
               FROM expansion_suggestions WHERE status = 'PENDING_REVIEW'
               ORDER BY created_at DESC LIMIT 50""",
        )
        results = []
        for r in rows:
            parent_id = r["suggested_parent_id"] or ""
            mount_path = r["suggested_level_position"] or ""
            if not mount_path and parent_id:
                mount_path = format_category_path(_page_tree, parent_id)

            path_nodes = []
            if parent_id:
                path_to_root = _page_tree.get_path_to_root(parent_id)
                for node in path_to_root:
                    path_nodes.append({
                        "level": node.depth + 1,
                        "category_id": node.category_id,
                        "category_name": node.category_name,
                        "is_new": False,
                    })

            suggested_name = r["suggested_category_name"] or ""
            if suggested_name and (not path_nodes or suggested_name != path_nodes[-1]["category_name"]):
                path_nodes.append({
                    "level": len(path_nodes) + 1,
                    "category_id": None,
                    "category_name": suggested_name,
                    "is_new": True,
                })

            sibling_nodes = []
            if parent_id:
                parent_node = _page_tree.get_node(parent_id)
                if parent_node:
                    for child in parent_node.children:
                        if not child.children:
                            sibling_nodes.append({
                                "category_id": child.category_id,
                                "category_name": child.category_name,
                            })

            parent_name = ""
            if parent_id:
                parent_node = _page_tree.get_node(parent_id)
                parent_name = parent_node.category_name if parent_node else ""

            results.append({
                "product_name": r["product_name"],
                "suggested_parent_id": parent_id,
                "suggested_parent_name": parent_name,
                "mount_path": mount_path,
                "path": path_nodes,
                "suggested_category_name": suggested_name,
                "llm_analysis": r["llm_analysis"],
                "sibling_nodes": sibling_nodes[:20],
                "status": r["status"],
                "created_at": str(r["created_at"]),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_taxonomy_overview_for_llm() -> str:
    if _page_tree is None:
        return ""
    try:
        roots = _page_tree.get_root_nodes()
        lines = []
        for r in roots[:30]:
            child_names = [c.category_name for c in r.children[:8]]
            children_str = "、".join(child_names) if child_names else "无子分类"
            lines.append(f"- {r.category_name}(#{r.category_id}): {children_str}")
        return "\n".join(lines)
    except Exception:
        return ""


@app.route("/api/expansion/stash", methods=["POST"])
def api_expansion_stash():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        category_id = data.get("category_id", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400

        if not category_id:
            start = time.perf_counter()
            match_result = _rag_rerank_engine.match(product_name)
            if match_result.candidates:
                category_id = match_result.candidates[0].category_id
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            if not category_id:
                expansion_config = _config.get_expansion_config()
                trgm_candidates = _trigram_fallback_match(
                    product_name,
                    top_k=expansion_config.trgm_top_k,
                    trgm_threshold=expansion_config.trgm_threshold,
                    vector_threshold=expansion_config.vector_threshold,
                )
                if trgm_candidates:
                    category_id = trgm_candidates[0]["category_id"]
                    elapsed += round((time.perf_counter() - start) * 1000, 1) - elapsed
            if not category_id:
                return jsonify({"error": "未找到匹配的分类节点", "product_name": product_name}), 404
        else:
            elapsed = 0

        existing = _db.execute_one(
            "SELECT category_name, syn_list, expansion_syn_list FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        if not existing:
            return jsonify({"error": f"category_id={category_id} 不存在"}), 404

        cat_name = existing["category_name"]
        syn_list = existing.get("syn_list") or []
        exp_syn_list = existing.get("expansion_syn_list") or []

        if product_name in syn_list:
            return jsonify({"status": "already_exists", "message": f"'{product_name}' 已是 #{category_id}({cat_name}) 的同义词"})

        from src.data.synonym_sanitizer import sanitize_syn_list
        cleaned, removed = sanitize_syn_list([product_name], cat_name)
        if removed or not cleaned:
            return jsonify({"status": "rejected", "message": "同义词被清洗规则拒绝"})

        _db.execute(
            "UPDATE category_texts SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, product_name, category_id),
        )
        _db.execute(
            "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, product_name, category_id),
        )

        match_path = ""
        path_row = _db.execute_one("SELECT category_group_name FROM category_texts WHERE category_id = %s", (category_id,))
        if path_row and path_row.get("category_group_name"):
            match_path = path_row["category_group_name"] + " > " + cat_name
        else:
            match_path = format_category_path(_page_tree, category_id)
        _db.execute(
            "INSERT INTO expansion_log (product_name, category_id, category_name, match_path, match_status, source) VALUES (%s, %s, %s, %s, %s, %s)",
            (product_name, category_id, cat_name, match_path, data.get("match_status", "NO_MATCH"), data.get("source", "web")),
        )

        new_exp_count = len(exp_syn_list) + 1
        expansion_config = _config.get_expansion_config()
        threshold_reached = new_exp_count >= expansion_config.syn_threshold

        return jsonify({
            "status": "ok",
            "product_name": product_name,
            "category_id": category_id,
            "category_name": cat_name,
            "match_path": match_path,
            "expansion_syn_count": new_exp_count,
            "threshold": expansion_config.syn_threshold,
            "threshold_reached": threshold_reached,
            "match_elapsed_ms": elapsed,
            "message": f"已将 '{product_name}' 添加为 #{category_id}({cat_name}) 的扩展同义词（{new_exp_count}/{expansion_config.syn_threshold}）",
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/batch_stash", methods=["POST"])
def api_expansion_batch_stash():
    try:
        _init_components()
        data = request.get_json(force=True)
        items = data.get("items", [])
        if not items or not isinstance(items, list):
            return jsonify({"error": "items (list of {product_name, category_id?}) required"}), 400

        from src.data.synonym_sanitizer import sanitize_syn_list

        stashed = []
        skipped = []
        failed = []

        for item in items:
            product_name = item.get("product_name", "").strip()
            category_id = item.get("category_id", "").strip()
            if not product_name:
                continue

            try:
                if not category_id:
                    match_result = _rag_rerank_engine.match(product_name)
                    if match_result.candidates:
                        category_id = match_result.candidates[0].category_id
                    if not category_id:
                        expansion_config = _config.get_expansion_config()
                        trgm_candidates = _trigram_fallback_match(
                            product_name,
                            top_k=expansion_config.trgm_top_k,
                            trgm_threshold=expansion_config.trgm_threshold,
                            vector_threshold=expansion_config.vector_threshold,
                        )
                        if trgm_candidates:
                            category_id = trgm_candidates[0]["category_id"]
                    if not category_id:
                        failed.append({"product_name": product_name, "error": "未找到匹配节点"})
                        continue

                existing = _db.execute_one(
                    "SELECT category_name, syn_list, expansion_syn_list FROM category_texts WHERE category_id = %s",
                    (category_id,),
                )
                if not existing:
                    failed.append({"product_name": product_name, "error": f"category_id={category_id} 不存在"})
                    continue

                cat_name = existing["category_name"]
                syn_list = existing.get("syn_list") or []

                if product_name in syn_list:
                    skipped.append({"product_name": product_name, "reason": f"已是 #{category_id} 的同义词"})
                    continue

                cleaned, removed = sanitize_syn_list([product_name], cat_name)
                if removed or not cleaned:
                    skipped.append({"product_name": product_name, "reason": "被清洗规则拒绝"})
                    continue

                _db.execute(
                    "UPDATE category_texts SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                    (product_name, product_name, category_id),
                )
                _db.execute(
                    "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                    (product_name, product_name, category_id),
                )

                match_path = ""
                path_row = _db.execute_one("SELECT category_group_name FROM category_texts WHERE category_id = %s", (category_id,))
                if path_row and path_row.get("category_group_name"):
                    match_path = path_row["category_group_name"] + " > " + cat_name
                else:
                    match_path = format_category_path(_page_tree, category_id)
                _db.execute(
                    "INSERT INTO expansion_log (product_name, category_id, category_name, match_path, match_status, source) VALUES (%s, %s, %s, %s, %s, %s)",
                    (product_name, category_id, cat_name, match_path, "NO_MATCH", "batch"),
                )

                exp_count = len(existing.get("expansion_syn_list") or []) + 1
                stashed.append({
                    "product_name": product_name,
                    "category_id": category_id,
                    "category_name": cat_name,
                    "match_path": match_path,
                    "expansion_syn_count": exp_count,
                })
            except Exception as e:
                failed.append({"product_name": product_name, "error": str(e)})

        return jsonify({
            "status": "ok",
            "total": len(items),
            "stashed": len(stashed),
            "skipped": len(skipped),
            "failed": len(failed),
            "details": {"stashed": stashed, "skipped": skipped, "failed": failed},
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/suggest", methods=["POST"])
def api_expansion_suggest():
    """获取扩展建议：向量定位 + LLM推理"""
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400

        match_result = _rag_rerank_engine.match(product_name)
        if match_result.match_status.value == "MATCHED":
            return jsonify({
                "status": "already_matched",
                "message": f"该产品已匹配到 #{match_result.matched_category_id}",
                "matched_category_id": match_result.matched_category_id,
                "confidence": round(match_result.confidence, 4),
            })

        vector_result = _vector_semantic_locate(product_name, top_k=10)
        vector_candidates = vector_result.get("candidates", [])
        path_analysis = vector_result.get("path_analysis", {})

        llm_result = None
        llm_path_validation = None
        if vector_candidates and _llm:
            llm_result = _llm_path_reasoning(product_name, vector_candidates)
            
            if llm_result:
                llm_full_path = llm_result.get("full_path", "")
                if llm_full_path:
                    llm_path_validation = _validate_path_nodes(llm_full_path)

        recommendation = None
        if llm_result:
            llm_confidence = llm_result.get("confidence", 0.0)
            llm_parent_id = llm_result.get("suggested_parent_id", "")
            llm_parent_name = llm_result.get("suggested_parent_name", "")
            llm_should_create = llm_result.get("should_create_new_node", False)
            
            if llm_parent_id:
                llm_parent_id = str(llm_parent_id)
                if llm_parent_id.startswith("#"):
                    llm_parent_id = llm_parent_id[1:]
            
            parent_id = ""
            parent_name = ""
            decision_reason = ""
            trust_llm = False
            
            if llm_confidence >= 0.85:
                trust_llm = True
                decision_reason = f"LLM置信度{llm_confidence:.2f}≥0.85，完全采纳LLM建议"
            elif llm_should_create:
                trust_llm = True
                decision_reason = f"LLM建议创建新节点，优先采纳"
            elif llm_confidence >= 0.7:
                trust_llm = True
                decision_reason = f"LLM置信度{llm_confidence:.2f}≥0.7，采纳LLM建议"
            else:
                if vector_candidates:
                    top_vector_candidate = vector_candidates[0]
                    vector_sim = top_vector_candidate.get("similarity", 0)
                    if llm_confidence >= vector_sim:
                        trust_llm = True
                        decision_reason = f"LLM置信度{llm_confidence:.2f}≥向量相似度{vector_sim:.2f}，采纳LLM建议"
                    else:
                        trust_llm = True
                        decision_reason = f"向量相似度{vector_sim:.2f}>LLM置信度{llm_confidence:.2f}，但LLM推理更合理，仍采纳LLM建议"
                else:
                    trust_llm = True
                    decision_reason = f"无向量候选，采纳LLM建议"
            
            if trust_llm and llm_parent_id:
                parent_id = llm_parent_id
                parent_name = llm_parent_name
                
                if parent_id and not parent_id.isdigit():
                    name_row = _db.execute_one(
                        "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_name = %s LIMIT 1",
                        (parent_id,),
                    )
                    if name_row:
                        parent_id = name_row["category_id"]
                        parent_name = name_row["category_name"]
                
                if parent_id and parent_id.isdigit():
                    parent_row = _db.execute_one(
                        "SELECT category_name, category_group_name FROM category_texts WHERE category_id = %s",
                        (parent_id,),
                    )
                    if parent_row:
                        if not parent_name:
                            parent_name = parent_row["category_name"]
                        path = parent_row.get("category_group_name", "")
                        if path:
                            full_path = path.replace(",", " > ") + " > " + parent_name
                        else:
                            full_path = parent_name
                        recommendation = {
                            "parent_id": parent_id,
                            "parent_name": parent_name,
                            "full_path": full_path,
                            "should_create_new_node": llm_should_create,
                            "new_node_name": llm_result.get("new_node_name", ""),
                            "confidence": llm_confidence,
                            "reason": llm_result.get("reasoning", ""),
                            "decision_reason": decision_reason,
                            "trust_llm": True,
                            "llm_weight": "high" if llm_confidence >= 0.85 else "medium",
                        }
                elif parent_name:
                    name_row = _db.execute_one(
                        "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_name = %s LIMIT 1",
                        (parent_name,),
                    )
                    if name_row:
                        path = name_row.get("category_group_name", "")
                        if path:
                            full_path = path.replace(",", " > ") + " > " + name_row["category_name"]
                        else:
                            full_path = name_row["category_name"]
                        recommendation = {
                            "parent_id": name_row["category_id"],
                            "parent_name": name_row["category_name"],
                            "full_path": full_path,
                            "should_create_new_node": llm_should_create,
                            "new_node_name": llm_result.get("new_node_name", ""),
                            "confidence": llm_confidence,
                            "reason": llm_result.get("reasoning", ""),
                            "decision_reason": decision_reason,
                            "trust_llm": True,
                            "llm_weight": "high" if llm_confidence >= 0.85 else "medium",
                        }
            elif not trust_llm and vector_candidates:
                top_candidate = vector_candidates[0]
                parent_id = top_candidate.get("category_id", "")
                parent_name = top_candidate.get("category_name", "")
                path = top_candidate.get("path", "")
                full_path = path.replace(",", " > ") if path else parent_name
                
                parent_row = _db.execute_one(
                    "SELECT category_name, category_group_name FROM category_texts WHERE category_id = %s",
                    (parent_id,),
                )
                if parent_row:
                    path = parent_row.get("category_group_name", "")
                    full_path = (path.replace(",", " > ") + " > " + parent_row["category_name"]) if path else parent_row["category_name"]
                
                recommendation = {
                    "parent_id": parent_id,
                    "parent_name": parent_name,
                    "full_path": full_path,
                    "should_create_new_node": False,
                    "new_node_name": "",
                    "confidence": top_candidate.get("similarity", 0),
                    "reason": "基于向量相似度定位（LLM置信度较低）",
                    "decision_reason": decision_reason,
                    "trust_llm": False,
                    "llm_weight": "low",
                }

        return jsonify({
            "status": "ok",
            "product_name": product_name,
            "method": "hybrid",
            "vector_candidates": vector_candidates[:5],
            "path_analysis": path_analysis,
            "llm_reasoning": llm_result,
            "llm_path_validation": llm_path_validation,
            "recommendation": recommendation,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/execute", methods=["POST"])
def api_expansion_execute():
    """执行扩展：用户确认后执行"""
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        parent_id = data.get("parent_id", "").strip()
        create_new_node = data.get("create_new_node", False)
        new_node_name = data.get("new_node_name", "").strip()
        reason = data.get("reason", "user_confirmed")

        if not product_name or not parent_id:
            return jsonify({"error": "product_name and parent_id required"}), 400

        existing = _db.execute_one(
            "SELECT category_name, syn_list, expansion_syn_list FROM category_texts WHERE category_id = %s",
            (parent_id,),
        )
        if not existing:
            return jsonify({"error": f"parent_id={parent_id} not found"}), 404

        cat_name = existing["category_name"]
        syn_list = existing.get("syn_list") or []

        if product_name in syn_list:
            return jsonify({"status": "already_exists", "message": f"'{product_name}' 已是 #{parent_id} 的同义词"})

        from src.data.synonym_sanitizer import sanitize_syn_list
        cleaned, removed = sanitize_syn_list([product_name], cat_name)
        if removed or not cleaned:
            return jsonify({"status": "rejected", "message": "同义词被清洗规则拒绝"})

        _db.execute(
            "UPDATE category_texts SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, product_name, parent_id),
        )
        _db.execute(
            "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (product_name, product_name, parent_id),
        )

        match_path = ""
        path_row = _db.execute_one("SELECT category_group_name FROM category_texts WHERE category_id = %s", (parent_id,))
        if path_row and path_row.get("category_group_name"):
            match_path = path_row["category_group_name"] + " > " + cat_name
        else:
            match_path = format_category_path(_page_tree, parent_id)

        _db.execute(
            "INSERT INTO expansion_log (product_name, category_id, category_name, match_path, match_status, source) VALUES (%s, %s, %s, %s, %s, %s)",
            (product_name, parent_id, cat_name, match_path, "NO_MATCH", "suggest"),
        )

        exp_count = len(existing.get("expansion_syn_list") or []) + 1
        expansion_config = _config.get_expansion_config()

        return jsonify({
            "status": "ok",
            "product_name": product_name,
            "parent_id": parent_id,
            "parent_name": cat_name,
            "match_path": match_path,
            "expansion_syn_count": exp_count,
            "reason": reason,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/staging/add", methods=["POST"])
def api_staging_add():
    """添加产品到暂存箱"""
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400

        existing = _db.execute_one(
            "SELECT id, status FROM staging_box WHERE product_name = %s",
            (product_name,),
        )
        if existing:
            return jsonify({"status": "already_exists", "message": f"'{product_name}' 已在暂存箱中"})

        _db.execute(
            "INSERT INTO staging_box (product_name, source) VALUES (%s, %s)",
            (product_name, data.get("source", "web")),
        )
        return jsonify({"status": "ok", "message": f"'{product_name}' 已加入暂存箱"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/staging/list", methods=["GET"])
def api_staging_list():
    """获取暂存箱列表"""
    try:
        _init_components()
        rows = _db.execute(
            "SELECT id, product_name, source, status, created_at FROM staging_box ORDER BY created_at DESC"
        )
        return jsonify({"status": "ok", "total": len(rows), "items": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/staging/remove", methods=["POST"])
def api_staging_remove():
    """从暂存箱移除"""
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        if not product_name:
            return jsonify({"error": "product_name required"}), 400
        _db.execute("DELETE FROM staging_box WHERE product_name = %s", (product_name,))
        return jsonify({"status": "ok", "message": f"'{product_name}' 已从暂存箱移除"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/staging/clear", methods=["POST"])
def api_staging_clear():
    """清空暂存箱"""
    try:
        _init_components()
        _db.execute("DELETE FROM staging_box")
        return jsonify({"status": "ok", "message": "暂存箱已清空"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/staging/batch_suggest", methods=["POST"])
def api_staging_batch_suggest():
    """暂存箱批量智能扩展建议。

    暂存项默认已是 RAG+Rerank 未匹配结果，不再重复完整匹配。
    流程：批量 embedding → 缓存矩阵 TopK → LLM 小批并行推理（默认每批3条）。
    """
    try:
        _init_components()
        rows = _db.execute(
            "SELECT product_name FROM staging_box WHERE status = 'pending' ORDER BY created_at"
        )
        if not rows:
            return jsonify({"error": "暂存箱中没有待处理的产品"}), 400

        product_names = [r["product_name"] for r in rows]
        logger = logging.getLogger("app")
        logger.info(f"批量智能扩展开始: {len(product_names)}个产品（跳过RAG+Rerank，单条多路并发LLM）")
        start_time = time.time()

        # 1) 一次加载向量缓存 + 批量 embed
        _ensure_locate_matrix_cache()
        try:
            query_vecs = _embed_queries_batch(product_names)
        except Exception as e:
            logger.warning(f"批量embedding失败，回退逐条: {e}")
            query_vecs = []
            for name in product_names:
                try:
                    query_vecs.append(_vec_mgr.embed_query(name) if _vec_mgr else [])
                except Exception:
                    query_vecs.append([])

        need_llm_products = []
        suggestions_by_name: dict[str, dict] = {}
        for name, qvec in zip(product_names, query_vecs):
            try:
                if _vec_is_empty(qvec):
                    suggestions_by_name[name] = {
                        "product_name": name,
                        "status": "error",
                        "error": "embedding失败",
                    }
                    continue
                located = _top_k_from_query_vec(qvec, top_k=5)
                need_llm_products.append({
                    "product_name": name,
                    "vector_candidates": located.get("candidates", []),
                    "path_analysis": located.get("path_analysis", {}),
                })
            except Exception as e:
                logger.warning(f"向量定位失败 {name}: {e}")
                suggestions_by_name[name] = {
                    "product_name": name,
                    "status": "error",
                    "error": str(e),
                }

        vector_time = time.time()
        logger.info(
            f"向量定位完成: {len(need_llm_products)}条待LLM，耗时 {vector_time - start_time:.2f}秒"
        )

        # 2) LLM 单条多路并发（每条产品独立一轮，降低幻觉；多路并行提速）
        llm_workers = 8
        llm_results_map: dict[str, dict | None] = {
            p["product_name"]: None for p in need_llm_products
        }
        llm_time = vector_time

        if need_llm_products and _llm:
            root_rows = _db.execute(
                "SELECT category_name FROM category_texts WHERE category_pids = '{}' LIMIT 30"
            )
            taxonomy_overview = (
                ", ".join([r["category_name"] for r in root_rows]) if root_rows else ""
            )

            def _run_one_llm(product_info: dict) -> tuple[str, dict | None]:
                pname = product_info["product_name"]
                try:
                    results = _llm.batch_path_reasoning(
                        [{
                            "product_name": pname,
                            "vector_candidates": product_info["vector_candidates"],
                        }],
                        taxonomy_overview,
                    )
                    return pname, (results[0] if results else None)
                except Exception as e:
                    logger.warning(f"单条LLM失败 {pname}: {e}")
                    return pname, None

            with ThreadPoolExecutor(
                max_workers=min(llm_workers, max(len(need_llm_products), 1))
            ) as executor:
                futures = [
                    executor.submit(_run_one_llm, p) for p in need_llm_products
                ]
                for fut in as_completed(futures):
                    pname, res = fut.result()
                    llm_results_map[pname] = res

            llm_time = time.time()
            logger.info(
                f"LLM单条并发完成: {len(need_llm_products)}条 × workers≤{llm_workers}，"
                f"耗时 {llm_time - vector_time:.2f}秒"
            )

        for product_info in need_llm_products:
            pname = product_info["product_name"]
            suggestions_by_name[pname] = _build_expansion_suggestion(
                pname,
                product_info["vector_candidates"],
                product_info["path_analysis"],
                llm_results_map.get(pname),
            )

        suggestions = [suggestions_by_name[n] for n in product_names if n in suggestions_by_name]

        total_time = time.time() - start_time
        logger.info(
            f"批量智能扩展完成，总耗时: {total_time:.2f}秒，平均每个产品: {total_time / max(len(product_names), 1):.2f}秒"
        )

        return jsonify({
            "status": "ok",
            "total_products": len(product_names),
            "suggestions": suggestions,
            "performance": {
                "total_time": round(total_time, 2),
                "avg_time_per_product": round(total_time / max(len(product_names), 1), 2),
                "vector_time": round(vector_time - start_time, 2),
                "llm_time": round(llm_time - vector_time, 2) if need_llm_products and _llm else 0,
                "llm_workers": llm_workers,
                "llm_mode": "one_product_per_call",
                "skipped_rag_rerank": True,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500



@app.route("/api/staging/batch_execute", methods=["POST"])
def api_staging_batch_execute():
    """批量执行扩展：将暂存箱产品按建议扩展到树中"""
    try:
        _init_components()
        data = request.get_json(force=True)
        items = data.get("items", [])
        if not items:
            return jsonify({"error": "没有有效的扩展项（可能所有产品已在对应节点中，或LLM建议的父节点不存在）"}), 400

        from src.data.synonym_sanitizer import GENERIC_SHORT_SYNONYMS

        stashed = []
        skipped = []
        failed = []

        for item in items:
            product_name = item.get("product_name", "").strip()
            parent_id = item.get("parent_id", "").strip()
            if not product_name or not parent_id:
                continue

            existing = _db.execute_one(
                "SELECT category_name, syn_list, expansion_syn_list FROM category_texts WHERE category_id = %s",
                (parent_id,),
            )
            if not existing:
                failed.append({"product_name": product_name, "error": f"#{parent_id}不存在"})
                continue

            cat_name = existing["category_name"]
            syn_list = existing.get("syn_list") or []

            if product_name in syn_list:
                skipped.append({"product_name": product_name, "reason": f"已是 #{parent_id} 的同义词"})
                continue

            # 暂存扩展是用户确认的产品名：只拦截泛词短同义词，允许「弯头」「黄饼」等具体短名
            if product_name in GENERIC_SHORT_SYNONYMS:
                skipped.append({
                    "product_name": product_name,
                    "reason": f"泛词短同义词被拒绝（相对分类「{cat_name}」）",
                })
                continue

            _db.execute(
                "UPDATE category_texts SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                (product_name, product_name, parent_id),
            )
            _db.execute(
                "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                (product_name, product_name, parent_id),
            )

            match_path = ""
            path_row = _db.execute_one("SELECT category_group_name FROM category_texts WHERE category_id = %s", (parent_id,))
            if path_row and path_row.get("category_group_name"):
                match_path = path_row["category_group_name"] + " > " + cat_name
            else:
                match_path = format_category_path(_page_tree, parent_id)
            _db.execute(
                "INSERT INTO expansion_log (product_name, category_id, category_name, match_path, match_status, source) VALUES (%s, %s, %s, %s, %s, %s)",
                (product_name, parent_id, cat_name, match_path, "NO_MATCH", "staging_batch"),
            )

            _db.execute("DELETE FROM staging_box WHERE product_name = %s", (product_name,))

            stashed.append({
                "product_name": product_name,
                "parent_id": parent_id,
                "parent_name": cat_name,
            })

        if stashed:
            _invalidate_locate_matrix_cache()

        return jsonify({
            "status": "ok",
            "stashed": len(stashed),
            "skipped": len(skipped),
            "failed": len(failed),
            "details": {"stashed": stashed, "skipped": skipped, "failed": failed},
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/staging/batch_smart", methods=["POST"])
def api_staging_batch_smart():
    """批量智能暂存：先对批次内产品聚类，同簇产品挂到同一父节点"""
    try:
        _init_components()
        data = request.get_json(force=True)
        items = data.get("items", [])
        if not items or not isinstance(items, list):
            return jsonify({"error": "items (list of {product_name}) required"}), 400

        from src.data.synonym_sanitizer import sanitize_syn_list

        product_names = [item.get("product_name", "").strip() for item in items if item.get("product_name", "").strip()]
        if not product_names:
            return jsonify({"error": "无有效产品名"}), 400

        product_entries = []
        for p in product_names:
            match_result = _rag_rerank_engine.match(p)
            suggested_parent_id = ""
            suggested_parent_name = ""
            if match_result.candidates:
                suggested_parent_id = match_result.candidates[0].category_id
                suggested_parent_name = match_result.candidates[0].category_name
            product_entries.append({
                "id": p,
                "product_name": p,
                "suggested_parent_id": suggested_parent_id,
                "suggested_parent_name": suggested_parent_name,
            })

        if not _llm:
            return jsonify({"error": "LLM未初始化，无法执行智能聚类"}), 500

        taxonomy_overview = ""
        root_rows = _db.execute("SELECT category_name FROM category_texts WHERE category_pids = '{}' LIMIT 20")
        if root_rows:
            taxonomy_overview = ", ".join([r["category_name"] for r in root_rows])

        clusters, outliers = _llm.cluster_products(product_entries, taxonomy_overview)

        if not clusters:
            return jsonify({"error": "LLM聚类失败或无有效聚类结果"}), 500

        stashed = []
        skipped = []
        failed = []

        for cluster in clusters:
            parent_id = cluster.get("suggested_parent_id", "")
            parent_name = cluster.get("suggested_parent_name", "")
            product_names_in_cluster = cluster.get("product_names", [])

            if not parent_id or not product_names_in_cluster:
                continue

            existing = _db.execute_one(
                "SELECT category_name, syn_list, expansion_syn_list FROM category_texts WHERE category_id = %s",
                (parent_id,),
            )
            if not existing:
                for p in product_names_in_cluster:
                    failed.append({"product_name": p, "error": f"父节点#{parent_id}不存在"})
                continue

            cat_name = existing["category_name"]
            syn_list = existing.get("syn_list") or []

            for p in product_names_in_cluster:
                if p in syn_list:
                    skipped.append({"product_name": p, "reason": f"已是 #{parent_id} 的同义词"})
                    continue

                cleaned, removed = sanitize_syn_list([p], cat_name)
                if removed or not cleaned:
                    skipped.append({"product_name": p, "reason": "被清洗规则拒绝"})
                    continue

                _db.execute(
                    "UPDATE category_texts SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                    (p, p, parent_id),
                )
                _db.execute(
                    "UPDATE category_vectors SET syn_list = array_append(syn_list, %s), expansion_syn_list = array_append(expansion_syn_list, %s), updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
                    (p, p, parent_id),
                )

                match_path = ""
                path_row = _db.execute_one("SELECT category_group_name FROM category_texts WHERE category_id = %s", (parent_id,))
                if path_row and path_row.get("category_group_name"):
                    match_path = path_row["category_group_name"] + " > " + cat_name
                else:
                    match_path = format_category_path(_page_tree, parent_id)
                _db.execute(
                    "INSERT INTO expansion_log (product_name, category_id, category_name, match_path, match_status, source) VALUES (%s, %s, %s, %s, %s, %s)",
                    (p, parent_id, cat_name, match_path, "NO_MATCH", "batch_smart"),
                )

                exp_count = len(existing.get("expansion_syn_list") or []) + 1
                stashed.append({
                    "product_name": p,
                    "category_id": parent_id,
                    "category_name": cat_name,
                    "match_path": match_path,
                    "expansion_syn_count": exp_count,
                    "cluster_group": cluster.get("suggested_category_name", ""),
                })

        return jsonify({
            "status": "ok",
            "total": len(items),
            "stashed": len(stashed),
            "skipped": len(skipped),
            "failed": len(failed),
            "clusters": len(clusters),
            "details": {"stashed": stashed, "skipped": skipped, "failed": failed},
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/grouped", methods=["GET"])
def api_expansion_grouped():
    try:
        _init_components()
        expansion_config = _config.get_expansion_config()
        threshold = expansion_config.syn_threshold

        rows = _db.execute(
            """SELECT category_id, category_name, category_group_name, expansion_syn_list
               FROM category_texts
               WHERE expansion_syn_list IS NOT NULL AND array_length(expansion_syn_list, 1) > 0
               ORDER BY array_length(expansion_syn_list, 1) DESC"""
        )

        groups = []
        total_syns = 0
        for r in rows:
            syns = r.get("expansion_syn_list") or []
            count = len(syns)
            total_syns += count
            path_parts = []
            group_name = r.get("category_group_name") or ""
            if group_name:
                path_parts = [p.strip() for p in group_name.split(",") if p.strip()]
            path_parts.append(r["category_name"])

            groups.append({
                "category_id": r["category_id"],
                "category_name": r["category_name"],
                "path": path_parts,
                "products": syns,
                "count": count,
                "threshold_reached": count >= threshold,
            })

        return jsonify({
            "status": "ok",
            "threshold": threshold,
            "total_groups": len(groups),
            "total_products": total_syns,
            "groups": groups,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/remove", methods=["POST"])
def api_expansion_remove():
    try:
        _init_components()
        data = request.get_json(force=True)
        product_name = data.get("product_name", "").strip()
        category_id = data.get("category_id", "").strip()
        if not product_name or not category_id:
            return jsonify({"error": "product_name and category_id required"}), 400

        existing = _db.execute_one(
            "SELECT expansion_syn_list FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        if not existing:
            return jsonify({"error": f"category_id={category_id} not found"}), 404

        syns = existing.get("expansion_syn_list") or []
        if product_name not in syns:
            return jsonify({"error": f"'{product_name}' not in expansion_syn_list of #{category_id}"}), 404

        new_syns = [s for s in syns if s != product_name]
        _db.execute(
            "UPDATE category_texts SET expansion_syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (new_syns, category_id),
        )
        _db.execute(
            "UPDATE category_vectors SET expansion_syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (new_syns, category_id),
        )
        _db.execute(
            "DELETE FROM expansion_log WHERE product_name = %s AND category_id = %s",
            (product_name, category_id),
        )

        return jsonify({
            "status": "ok",
            "message": f"已从 #{category_id} 移除 '{product_name}'",
            "remaining_count": len(new_syns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/log", methods=["GET"])
def api_expansion_log():
    try:
        _init_components()
        limit = request.args.get("limit", 50, type=int)
        rows = _db.execute(
            """SELECT id, product_name, category_id, category_name, match_path, match_status, source, created_at
               FROM expansion_log ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        return jsonify({"status": "ok", "total": len(rows), "entries": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/node_stats", methods=["GET"])
def api_expansion_node_stats():
    try:
        _init_components()
        expansion_config = _config.get_expansion_config()
        threshold = expansion_config.syn_threshold

        rows = _db.execute(
            """SELECT category_id, category_name, category_group_name, expansion_syn_list
               FROM category_texts
               WHERE expansion_syn_list IS NOT NULL AND array_length(expansion_syn_list, 1) > 0
               ORDER BY array_length(expansion_syn_list, 1) DESC"""
        )

        nodes = []
        for r in rows:
            syns = r.get("expansion_syn_list") or []
            count = len(syns)
            match_path = ""
            if r.get("category_group_name"):
                match_path = r["category_group_name"] + " > " + r["category_name"]
            else:
                match_path = format_category_path(_page_tree, r["category_id"])
            nodes.append({
                "category_id": r["category_id"],
                "category_name": r["category_name"],
                "match_path": match_path,
                "expansion_syn_count": count,
                "expansion_syns": syns,
                "threshold_reached": count >= threshold,
            })

        return jsonify({
            "status": "ok",
            "threshold": threshold,
            "total_nodes": len(nodes),
            "ready_nodes": len([n for n in nodes if n["threshold_reached"]]),
            "nodes": nodes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/cluster_node", methods=["POST"])
def api_expansion_cluster_node():
    try:
        _init_components()
        data = request.get_json(force=True)
        category_id = data.get("category_id", "").strip()
        if not category_id:
            return jsonify({"error": "category_id required"}), 400

        parent_row = _db.execute_one(
            "SELECT category_id, category_name, expansion_syn_list FROM category_texts WHERE category_id = %s",
            (category_id,),
        )
        if not parent_row:
            return jsonify({"error": f"category_id={category_id} 不存在"}), 404

        expansion_syns = parent_row.get("expansion_syn_list") or []
        if len(expansion_syns) < 2:
            return jsonify({"error": f"扩展同义词不足2条(当前{len(expansion_syns)}条)，无法聚类"}), 400

        taxonomy_overview = _build_taxonomy_overview_for_llm()
        parent_name = parent_row["category_name"]

        entries_for_cluster = []
        for syn in expansion_syns:
            entries_for_cluster.append({
                "id": syn,
                "product_name": syn,
                "suggested_parent_id": category_id,
                "suggested_parent_name": parent_name,
                "suggested_category_name": "",
                "path_text": "",
                "confidence": 0.0,
            })

        clusters_out = []
        outliers_out = []

        if _llm is not None and len(expansion_syns) >= 2:
            try:
                lc, lo = _llm.cluster_products(entries_for_cluster, taxonomy_overview)
                clusters_out = lc
                outliers_out = lo
            except Exception as e:
                logging.getLogger("app").warning(f"LLM聚类失败，降级为名称分组: {e}")

        if not clusters_out:
            from collections import defaultdict
            name_groups: dict[str, list[int]] = defaultdict(list)
            for i, syn in enumerate(expansion_syns):
                name_groups[syn].append(i)
            for name, indices in name_groups.items():
                clusters_out.append({
                    "suggested_parent_id": category_id,
                    "suggested_parent_name": parent_name,
                    "suggested_category_name": name,
                    "merged_category_name": name,
                    "full_path": f"{parent_name} > {name}",
                    "entry_count": len(indices),
                    "entries": [expansion_syns[i] for i in indices],
                    "product_names": [expansion_syns[i] for i in indices],
                    "llm_reason": "",
                    "is_llm_clustered": False,
                })
            outliers_out = []

        created_nodes = []
        remaining_syns = set(expansion_syns)

        for cluster in clusters_out:
            if cluster.get("entry_count", 0) < 2:
                continue

            product_names = cluster.get("product_names", [])
            if not product_names:
                product_names = cluster.get("entries", [])
            if not product_names:
                continue

            cluster_name = cluster.get("merged_category_name") or cluster.get("suggested_category_name") or product_names[0]
            name_override = data.get("category_name_overrides", {}).get(cluster.get("suggested_category_name", ""), "")
            if name_override:
                cluster_name = name_override

            new_id = allocate_next_category_id(_db)
            category_pids, category_group_name = build_category_path_fields(_page_tree, category_id)
            mount_path = format_category_path(_page_tree, category_id)

            _db.execute(
                """INSERT INTO category_texts (category_id, category_name, category_pids, syn_list, expansion_syn_list, category_group_name)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (new_id, cluster_name, category_pids, product_names, [], category_group_name),
            )

            try:
                from src.index.api_embedder import ApiEmbedder
                embedding_config = _config.get_embedding_config()
                embedder = ApiEmbedder(
                    api_key=embedding_config.api_key,
                    base_url=embedding_config.base_url,
                    model=embedding_config.model,
                    embedding_dim=embedding_config.dimension,
                )
                embed_text = cluster_name + " " + " ".join(product_names)
                embedding = embedder.embed(embed_text)

                import numpy as np
                import pickle
                embedding_bytes = pickle.dumps(embedding)
                emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

                _db.execute(
                    """INSERT INTO category_vectors (category_id, category_name, embedding, syn_list, expansion_syn_list)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (new_id, cluster_name, embedding_bytes, product_names, []),
                )

                vec_str = "[" + ",".join(str(float(v)) for v in emb_list) + "]"
                try:
                    _db.execute(
                        "UPDATE category_vectors SET vec_search = %s::vector WHERE category_id = %s",
                        (vec_str, new_id),
                    )
                except Exception as vec_err:
                    logging.getLogger("app").warning(f"vec_search写入失败(非致命): {vec_err}")
            except Exception as embed_ex:
                logging.getLogger("app").warning(f"向量写入失败(非致命): {embed_ex}")

            _page_tree.add_node(new_id, cluster_name, category_id, product_names)
            try:
                _vec_mgr.invalidate_matrix()
            except Exception:
                pass

            for syn in product_names:
                remaining_syns.discard(syn)

            created_nodes.append({
                "new_category_id": new_id,
                "category_name": cluster_name,
                "parent_id": category_id,
                "mount_path": mount_path,
                "synonyms": product_names,
            })

        remaining_list = list(remaining_syns)
        _db.execute(
            "UPDATE category_texts SET expansion_syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (remaining_list, category_id),
        )
        _db.execute(
            "UPDATE category_vectors SET expansion_syn_list = %s, updated_at = CURRENT_TIMESTAMP WHERE category_id = %s",
            (remaining_list, category_id),
        )

        return jsonify({
            "status": "ok",
            "parent_id": category_id,
            "parent_name": parent_name,
            "total_expansion_syns": len(expansion_syns),
            "clusters_created": len(created_nodes),
            "remaining_syns": len(remaining_list),
            "created_nodes": created_nodes,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/pool", methods=["GET"])
def api_expansion_pool():
    try:
        _init_components()
        from src.data.expansion_pool import load_pool
        pool = load_pool()

        parent_id_filter = request.args.get("parent_id", "").strip()
        limit = request.args.get("limit", 100, type=int)

        entries = pool.get("entries", [])
        if parent_id_filter:
            entries = [e for e in entries if e.get("suggested_parent_id") == parent_id_filter]
        entries = entries[:limit]

        return jsonify({
            "total": len(pool.get("entries", [])),
            "last_cluster_time": pool.get("last_cluster_time"),
            "entries": entries,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/pool_update", methods=["POST"])
def api_expansion_pool_update():
    """修改暂存池条目的分类名/挂载父节点等。"""
    try:
        _init_components()
        data = request.get_json(force=True)
        entry_id = data.get("entry_id", "").strip()
        if not entry_id:
            return jsonify({"error": "entry_id required"}), 400
        from src.data.expansion_pool import update_entry
        result = update_entry(
            entry_id,
            suggested_category_name=data.get("category_name"),
            suggested_parent_id=data.get("parent_id"),
            suggested_parent_name=data.get("parent_name"),
            path_text=data.get("path_text"),
            llm_reason=data.get("llm_reason"),
        )
        if result.get("status") == "not_found":
            return jsonify({"error": f"entry_id={entry_id} not found"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/staging/prepare_new_nodes", methods=["POST"])
def api_staging_prepare_new_nodes():
    """从暂存箱生成新节点扩展建议（PageIndex 定父节点 + 正式类名）。

    1) PageIndex 自上而下选路，某层无合适子类则停在该层作父节点
    2) 在父节点下建议正式标准分类名（禁止产品级细叶子）
    3) 父节点必须落库存在
    """
    try:
        _init_components()
        from src.data.expansion_pool import add_entry, load_pool, clear_pool

        # 前端按钮可能不带 body；空 body + application/json 会导致 get_json 抛 400
        data = request.get_json(silent=True) or {}
        # 可选：只跑指定产品（便于演示），否则跑整个暂存箱
        only_names = data.get("product_names") or []
        clear_first = bool(data.get("clear_pool", False))
        if clear_first:
            try:
                clear_pool()
            except Exception:
                pass

        if only_names:
            product_names = [str(n).strip() for n in only_names if str(n).strip()]
        else:
            rows = _db.execute(
                "SELECT product_name FROM staging_box WHERE status = 'pending' ORDER BY created_at"
            )
            if not rows:
                return jsonify({"error": "暂存箱中没有待处理的产品"}), 400
            product_names = [r["product_name"] for r in rows]

        if not _llm:
            return jsonify({"error": "LLM未初始化"}), 500
        if not _ensure_page_tree_ready():
            return jsonify({"error": "PageIndex树不可用（Excel/DB均未能建树）"}), 500

        logger = logging.getLogger("app")
        logger.info(f"新节点建议(PageIndex): {len(product_names)} 条")

        def _prepare_one(product_name: str) -> dict:
            try:
                located = _pageindex_locate_expansion_parent(product_name)
                if not located.get("ok"):
                    return {
                        "product_name": product_name,
                        "status": "error",
                        "error": located.get("error") or "PageIndex定位失败",
                    }

                parent_id = str(located["parent_id"])
                parent_name = located["parent_name"]
                parent_path = located["path_text"]
                siblings = located.get("sibling_names") or []
                steps = located.get("steps") or []

                formal = _llm.suggest_formal_category_name(
                    product_name, parent_name, parent_path, siblings
                )
                raw_name = (formal.get("new_node_name") or "").strip()
                new_node_name = _formalize_new_category_name(
                    product_name, raw_name, parent_name
                )

                # 若正式类已存在：挂到该类的父级，提示并入
                reason_prefix = ""
                existing_formal = _db.execute_one(
                    "SELECT category_id, category_name, category_group_name FROM category_texts WHERE category_name = %s LIMIT 1",
                    (new_node_name,),
                )
                if existing_formal:
                    group = existing_formal.get("category_group_name") or ""
                    segs = [p.strip() for p in group.split(",") if p.strip()]
                    if segs:
                        up = _db.execute_one(
                            "SELECT category_id, category_name FROM category_texts WHERE category_name = %s LIMIT 1",
                            (segs[-1],),
                        )
                        if up:
                            parent_id = str(up["category_id"])
                            parent_name = up["category_name"]
                            parent_path_row = _db.execute_one(
                                "SELECT category_group_name, category_name FROM category_texts WHERE category_id = %s",
                                (parent_id,),
                            )
                            if parent_path_row:
                                g = (parent_path_row.get("category_group_name") or "").replace(",", " > ")
                                parent_path = (g + " > " if g else "") + (
                                    parent_path_row.get("category_name") or parent_name
                                )
                    reason_prefix = (
                        f"正式类「{new_node_name}」已存在(#{existing_formal['category_id']})，建议并入；"
                    )

                path_text = f"{parent_path} > {new_node_name}" if parent_path else new_node_name
                conf = float(formal.get("confidence") or located.get("confidence") or 0.6)
                if new_node_name == product_name:
                    conf = min(conf, 0.5)

                step_summary = " → ".join(
                    f"{s.get('action')}:{s.get('choice')}" for s in steps if s.get("choice")
                )
                reason = reason_prefix + (formal.get("reason") or "")
                stop_reason = next(
                    (s.get("reason") for s in reversed(steps) if s.get("action") == "stop"),
                    "",
                )
                if stop_reason:
                    reason = (reason + "；" if reason else "") + f"PageIndex停层: {stop_reason}"
                reason = (reason + "；" if reason else "") + f"路径决策: {step_summary}"
                if raw_name and raw_name != new_node_name:
                    reason += f"｜类名规范化: {raw_name}→{new_node_name}"

                path_nodes = []
                for i, node in enumerate(located.get("path_nodes") or []):
                    path_nodes.append({
                        "level": i + 1,
                        "category_id": str(getattr(node, "category_id", "") or ""),
                        "category_name": getattr(node, "category_name", ""),
                        "is_new": False,
                    })
                # 若父节点因已存在正式类被调整，保证末级真实父在 path 中
                if not path_nodes or str(path_nodes[-1].get("category_id")) != parent_id:
                    path_nodes.append({
                        "level": len(path_nodes) + 1,
                        "category_id": parent_id,
                        "category_name": parent_name,
                        "is_new": False,
                    })
                path_nodes.append({
                    "level": len(path_nodes) + 1,
                    "category_id": None,
                    "category_name": new_node_name,
                    "is_new": True,
                })

                result = add_entry(
                    product_name=product_name,
                    suggested_parent_id=parent_id,
                    suggested_parent_name=parent_name,
                    suggested_category_name=new_node_name,
                    path=path_nodes,
                    confidence=conf,
                    llm_reason=reason,
                    sibling_nodes=[{"category_name": s} for s in siblings[:10]],
                    source="staging_pageindex",
                    path_text=path_text,
                )
                return {
                    "product_name": product_name,
                    "status": result.get("status"),
                    "entry_id": result.get("entry_id"),
                    "path_text": path_text,
                    "category_name": new_node_name,
                    "parent_id": parent_id,
                    "parent_name": parent_name,
                    "confidence": conf,
                    "method": "pageindex",
                    "steps": steps,
                }
            except Exception as ex:
                logger.warning(f"新节点建议失败 {product_name}: {ex}\n{traceback.format_exc()}")
                return {"product_name": product_name, "status": "error", "error": str(ex)}

        prepared = []
        # PageIndex 逐层 LLM，并发不宜过高
        with ThreadPoolExecutor(max_workers=min(3, max(len(product_names), 1))) as executor:
            futures = {executor.submit(_prepare_one, n): n for n in product_names}
            for fut in as_completed(futures):
                prepared.append(fut.result())

        prepared.sort(
            key=lambda x: product_names.index(x["product_name"])
            if x.get("product_name") in product_names else 0
        )
        pool = load_pool()
        return jsonify({
            "status": "ok",
            "method": "pageindex",
            "total": len(product_names),
            "prepared": prepared,
            "pool_total": len(pool.get("entries", [])),
            "ok_count": sum(1 for p in prepared if p.get("status") in ("ok", "already_exists")),
            "error_count": sum(1 for p in prepared if p.get("status") == "error"),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/pool_stats", methods=["GET"])
def api_expansion_pool_stats():
    try:
        _init_components()
        from src.data.expansion_pool import get_pool_stats
        return jsonify(get_pool_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/pool_remove", methods=["POST"])
def api_expansion_pool_remove():
    try:
        _init_components()
        data = request.get_json(force=True)
        entry_id = data.get("entry_id", "").strip()
        if not entry_id:
            return jsonify({"error": "entry_id required"}), 400

        from src.data.expansion_pool import remove_entry
        result = remove_entry(entry_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/cluster", methods=["POST"])
def api_expansion_cluster():
    try:
        _init_components()
        from src.engine.cluster_engine import run_cluster

        similarity_threshold = request.get_json(force=True).get("threshold", 0.65) if request.is_json else 0.65

        embed_func = None
        try:
            embedding_config = _config.get_embedding_config()
            from src.index.api_embedder import ApiEmbedder
            embedder = ApiEmbedder(
                api_key=embedding_config.api_key,
                base_url=embedding_config.base_url,
                model=embedding_config.model,
                embedding_dim=embedding_config.dimension,
            )
            embed_func = embedder.embed_batch
        except Exception as ex:
            logging.getLogger("WebAPI").warning(f"Embedder初始化失败: {ex}")

        result = run_cluster(
            llm=_llm,
            page_tree=_page_tree,
            embed_func=embed_func,
            similarity_threshold=similarity_threshold,
        )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/cluster_report", methods=["GET"])
def api_expansion_cluster_report():
    try:
        _init_components()
        from src.data.expansion_pool import load_report
        report = load_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/approve_cluster", methods=["POST"])
def api_expansion_approve_cluster():
    try:
        _init_components()
        data = request.get_json(force=True)
        cluster_id = data.get("cluster_id", "").strip()
        category_name_override = data.get("category_name", "").strip()

        if not cluster_id:
            return jsonify({"error": "cluster_id required"}), 400

        from src.data.expansion_pool import load_report, save_report, load_pool, save_pool, remove_entries
        report = load_report()

        cluster = None
        for c in report.get("clusters", []):
            if c["cluster_id"] == cluster_id:
                cluster = c
                break

        if cluster is None:
            return jsonify({"error": f"cluster_id={cluster_id} not found"}), 404
        if cluster.get("status") != "PENDING_REVIEW":
            return jsonify({"error": f"簇状态为{cluster.get('status')}，无法批准"}), 400

        parent_id = data.get("parent_id", "").strip() or cluster.get("suggested_parent_id", "")
        if not parent_id:
            full_path = cluster.get("full_path", "")
            if full_path:
                path_parts = [p.strip() for p in full_path.split(">") if p.strip()]
                for part in reversed(path_parts[:-1]):
                    matched = _db.execute(
                        "SELECT category_id FROM category_texts WHERE category_name = %s LIMIT 1",
                        (part,),
                    )
                    if matched:
                        parent_id = matched[0]["category_id"]
                        break

        if not parent_id:
            return jsonify({"error": "未指定挂载父节点(parent_id)，请在审核时选择挂载位置"}), 400

        parent_node = _page_tree.get_node(parent_id)
        if not parent_node:
            return jsonify({"error": f"父节点 {parent_id} 不存在于树中"}), 404

        new_id = allocate_next_category_id(_db)
        category_name = category_name_override or cluster.get("merged_category_name", "") or cluster.get("suggested_category_name", "")
        product_names = cluster.get("product_names", [])

        category_pids, category_group_name = build_category_path_fields(_page_tree, parent_id)
        mount_path = format_category_path(_page_tree, parent_id)

        _db.execute(
            """INSERT INTO category_texts (category_id, category_name, category_pids, syn_list, category_group_name)
               VALUES (%s, %s, %s, %s, %s)""",
            (new_id, category_name, category_pids, product_names, category_group_name),
        )

        try:
            from src.index.api_embedder import ApiEmbedder
            embedding_config = _config.get_embedding_config()
            embedder = ApiEmbedder(
                api_key=embedding_config.api_key,
                base_url=embedding_config.base_url,
                model=embedding_config.model,
                embedding_dim=embedding_config.dimension,
            )
            embed_text = category_name + " " + " ".join(product_names)
            embedding = embedder.embed(embed_text)

            import numpy as np
            import pickle
            embedding_bytes = pickle.dumps(embedding)
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

            _db.execute(
                """INSERT INTO category_vectors (category_id, category_name, embedding, syn_list)
                   VALUES (%s, %s, %s, %s)""",
                (new_id, category_name, embedding_bytes, product_names),
            )

            vec_str = "[" + ",".join(str(float(v)) for v in emb_list) + "]"
            try:
                _db.execute(
                    "UPDATE category_vectors SET vec_bgem3 = %s::vector WHERE category_id = %s",
                    (vec_str, new_id),
                )
            except Exception as vec_err:
                logging.getLogger("app").warning(f"vec_bgem3写入失败(非致命): {vec_err}")
        except Exception as embed_ex:
            logging.getLogger("app").warning(f"向量写入失败(非致命): {embed_ex}")

        _page_tree.add_node(new_id, category_name, parent_id, product_names)

        try:
            _vec_mgr.invalidate_matrix()
        except Exception:
            pass

        cluster["status"] = "APPROVED"
        cluster["category_id_created"] = new_id
        if category_name_override:
            cluster["merged_category_name"] = category_name_override
        save_report(report)

        remove_entries(cluster.get("entries", []))

        return jsonify({
            "status": "ok",
            "new_category_id": new_id,
            "category_name": category_name,
            "parent_id": parent_id,
            "mount_path": mount_path,
            "synonyms_added": product_names,
            "message": f"已新增分类 #{new_id}({category_name})，挂载于 {mount_path}，同义词: {', '.join(product_names)}",
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/expansion/reject_cluster", methods=["POST"])
def api_expansion_reject_cluster():
    try:
        _init_components()
        data = request.get_json(force=True)
        cluster_id = data.get("cluster_id", "").strip()
        return_to_pool = data.get("return_to_pool", False)

        if not cluster_id:
            return jsonify({"error": "cluster_id required"}), 400

        from src.data.expansion_pool import load_report, save_report, remove_entries
        report = load_report()

        cluster = None
        for c in report.get("clusters", []):
            if c["cluster_id"] == cluster_id:
                cluster = c
                break

        if cluster is None:
            return jsonify({"error": f"cluster_id={cluster_id} not found"}), 404

        cluster["status"] = "REJECTED"
        save_report(report)

        if not return_to_pool:
            remove_entries(cluster.get("entries", []))

        return jsonify({
            "status": "ok",
            "message": f"已拒绝簇 {cluster_id}",
            "entries_returned": return_to_pool,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expansion/approve_single", methods=["POST"])
def api_expansion_approve_single():
    try:
        _init_components()
        data = request.get_json(force=True)
        entry_id = data.get("entry_id", "").strip()
        category_name_override = data.get("category_name", "").strip()

        if not entry_id:
            return jsonify({"error": "entry_id required"}), 400

        from src.data.expansion_pool import load_pool, save_pool, remove_entry
        pool = load_pool()

        entry = None
        for e in pool.get("entries", []):
            if e["id"] == entry_id:
                entry = e
                break

        if entry is None:
            return jsonify({"error": f"entry_id={entry_id} not found"}), 404

        parent_id = data.get("parent_id", "").strip() or entry.get("suggested_parent_id", "")
        if not parent_id:
            path_text = entry.get("path_text", "")
            if path_text:
                path_parts = [p.strip() for p in path_text.split(">") if p.strip()]
                for part in reversed(path_parts[:-1]):
                    matched = _db.execute(
                        "SELECT category_id FROM category_texts WHERE category_name = %s LIMIT 1",
                        (part,),
                    )
                    if matched:
                        parent_id = matched[0]["category_id"]
                        break

        if not parent_id:
            return jsonify({"error": "未指定挂载父节点(parent_id)，请在审核时选择挂载位置"}), 400

        parent_node = _page_tree.get_node(parent_id)
        if not parent_node:
            return jsonify({"error": f"父节点 {parent_id} 不存在于树中"}), 404

        new_id = allocate_next_category_id(_db)
        category_name = category_name_override or entry.get("suggested_category_name", "") or entry["product_name"]
        product_name = entry["product_name"]

        category_pids, category_group_name = build_category_path_fields(_page_tree, parent_id)
        mount_path = format_category_path(_page_tree, parent_id)

        _db.execute(
            """INSERT INTO category_texts (category_id, category_name, category_pids, syn_list, category_group_name)
               VALUES (%s, %s, %s, %s, %s)""",
            (new_id, category_name, category_pids, [product_name], category_group_name),
        )

        try:
            from src.index.api_embedder import ApiEmbedder
            embedding_config = _config.get_embedding_config()
            embedder = ApiEmbedder(
                api_key=embedding_config.api_key,
                base_url=embedding_config.base_url,
                model=embedding_config.model,
                embedding_dim=embedding_config.dimension,
            )
            embedding = embedder.embed(category_name + " " + product_name)

            import numpy as np
            import pickle
            embedding_bytes = pickle.dumps(embedding)
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

            _db.execute(
                """INSERT INTO category_vectors (category_id, category_name, embedding, syn_list)
                   VALUES (%s, %s, %s, %s)""",
                (new_id, category_name, embedding_bytes, [product_name]),
            )

            vec_str = "[" + ",".join(str(float(v)) for v in emb_list) + "]"
            try:
                _db.execute(
                    "UPDATE category_vectors SET vec_bgem3 = %s::vector WHERE category_id = %s",
                    (vec_str, new_id),
                )
            except Exception as vec_err:
                logging.getLogger("app").warning(f"vec_bgem3写入失败(非致命): {vec_err}")
        except Exception as embed_ex:
            logging.getLogger("app").warning(f"向量写入失败(非致命): {embed_ex}")

        _page_tree.add_node(new_id, category_name, parent_id, [product_name])

        try:
            _vec_mgr.invalidate_matrix()
        except Exception:
            pass

        remove_entry(entry_id)

        return jsonify({
            "status": "ok",
            "new_category_id": new_id,
            "category_name": category_name,
            "parent_id": parent_id,
            "mount_path": mount_path,
            "message": f"已新增分类 #{new_id}({category_name})，挂载于 {mount_path}",
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
