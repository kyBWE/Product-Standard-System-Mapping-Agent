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


def _init_components():
    global _config, _db, _llm, _trgm_mgr, _vec_mgr, _rag_engine, _rag_rerank_engine, _page_engine, _page_engine_force_llm, _page_tree, _excel_reader, _evolve_scheduler, _initialized
    if _initialized:
        return

    config = ConfigManager(CONFIG_PATH)
    db_config = config.get_db_config()
    llm_config = config.get_llm_config()

    db = DBConnectionManager(db_config)
    db.initialize()
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
            f"PageIndex树构建完成: {len(page_tree.get_root_nodes())}个根节点, "
            f"共{len(page_tree._node_map)}个节点"
        )
    except Exception as e:
        logging.getLogger("WebAPI").error(f"PageIndex树构建失败: {e}")
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
        if query_vec:
            try:
                cat_vec_row = _db.execute_one(
                    "SELECT embedding FROM category_vectors WHERE category_id = %s",
                    (row["category_id"],),
                )
                if cat_vec_row and cat_vec_row.get("embedding"):
                    cat_vec = cat_vec_row["embedding"]
                    if len(query_vec) == len(cat_vec):
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
        emb = row.get("embedding")
        if not emb:
            continue
        try:
            if isinstance(emb, memoryview):
                vec = pickle.loads(bytes(emb))
            elif isinstance(emb, bytes):
                vec = pickle.loads(emb)
            elif isinstance(emb, str) and emb.startswith("["):
                vec = [float(v) for v in emb.strip("[]").split(",")]
            elif isinstance(emb, (list, np.ndarray)):
                vec = list(emb) if isinstance(emb, list) else emb.tolist()
            else:
                continue
        except Exception:
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
            "depth_distribution": path_depth_count,
            "common_ancestor": common_ancestor,
        },
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


def _llm_path_reasoning(product_name: str, vector_candidates: list):
    """LLM路径推理：让LLM分析产品应该放在哪个位置"""
    if not _llm:
        return None

    root_rows = _db.execute("SELECT category_id, category_name FROM category_texts WHERE category_pids = '{}' LIMIT 30")
    taxonomy_overview = ", ".join([r["category_name"] for r in root_rows]) if root_rows else ""

    candidate_info = ""
    for i, c in enumerate(vector_candidates[:5], 1):
        path = c.get("path", "")
        path_display = path.replace(",", " > ") if path else c["category_name"]
        candidate_info += f"{i}. #{c['category_id']} {c['category_name']} (相似度{c['similarity']:.2f})\n   路径: {path_display}\n"

    prompt = f"""你是一个标准分类体系专家。现在有一个产品无法匹配到现有标准分类，需要你推理它应该放在什么位置。

产品名称: {product_name}

标准体系一级分类概览:
{taxonomy_overview}

向量语义匹配找到的相似节点:
{candidate_info}

请分析这个产品的特征，推理它在标准体系中的合理位置。要求:
1. 分析产品的本质属性和用途
2. 判断它属于哪个一级分类
3. 在该一级分类下，找到最合适的父节点（可以是叶子节点，也可以是中间节点）
4. 如果是新兴产品或现有分类不够精确，判断是否需要在某个节点下创建新的子分类
5. 给出推理理由

**重要提示**：
- suggested_parent_id必须从上面的向量候选中选择一个实际存在的节点ID（如#21447）
- 如果你认为应该放在一个不存在的分类下（如"其他有机化学原料"），请设置should_create_new_node=true
- 此时suggested_parent_id应该是新节点的父节点ID（从向量候选中选择），new_node_name填写要创建的新节点名称
- 不要返回节点名称作为suggested_parent_id，必须是数字ID或留空

请以JSON格式返回:
{{
  "product_analysis": "对产品特征的分析",
  "primary_category": "产品属于的一级分类名称",
  "suggested_parent_id": "推荐的父节点ID（必须从上面的候选中选择，如#21447）",
  "suggested_parent_name": "推荐的父节点名称",
  "should_create_new_node": true/false,
  "new_node_name": "如果需要新建，新节点的名称（如'其他有机化学原料'）",
  "full_path": "完整路径，从一级分类到最终位置",
  "reasoning": "推理理由",
  "confidence": 0.0-1.0
}}"""

    try:
        response = _llm._call_llm(prompt, method="path_reasoning")
        result = _llm._parse_json_response(response)
        return result
    except Exception as e:
        logging.getLogger("app").warning(f"LLM路径推理失败: {e}")
        return None


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
    """对暂存箱中所有产品批量获取智能扩展建议（优化版：并行向量计算 + 批量LLM推理）"""
    try:
        _init_components()
        rows = _db.execute(
            "SELECT product_name FROM staging_box WHERE status = 'pending' ORDER BY created_at"
        )
        if not rows:
            return jsonify({"error": "暂存箱中没有待处理的产品"}), 400

        product_names = [r["product_name"] for r in rows]
        logger = logging.getLogger("app")
        logger.info(f"批量智能扩展开始: {len(product_names)}个产品")
        start_time = time.time()

        def process_product(product_name):
            try:
                match_result = _rag_rerank_engine.match(product_name)
                if match_result.match_status.value == "MATCHED":
                    return {
                        "product_name": product_name,
                        "type": "already_matched",
                        "matched_category_id": match_result.matched_category_id,
                        "matched_category_name": match_result.matched_category_name,
                        "confidence": round(match_result.confidence, 4),
                    }
                
                vector_result = _vector_semantic_locate(product_name, top_k=5)
                return {
                    "product_name": product_name,
                    "type": "need_llm",
                    "vector_candidates": vector_result.get("candidates", []),
                    "path_analysis": vector_result.get("path_analysis", {}),
                }
            except Exception as e:
                logger.warning(f"处理产品失败 {product_name}: {e}")
                return {
                    "product_name": product_name,
                    "type": "error",
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_product, name): name for name in product_names}
            parallel_results = []
            for future in as_completed(futures):
                parallel_results.append(future.result())
        
        parallel_results.sort(key=lambda x: product_names.index(x["product_name"]))
        
        vector_time = time.time()
        logger.info(f"向量计算完成，耗时: {vector_time - start_time:.2f}秒")

        suggestions = []
        need_llm_products = []
        
        for result in parallel_results:
            if result["type"] == "already_matched":
                suggestions.append({
                    "product_name": result["product_name"],
                    "status": "already_matched",
                    "matched_category_id": result["matched_category_id"],
                    "matched_category_name": result["matched_category_name"],
                    "confidence": result["confidence"],
                })
            elif result["type"] == "need_llm":
                need_llm_products.append(result)
            else:
                suggestions.append({
                    "product_name": result["product_name"],
                    "status": "error",
                    "error": result.get("error", "未知错误"),
                })

        llm_results = []
        if need_llm_products and _llm:
            root_rows = _db.execute("SELECT category_name FROM category_texts WHERE category_pids = '{}' LIMIT 30")
            taxonomy_overview = ", ".join([r["category_name"] for r in root_rows]) if root_rows else ""
            
            llm_input = [
                {
                    "product_name": p["product_name"],
                    "vector_candidates": p["vector_candidates"],
                }
                for p in need_llm_products
            ]
            
            llm_results = _llm.batch_path_reasoning(llm_input, taxonomy_overview)
            llm_time = time.time()
            logger.info(f"LLM批量推理完成，耗时: {llm_time - vector_time:.2f}秒")

        for idx, product_info in enumerate(need_llm_products):
            product_name = product_info["product_name"]
            vector_candidates = product_info["vector_candidates"]
            path_analysis = product_info["path_analysis"]
            
            llm_result = llm_results[idx] if idx < len(llm_results) else None
            llm_path_validation = None
            
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
                            trust_llm = False
                            decision_reason = f"向量相似度{vector_sim:.2f}>LLM置信度{llm_confidence:.2f}，但LLM推理更合理，仍采纳LLM建议"
                            trust_llm = True
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
                            full_path = (path.replace(",", " > ") + " > " + parent_row["category_name"]) if path else parent_row["category_name"]
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

            suggestions.append({
                "product_name": product_name,
                "status": "no_match",
                "vector_candidates": vector_candidates,
                "path_analysis": path_analysis,
                "llm_reasoning": llm_result,
                "llm_path_validation": llm_path_validation,
                "recommendation": recommendation,
            })

        total_time = time.time() - start_time
        logger.info(f"批量智能扩展完成，总耗时: {total_time:.2f}秒，平均每个产品: {total_time/len(product_names):.2f}秒")

        return jsonify({
            "status": "ok",
            "total_products": len(product_names),
            "suggestions": suggestions,
            "performance": {
                "total_time": round(total_time, 2),
                "avg_time_per_product": round(total_time / len(product_names), 2),
                "vector_time": round(vector_time - start_time, 2),
                "llm_time": round(llm_time - vector_time, 2) if need_llm_products and _llm else 0,
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

        from src.data.synonym_sanitizer import sanitize_syn_list

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

            cleaned, removed = sanitize_syn_list([product_name], cat_name)
            if removed or not cleaned:
                skipped.append({"product_name": product_name, "reason": "被清洗规则拒绝"})
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

        return jsonify({
            "status": "ok",
            "stashed": len(stashed),
            "skipped": len(skipped),
            "failed": len(failed),
            "details": {"stashed": stashed, "skipped": skipped, "failed": failed},
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500



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
