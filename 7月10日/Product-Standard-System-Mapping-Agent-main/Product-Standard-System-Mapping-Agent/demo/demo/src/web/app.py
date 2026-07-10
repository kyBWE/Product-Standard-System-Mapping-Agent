from __future__ import annotations
import json
import logging
import os
import time
import traceback

from flask import Flask, request, jsonify, send_from_directory

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")

from src.data.excel_reader import ExcelDataReader
from src.data.taxonomy_utils import (
    allocate_next_category_id,
    build_category_path_fields,
    format_category_path,
    locate_expansion_parent,
)
from src.engine.llm_adapter import LLMAdapter
from src.engine.page_index_engine import PageIndexEngine
from src.engine.rag_match_engine import RAGMatchEngine
from src.engine.rerank_adapter import RerankAdapter
from src.index.page_index_tree import PageIndexTree
from src.index.trgm_index_manager import TrgmIndexManager
from src.index.vector_index_manager import VectorIndexManager
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.db_manager import DBConnectionManager
from src.models.enums import EngineType, MatchStatus
from src.orchestration.self_evolve_scheduler import SelfEvolveScheduler

app = Flask(__name__, static_folder=os.path.join(WEB_DIR, "static"), static_url_path="/static")

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
    trgm_mgr = TrgmIndexManager(db)
    vec_mgr = VectorIndexManager(
        db,
        embedding_model=llm_config.embedding_model,
        embedding_dimension=llm_config.embedding_dimension,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
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
    except Exception as e:
        logging.getLogger("WebAPI").error(f"PageIndex树构建失败: {e}")

    rerank_adapter = RerankAdapter(rerank_config) if rerank_config.api_key else None

    page_engine = PageIndexEngine(page_tree, llm, force_llm_each_layer=False, vec_mgr=vec_mgr, rerank=rerank_adapter, trgm_mgr=trgm_mgr)
    page_engine_force_llm = PageIndexEngine(page_tree, llm, force_llm_each_layer=True, vec_mgr=vec_mgr, rerank=rerank_adapter, trgm_mgr=trgm_mgr)

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
    standard_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), standard_file)
    _evolve_scheduler = SelfEvolveScheduler(llm, db, excel_reader, match_config, standard_file_path)
    _initialized = True


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

        rows5 = _db.execute("SELECT COUNT(*) as cnt FROM expansion_suggestions")
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
            """SELECT product_name, matched_category_id, confidence, match_status, engine_type, llm_participated, created_at
               FROM match_results ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        results = []
        for r in rows:
            results.append({
                "product_name": r["product_name"],
                "matched_category_id": r["matched_category_id"],
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

        suggested_parent_id, analysis, mount_path = locate_expansion_parent(
            _llm, _page_tree, product_name
        )

        suggested_category_name = analysis.get("suggested_category_name", product_name)
        reason = analysis.get("reason", "")
        confidence = analysis.get("confidence", 0)
        path_note = f"挂载路径: {mount_path}" if mount_path else "挂载路径: 未确定"

        _db.execute(
            """INSERT INTO expansion_suggestions
               (product_name, suggested_parent_id, suggested_category_name, suggested_level_position, llm_analysis, status)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (product_name, suggested_parent_id or None, suggested_category_name,
             mount_path or None, f"置信度={confidence:.2f} | {path_note} | {reason}", "PENDING_REVIEW"),
        )

        return jsonify({
            "status": "ok",
            "product_name": product_name,
            "suggested_parent_id": suggested_parent_id,
            "suggested_category_name": suggested_category_name,
            "mount_path": mount_path,
            "llm_analysis": reason,
            "confidence": confidence,
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

        category_pids, category_group_name = build_category_path_fields(_page_tree, parent_id)
        mount_path = format_category_path(_page_tree, parent_id)

        _db.execute(
            """INSERT INTO category_texts (category_id, category_name, category_pids, syn_list, category_group_name)
               VALUES (%s, %s, %s, %s, %s)""",
            (new_id, category_name, category_pids, [product_name], category_group_name),
        )

        from src.index.onnx_embedder import ONNXEmbedder
        embedder = ONNXEmbedder()
        text_parts = [category_name, product_name]
        embedding = embedder.encode(" ".join(text_parts))
        import numpy as np
        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

        _db.execute(
            """INSERT INTO category_vectors (category_id, category_name, embedding, syn_list)
               VALUES (%s, %s, %s, %s)""",
            (new_id, category_name, str(emb_list), [product_name]),
        )

        _page_tree.add_node(new_id, category_name, parent_id, [product_name])

        verify_ok = False
        try:
            verify_result = _rag_rerank_engine.match(product_name)
            if verify_result.matched_category_id == new_id and verify_result.confidence >= 0.3:
                verify_ok = True
        except Exception:
            pass

        if not verify_ok:
            try:
                _db.execute("DELETE FROM category_texts WHERE category_id = %s", (new_id,))
                _db.execute("DELETE FROM category_vectors WHERE category_id = %s", (new_id,))
                node = _page_tree.get_node(new_id)
                if node and node.parent:
                    node.parent.children = [c for c in node.parent.children if c.category_id != new_id]
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
            parent_name = ""
            if parent_id:
                parent_node = _page_tree.get_node(parent_id)
                parent_name = parent_node.category_name if parent_node else ""
            results.append({
                "product_name": r["product_name"],
                "suggested_parent_id": parent_id,
                "suggested_parent_name": parent_name,
                "mount_path": mount_path,
                "suggested_category_name": r["suggested_category_name"],
                "llm_analysis": r["llm_analysis"],
                "status": r["status"],
                "created_at": str(r["created_at"]),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)