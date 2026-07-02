from __future__ import annotations
import json
import logging
import os
import time
import traceback

from flask import Flask, request, jsonify, send_from_directory

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")

from src.data.excel_reader import ExcelDataReader
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
_initialized = False


def _init_components():
    global _config, _db, _llm, _trgm_mgr, _vec_mgr, _rag_engine, _rag_rerank_engine, _page_engine, _page_engine_force_llm, _page_tree, _excel_reader, _initialized
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

    page_engine = PageIndexEngine(page_tree, llm, force_llm_each_layer=False)
    page_engine_force_llm = PageIndexEngine(page_tree, llm, force_llm_each_layer=True)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)