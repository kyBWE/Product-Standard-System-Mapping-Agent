-- ============================================
-- 产品-标准体系映射智能体 数据库初始化脚本
-- PostgreSQL + pg_trgm
-- ============================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- 表1：标准分类向量表（使用BYTEA存储Python端向量）
-- ============================================
CREATE TABLE IF NOT EXISTS category_vectors (
    category_id     TEXT        PRIMARY KEY,
    category_name   TEXT        NOT NULL,
    syn_list        TEXT[]      DEFAULT '{}',
    embedding       BYTEA,
    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_category_vectors_id
    ON category_vectors (category_id);

-- ============================================
-- 表2：标准分类文本表
-- ============================================
CREATE TABLE IF NOT EXISTS category_texts (
    category_id         TEXT        PRIMARY KEY,
    category_name       TEXT        NOT NULL,
    syn_list            TEXT[]      DEFAULT '{}',
    category_pids       TEXT[]      DEFAULT '{}',
    category_group_name TEXT        NOT NULL DEFAULT '',
    created_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_category_texts_name_trgm
    ON category_texts USING gin (category_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_category_texts_syn_trgm
    ON category_texts USING gin (syn_list gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_category_texts_pids
    ON category_texts USING gin (category_pids);

-- ============================================
-- 表3：匹配结果表
-- ============================================
CREATE TABLE IF NOT EXISTS match_results (
    id                  SERIAL      PRIMARY KEY,
    product_name        TEXT        NOT NULL,
    matched_category_id TEXT        DEFAULT NULL,
    confidence          NUMERIC(6,4) DEFAULT 0,
    match_status        TEXT        NOT NULL DEFAULT 'NO_MATCH',
    engine_type         TEXT        NOT NULL DEFAULT 'RAG_VECTOR',
    llm_participated    BOOLEAN     DEFAULT TRUE,
    vector_similarity   NUMERIC(6,4) DEFAULT NULL,
    trgm_similarity     NUMERIC(6,4) DEFAULT NULL,
    coarse_score        NUMERIC(6,4) DEFAULT NULL,
    llm_score           NUMERIC(6,4) DEFAULT NULL,
    candidates_snapshot JSONB       DEFAULT '[]',
    created_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_match_results_product
    ON match_results (product_name);

CREATE INDEX IF NOT EXISTS idx_match_results_status
    ON match_results (match_status);

CREATE INDEX IF NOT EXISTS idx_match_results_category
    ON match_results (matched_category_id);

-- ============================================
-- 表4：同义词更新记录表
-- ============================================
CREATE TABLE IF NOT EXISTS synonym_updates (
    id              SERIAL      PRIMARY KEY,
    category_id     TEXT        NOT NULL,
    new_synonym     TEXT        NOT NULL,
    llm_verified    BOOLEAN     DEFAULT FALSE,
    trigger_reason  TEXT        NOT NULL,
    trgm_similarity NUMERIC(6,4) DEFAULT NULL,
    match_confidence NUMERIC(6,4) DEFAULT NULL,
    status          TEXT        NOT NULL DEFAULT 'COMPLETED',
    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_synonym_updates_category
    ON synonym_updates (category_id);

CREATE INDEX IF NOT EXISTS idx_synonym_updates_status
    ON synonym_updates (status);

-- ============================================
-- 表5：体系扩展建议表
-- ============================================
CREATE TABLE IF NOT EXISTS expansion_suggestions (
    id                      SERIAL      PRIMARY KEY,
    product_name            TEXT        NOT NULL,
    suggested_parent_id     TEXT        DEFAULT NULL,
    suggested_category_name TEXT        DEFAULT NULL,
    suggested_level_position TEXT       DEFAULT NULL,
    llm_analysis            TEXT        DEFAULT '',
    status                  TEXT        NOT NULL DEFAULT 'PENDING_REVIEW',
    reviewed_by             TEXT        DEFAULT NULL,
    reviewed_at             TIMESTAMP   DEFAULT NULL,
    created_at              TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_expansion_suggestions_status
    ON expansion_suggestions (status);

CREATE INDEX IF NOT EXISTS idx_expansion_suggestions_parent
    ON expansion_suggestions (suggested_parent_id);
