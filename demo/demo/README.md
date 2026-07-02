# 产品-标准体系映射智能体

## 项目简介

本系统是一个Python智能体程序，实现企业非标产品名称与产品标准分类体系的智能匹配，包含RAG向量匹配主方案、PageIndex无向量轻量化备选方案、体系自进化闭环模块。

## 系统架构

```
入口层 (Entry Layer)
├── load_data       — 数据加载入口
├── run_matching    — 匹配测试入口
├── update_synonyms — 同义词更新入口
└── generate_taxonomy — 体系生成入口

调度层 (Orchestration Layer)
├── MatchOrchestrator    — 匹配流程编排器
├── SelfEvolveScheduler  — 自进化调度模块
└── ResultExporter       — 结果导出模块

引擎层 (Engine Layer)
├── RAGMatchEngine  — RAG向量匹配引擎
├── PageIndexEngine — PageIndex树形匹配引擎
└── LLMAdapter      — LLM交互适配器

索引层 (Index Layer)
├── VectorIndexManager — 向量索引管理器
├── TrgmIndexManager   — pg_trgm索引管理器
└── PageIndexTree      — PageIndex树形索引

数据层 (Data Layer)
├── ExcelDataReader      — Excel数据读取器
├── DBConnectionManager  — 数据库连接管理器
└── ConfigManager        — 配置管理器
```

## 环境要求

- Python 3.9+
- PostgreSQL 14+（需安装 pg_trgm、pgvector 扩展）
- OpenAI API 兼容的大语言模型服务

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化PostgreSQL数据库

```bash
# 创建数据库
createdb product_standard_mapping

# 执行建表脚本
psql -d product_standard_mapping -f scripts/init_db.sql
```

### 3. 配置环境变量

```bash
# Linux/macOS
export DB_USER="your_db_user"
export DB_PASSWORD="your_db_password"
export LLM_API_KEY="your_llm_api_key"

# Windows PowerShell
$env:DB_USER = "your_db_user"
$env:DB_PASSWORD = "your_db_password"
$env:LLM_API_KEY = "your_llm_api_key"
```

### 4. 修改配置文件

编辑 `config.yaml`，确认数据库连接信息、LLM服务地址等参数正确。

## 运行说明

### 数据加载

```bash
# 默认配置
python -m src.entry.load_data

# 指定配置文件
python -m src.entry.load_data path/to/config.yaml
```

### 匹配测试

```bash
# RAG向量匹配（批量）
python -m src.entry.run_matching

# RAG向量匹配（单条测试）
python -m src.entry.run_matching config.yaml rag "测试产品名称"

# PageIndex树形匹配（批量）
python -m src.entry.run_matching config.yaml page_index
```

### 同义词更新

```bash
# 批量自动发现
python -m src.entry.update_synonyms

# 手动追加同义词
python -m src.entry.update_synonyms config.yaml category_id new_synonym
```

### 体系生成

```bash
# 生成扩展建议
python -m src.entry.generate_taxonomy

# 指定配置文件
python -m src.entry.generate_taxonomy path/to/config.yaml
```

## 核心算法说明

### RAG向量匹配主方案

1. **粗召回**：对 product_name 进行向量化，在 category_vectors 表中执行向量相似度检索（top_k=20），同时执行 pg_trgm 模糊文本检索（阈值0.3，limit=20），两种结果取并集，按融合分数排序：`融合分数 = 0.6 × 向量相似度 + 0.4 × 文本相似度`
2. **精匹配**：对 product_name 调用LLM提取关键词并过滤候选，然后将 product_name 与候选节点提交LLM语义打分
3. **综合置信度**：`0.4 × 粗召回融合分数 + 0.6 × LLM语义分数`
4. **结果判定**：≥0.5 已匹配，0.3~0.5 低置信度，<0.3 无匹配

### PageIndex无向量备选方案

1. 从根节点开始逐层规则匹配（关键词命中+字符串包含）
2. 规则匹配返回多候选时调用LLM消歧
3. 无匹配时回溯到父节点选择次优候选
4. 到达叶节点输出结果

### 体系自进化闭环

- **同义词自动更新**：置信度≥0.95 且文本相似度<0.3 → LLM同义校验 → 去重 → 追加到syn_list → 持久化回写
- **标准体系扩展**：所有候选置信度<0.3 → LLM品类分析 → 生成扩展建议 → 标记"待审核"

## 数据文件说明

| 文件 | 说明 |
|------|------|
| 产品标准体系.xlsx | 标准分类知识库（category_id, category_name, category_pids, category_group_name, syn_list） |
| temp_company_product_0522_1.xlsx | 待匹配企业产品数据（product_name） |

## 输出文件

| 文件 | 说明 |
|------|------|
| output/match_results_*.csv | 匹配结果（product_name, matched_category_id, confidence, match_status） |
| output/expansion_suggestions_*.csv | 扩展建议 |
| logs/app_*.log | 运行日志 |

## 数据库表结构

| 表名 | 说明 |
|------|------|
| category_vectors | 标准分类向量表（含IVFFlat向量索引） |
| category_texts | 标准分类文本表（含GIN三元组索引） |
| match_results | 匹配结果表 |
| synonym_updates | 同义词更新记录表 |
| expansion_suggestions | 体系扩展建议表 |

## 项目结构

```
demo/
├── config.yaml                  # 配置文件
├── requirements.txt             # Python依赖
├── scripts/
│   └── init_db.sql             # 数据库建表脚本
├── src/
│   ├── data/
│   │   └── excel_reader.py     # Excel数据读取模块
│   ├── engine/
│   │   ├── llm_adapter.py      # LLM交互适配器
│   │   ├── rag_match_engine.py # RAG向量匹配引擎
│   │   └── page_index_engine.py# PageIndex树形匹配引擎
│   ├── entry/
│   │   ├── load_data.py        # 数据加载入口
│   │   ├── run_matching.py     # 匹配测试入口
│   │   ├── update_synonyms.py  # 同义词更新入口
│   │   └── generate_taxonomy.py# 体系生成入口
│   ├── index/
│   │   ├── vector_index_manager.py  # 向量索引管理器
│   │   ├── trgm_index_manager.py    # pg_trgm索引管理器
│   │   └── page_index_tree.py       # PageIndex树形索引
│   ├── infrastructure/
│   │   ├── config_manager.py   # 配置管理器
│   │   ├── db_manager.py       # 数据库连接管理器
│   │   └── logger.py           # 结构化日志
│   ├── models/
│   │   ├── api_result.py       # API结果模型
│   │   ├── category_node.py    # 标准分类节点模型
│   │   ├── config_models.py    # 配置模型
│   │   ├── enums.py            # 枚举定义
│   │   ├── evolve_models.py    # 自进化模型
│   │   ├── index_result.py     # 索引结果模型
│   │   ├── match_result.py     # 匹配结果模型
│   │   └── treenode.py         # 树节点模型
│   └── orchestration/
│       ├── match_orchestrator.py    # 匹配流程编排器
│       ├── result_exporter.py       # 结果导出模块
│       └── self_evolve_scheduler.py # 自进化调度模块
├── output/                      # 输出目录
└── logs/                        # 日志目录
```
