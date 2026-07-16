# 体系拓展建议 - 逻辑与流程

## 1. 背景

当产品名无法匹配到现有标准分类时（NO_MATCH 或 LOW_CONFIDENCE），系统通过 LLM 推理给出分类路径建议，暂存后积累到一定数量进行聚类分析，人工审核后决定是否加入标准体系。

核心设计原则：**LLM 自由规划分类路径，不受现有标准体系树约束**。路径从大类逐级细分到小类，如：

```
石油、化工、医药产品 > 化学原料及化学制品 > 专项化学用品 > 高功能化工产品 > 碳纤维增强复合材料
```

---

## 2. 整体流程

```
用户输入产品名 → 匹配引擎 → NO_MATCH/LOW_CONFIDENCE
                                    │
                    ┌───────────────┴───────────────┐
                    │ 前端自动触发或用户点"暂存待聚类" │
                    └───────────────┬───────────────┘
                                    │
                              ① 暂存（stash）
                                    │
                              ② 聚类分析（cluster）
                                    │
                              ③ 人工审核（approve/reject）
                                    │
                              ④ 入库生效
```

---

## 3. 阶段一：暂存（stash）

### 触发条件

- 匹配结果为 `NO_MATCH` 时前端自动触发
- 匹配结果为 `LOW_CONFIDENCE` 时用户手动点击"暂存待聚类"按钮

### API

```
POST /api/expansion/stash
Body: { "product_name": "碳纤维" }
```

### 处理逻辑

1. 调用 `_llm.suggest_free_path(product_name, taxonomy_overview)`
   - `taxonomy_overview`：当前标准体系一级分类概览（根节点名称 + 前几个子节点），帮助 LLM 参考但不约束
   - LLM 自由规划完整分类路径，返回：
     ```json
     {
       "full_path": "石油、化工、医药产品 > 化学原料及化学制品 > 专项化学用品 > 碳纤维增强复合材料",
       "path_parts": ["石油、化工、医药产品", "化学原料及化学制品", "专项化学用品", "碳纤维增强复合材料"],
       "suggested_category_name": "碳纤维增强复合材料",
       "reason": "碳纤维属于高性能纤维材料，应在化学原料下逐级细分",
       "confidence": 0.72
     }
     ```
   - 路径深度由 LLM 根据产品特性决定：宽泛产品 2-3 层，专业产品 4-6 层

2. 将结果存入 `pending_pool.json`
   ```json
   {
     "id": "e_001",
     "product_name": "碳纤维",
     "suggested_parent_id": "",
     "suggested_parent_name": "",
     "suggested_category_name": "碳纤维增强复合材料",
     "path": [
       { "level": 1, "category_id": null, "category_name": "石油、化工、医药产品", "is_new": true },
       { "level": 2, "category_id": null, "category_name": "化学原料及化学制品", "is_new": true },
       { "level": 3, "category_id": null, "category_name": "专项化学用品", "is_new": true },
       { "level": 4, "category_id": null, "category_name": "碳纤维增强复合材料", "is_new": true }
     ],
     "path_text": "石油、化工、医药产品 > 化学原料及化学制品 > 专项化学用品 > 碳纤维增强复合材料",
     "confidence": 0.72,
     "llm_reason": "碳纤维属于高性能纤维材料...",
     "source": "web",
     "created_at": "2026-07-16T12:00:00"
   }
   ```

3. 返回前端显示建议路径

### 关键设计

- `suggested_parent_id` 为空：暂存阶段不绑定现有树节点
- `path` 中所有节点 `is_new: true`、`category_id: null`：路径由 LLM 自由生成
- `path_text`：完整路径文本，用于后续聚类分组

---

## 4. 阶段二：聚类分析（cluster）

### 触发条件

- 用户在前端暂存池面板点击"聚类分析"按钮
- 建议暂存池积累 10+ 条后再执行聚类，效果更好

### API

```
POST /api/expansion/cluster
Body: { "batch_size": 15 }
```

### 处理逻辑

1. 从 `pending_pool.json` 读取所有条目

2. **按 `path_text` 首段（大类）分组**
   ```
   "石油、化工、医药产品" → [碳纤维, 航空煤油, mRNA疫苗, ...]
   "机械、设备产品"       → [工业机器人, 3D打印机, ...]
   "电子信息、仪器仪表产品" → [智能手表, 柔性显示屏, ...]
   ```

3. **分批调用 LLM 聚类**（每批 ≤ batch_size 条，默认 15）
   - 调用 `_llm.cluster_products(batch, taxonomy_overview)`
   - LLM 重新审视所有条目，将语义相近的归为一组，给出统一的 `full_path`
   - 无法归类的列为 outlier

   LLM 返回示例：
   ```json
   {
     "clusters": [
       {
         "group_name": "新能源发电设备",
         "full_path": "电气机械及器材 > 新能源发电设备 > 风力发电设备",
         "product_indices": [1, 3, 5],
         "reason": "风电叶片、充电桩、储能锂电池均属于新能源设备"
       }
     ],
     "outliers": [2]
   }
   ```

4. **仅 1 条的大类直接归入 outlier**，不调用 LLM

5. 生成 `cluster_report.json`
   ```json
   {
     "version": 1,
     "cluster_time": "2026-07-16T12:30:00",
     "cluster_method": "llm",
     "total_entries": 50,
     "cluster_count": 8,
     "outlier_count": 5,
     "clusters": [
       {
         "cluster_id": "c_001",
         "full_path": "电气机械及器材 > 新能源发电设备",
         "suggested_category_name": "新能源发电设备",
         "merged_category_name": "新能源发电设备",
         "product_names": ["风电叶片", "储能锂电池", "充电桩"],
         "entries": ["e_021", "e_022", "e_023"],
         "entry_count": 3,
         "avg_confidence": 0.62,
         "star_rating": 2,
         "status": "PENDING_REVIEW",
         "is_llm_clustered": true,
         "llm_reason": "..."
       }
     ],
     "outliers": [
       {
         "entry_id": "e_050",
         "product_name": "特医食品",
         "path_text": "食品、饮料、烟、酒类产品 > 特殊膳食食品 > 特殊医学用途食品",
         "reason": "该大类下仅1条，无法聚类"
       }
     ]
   }
   ```

6. 返回前端显示聚类报告

### 分批策略

- 按 `path_text` 首段分组后，每组内按 `batch_size` 切分
- 每批独立调用 LLM，失败不影响其他批次
- 失败批次中的条目降级为 outlier

---

## 5. 阶段三：人工审核

### 5.1 批准簇（approve_cluster）

```
POST /api/expansion/approve_cluster
Body: {
  "cluster_id": "c_001",
  "category_name": "新能源发电设备",    // 可选，覆盖簇的分类名
  "parent_id": "9800"                  // 可选，指定挂载父节点
}
```

#### 处理逻辑

1. 校验簇状态为 `PENDING_REVIEW`

2. **确定挂载父节点 `parent_id`**（优先级从高到低）：
   - 前端传入的 `parent_id`
   - 簇自身的 `suggested_parent_id`
   - 从 `full_path` 中逆序匹配现有树节点：
     ```
     full_path: "电气机械及器材 > 新能源发电设备 > 风力发电设备"
     逆序查找: "风力发电设备" → 不存在
               "新能源发电设备" → 不存在
               "电气机械及器材" → 存在(category_id=9800) ✓
     ```
   - 全部匹配不到 → 返回 400，要求前端指定

3. 分配新 `category_id`

4. 写入数据库：
   - `category_texts` 表：category_id, category_name, category_pids, syn_list(产品名列表), category_group_name
   - `category_vectors` 表：embedding + vec_bgem3 向量

5. 更新内存：
   - `_page_tree.add_node()` 更新树
   - `_vec_mgr.invalidate_matrix()` 刷新向量矩阵

6. 更新报告：簇状态 → `APPROVED`

7. 从暂存池移除已批准条目

### 5.2 拒绝簇（reject_cluster）

```
POST /api/expansion/reject_cluster
Body: {
  "cluster_id": "c_001",
  "return_to_pool": false    // 是否退回暂存池
}
```

- 簇状态 → `REJECTED`
- `return_to_pool=true`：条目退回暂存池，可重新聚类
- `return_to_pool=false`：条目直接删除

### 5.3 单独批准（approve_single）

```
POST /api/expansion/approve_single
Body: {
  "entry_id": "e_050",
  "category_name": "特殊医学用途食品",   // 可选
  "parent_id": "1385"                     // 可选
}
```

- 处理逻辑与 approve_cluster 类似，但只处理单条
- `parent_id` 确定逻辑同上（前端传入 > 自身 > path_text 逆序匹配）

---

## 6. 阶段四：入库生效

批准后系统自动完成：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `allocate_next_category_id()` | 分配全局唯一分类ID |
| 2 | 写入 `category_texts` | 分类名、路径、同义词（产品名列表） |
| 3 | 调 embedding API | 生成分类名+产品名的向量 |
| 4 | 写入 `category_vectors` | embedding + vec_bgem3 |
| 5 | `_page_tree.add_node()` | 内存树新增节点 |
| 6 | `_vec_mgr.invalidate_matrix()` | 标记向量矩阵失效，下次查询时重建 |

生效后，该新分类可被所有匹配引擎检索到。

---

## 7. 数据文件

| 文件 | 用途 |
|------|------|
| `data/pending_pool.json` | 暂存池，存储待聚类的扩展建议条目 |
| `data/cluster_report.json` | 聚类报告，存储聚类结果和审核状态 |

---

## 8. API 汇总

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/expansion/stash` | POST | 暂存一条扩展建议 |
| `/api/expansion/pool` | GET | 查看暂存池 |
| `/api/expansion/pool_stats` | GET | 暂存池统计 |
| `/api/expansion/pool_remove` | POST | 从暂存池移除条目 |
| `/api/expansion/cluster` | POST | 执行聚类分析 |
| `/api/expansion/cluster_report` | GET | 查看聚类报告 |
| `/api/expansion/approve_cluster` | POST | 批准某个簇 |
| `/api/expansion/reject_cluster` | POST | 拒绝某个簇 |
| `/api/expansion/approve_single` | POST | 单独批准某条 |

---

## 9. 已知限制与待改进

1. **挂载点选择**：当 `full_path` 中无节点匹配现有树时，需前端传入 `parent_id`，但前端目前缺少选择挂载点的 UI
2. **聚类质量**：LLM 聚类结果依赖 prompt 质量，同类产品可能被分到不同簇
3. **中间层级创建**：当前批准只创建叶子节点，`full_path` 中的中间层级（如"新能源发电设备"）不会自动创建，需多次批准或手动处理
4. **并发安全**：`pending_pool.json` 和 `cluster_report.json` 是文件读写，无锁机制，高并发下可能丢数据