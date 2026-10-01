# API 表面

所有端点的请求/响应模型用 Pydantic 校验，定义在 `src/vector_service/schemas/`。非 2xx 响应统一信封见 [errors.md](errors.md)。

> 各推理端点的详细参数、配置项和错误码见 [embedding-subsystems.md](embedding-subsystems.md)；向量库端点的 schema/upsert/search 规则见 [vector-store.md](vector-store.md)；模型热加载/卸载端点见 [model-lifecycle.md](model-lifecycle.md)；摄取管线（Docling → 分片 → 嵌入 → 写入）见 [ingest-pipeline.md](ingest-pipeline.md)。

## Health / Metrics

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 进程存活（不依赖任何后端） |
| `GET` | `/readyz` | 文本/图像嵌入器加载 + Milvus 可达；任一失败则 503 `degraded` / `not_ready` |
| `GET` | `/metrics` | Prometheus 文本格式 |
| `GET` | `/dashboard` | 内嵌调试页面（每类接口一个 tab） |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### 指标清单

`/metrics` 暴露 Prometheus 文本。按子系统：

- **嵌入 / rerank**：`vs_embedding_*`、`vs_image_embedding_*`、`vs_multimodal_embedding_*`（请求数、耗时、处理量），`vs_rerank_*`（`/v1/rerank` 端到端）。
- **检索管线**（`POST /v1/retrieval`）：
  - `vs_retrieval_requests_total` — 执行的检索次数（各比率的分母）；
  - `vs_retrieval_channel_runs_total{channel,status}` — 各通道召回腿数（`ok`/`empty`）；
  - `vs_retrieval_recalled_hits_total{channel}` — 每通道原始召回量；
  - `vs_retrieval_fusion_input_total{method}` / `vs_retrieval_fusion_output_total{method}` — **融合存活率 = output / input**；
  - `vs_retrieval_channel_hit_requests_total{channel}` — 通道对最终结果有贡献的请求数，**通道命中率 = 该值 / requests**；
  - `vs_retrieval_rerank_stage_duration_seconds{model}` — 管线内 rerank 阶段耗时；
  - `vs_retrieval_rerank_input_chars{model}` / `vs_retrieval_rerank_candidates{model}` — rerank 输入长度（字符数）与候选数分布。
- **向量库 / 运行时**：`vs_store_operation_duration_seconds`、`vs_store_vectors_total`、`vs_store_databases/collections`、`vs_model_loaded`、`vs_info`。

### 分布式追踪

OpenTelemetry（`VS_TRACING__ENABLED=true` 开启，OTLP/HTTP 导出）。一次检索的 span 树：

```
retrieval                              # database / collection / query_length / top_k
├─ route        (routing.enabled)      # router / intents_count / predicates_count
├─ rewrite                             # dense_queries / lexical_queries
├─ recall                              # legs / index_ref / served_canary
│  ├─ recall.dense                     # query_length（每个 query 一个）
│  ├─ recall.bm25
│  └─ recall.summary
├─ graph        (graph.enabled)        # entities / edges / communities
├─ fuse                                # method / chunks
├─ hydrate                             # chunks
├─ mmr          (mmr.enabled)          # pool / selected
├─ rerank       (rerank.enabled)       # model / candidates / input_chars
├─ compress     (context_budget)       # max_tokens / used_tokens / chunks
└─ expand       (context_level≠chunk)  # level / leaf_hits / anchors
```

阶段失败时根 span 记录 `exception` 事件并置为 ERROR；W3C `traceparent` 请求头把检索接入上游 trace。

## Model registry

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列已注册的全部后端（embedder / image_embedder / multimodal_embedder / reranker），用 `type` 区分 |
| `GET` | `/v1/models/{model_id}` | 单个模型元信息 |
| `POST` | `/v1/models/{model_id}/load` | 显式加载模型（详见 [model-lifecycle.md](model-lifecycle.md)） |
| `POST` | `/v1/models/{model_id}/unload` | 显式卸载模型（同上） |

## 嵌入

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/embeddings` | OpenAI 兼容文本嵌入（BGE-M3） |
| `POST` | `/v1/image_embeddings` | base64 图片嵌入（OpenCLIP ViT-L/14） |
| `POST` | `/v1/multimodal_embeddings` | 中文文本 + 图片跨模态嵌入（Chinese-CLIP） |

参数、请求体格式、错误码见 [embedding-subsystems.md](embedding-subsystems.md)。

## 重排序

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/rerank` | cross-encoder 重排，返回按相关性降序的 `{index, score}` 列表 |

参数与错误码见 [embedding-subsystems.md § 重排序](embedding-subsystems.md#重排序cross-encoder)。

## 向量库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/databases` | 列 databases |
| `POST` | `/v1/databases` | 建（body: `{"name": "..."}`） |
| `GET` | `/v1/databases/{name}` | 查元信息 |
| `DELETE` | `/v1/databases/{name}` | 删（含其下所有 collection） |
| `GET` | `/v1/databases/{db}/collections` | 列 collection |
| `POST` | `/v1/databases/{db}/collections` | 建（body 见 [vector-store.md § Collection schema](vector-store.md#collection-schema)） |
| `GET` | `/v1/databases/{db}/collections/{coll}` | 查元信息 |
| `DELETE` | `/v1/databases/{db}/collections/{coll}` | 删 |
| `PUT` | `/v1/databases/{db}/collections/{coll}/vectors` | upsert（texts / vectors / images 三选一） |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/delete` | 按 id 删 |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/get` | 按 id 取 |
| `POST` | `/v1/databases/{db}/collections/{coll}/search` | k-NN 检索（query_text / query_vector / query_image 三选一） |

详细 schema 约束、字段类型、距离度量、upsert/search 规则见 [vector-store.md](vector-store.md)。

## Backend 诊断（仅运维）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/backend/raw` | 返回当前 backend 名称与 URI |
| `POST` | `/backend/raw/call` | debug-only 透传调底层 store 方法（`VS_DEBUG=true` 时启用，否则 404） |

## 摄取管线（知识库）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/parse` | 文件 → Markdown + 元数据 |
| `POST` | `/v1/chunk` | Markdown → chunks（含 token 数 / 章节 / 页码） |
| `POST` | `/v1/jobs/ingest` | 提交异步摄取任务：校验 + spool 上传字节，`202` 返回 `job_id`；worker 在后台执行 解析→分片→嵌入→写入（失败原子回滚），可选 `add_summary` / `add_context` |
| `GET` | `/v1/jobs` / `/v1/jobs/{id}` | 列出 / 查询任务：阶段、进度、attempts、结果、错误 |
| `GET` | `/v1/jobs/{id}/events` | **SSE** 任务进度流：连接即发当前完整状态，之后每次变更（阶段切换 / 进度 tick / 重排 / 取消 / 终态）推送一帧 `job` 事件，心跳注释帧保活，终态后关闭 |
| `POST` | `/v1/jobs/{id}/cancel` | 请求取消：worker 在阶段边界观察、回滚并置 `cancelled` |
| `POST` | `/v1/retrieval` | 多路检索：dense/BM25/摘要 召回 + RRF/加权融合 + 改写 + MMR + 重排；`context_level` 可把命中上扩到 section/document（同父折叠）；`routing` 段按查询意图自动路由关键词/语义/图谱/元数据（详见 [retrieval.md](retrieval.md)） |
| `POST` | `/v1/retrieval/stream` | 同上，NDJSON 阶段事件流（stage 含 route/graph/expand → result/error） |

进程重启时，启动恢复（startup recovery）会把遗留的阶段状态任务清理半成品后重排（spool 缺失 → `failed/interrupted`）。

摄取端点的请求 / 响应 / collection schema 详见 [ingest-pipeline.md](ingest-pipeline.md)；
检索端点的参数与调优建议详见 [retrieval.md](retrieval.md)。

## 索引重建与一致性（corpus 知识库）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/databases/{db}/collections/{coll}/reindex` | 提交索引重建任务：新建物理 collection，从 SQLite 叶子重算 dense/sparse/summary 全部向量并拟合全新 BM25 统计，停为 binding 的 canary；`202` 返回 `job_id` / `index_ref`。非 corpus 逻辑 collection → 422 `invalid_request` |
| `POST` | `/v1/databases/{db}/collections/{coll}/reindex/promote` | 单事务提升 canary（`index_bindings` + `chunk_indexes` 一起切），随后删除旧物理 collection 及其 BM25 状态；无 canary → 409 `reindex_no_canary`，scope 有门禁但最近 check 未通过（或通过的是别的 canary）→ 409 `gate_blocked` |
| `POST` | `/v1/databases/{db}/collections/{coll}/consistency` | 对比 SQLite 叶子与物理行（active + canary），输出 `missing_in_milvus` / `orphans_in_milvus`；`{"repair": true}` 一键删孤儿、重算缺失（整个物理 collection 缺失则按薄 schema 重建），修复需 embedder 已加载，否则 503 `embedder_unavailable` |

重建期间新旧物理索引并存，检索侧灰度选择（详见 [retrieval.md § 索引灰度与重建](retrieval.md#索引灰度与重建)）：`/v1/retrieval` 请求体带 `index_ref` 可精确指定 active 或 canary；否则按 `sha256(query) % 100 < canary_percent` 的稳定查询哈希桶分流——相同查询总落在同一侧。

## GraphRAG 知识图谱（corpus 知识库）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/databases/{db}/collections/{coll}/graph/build` | 提交图谱构建任务（`job_type=graph_build`）：从 SQLite 叶子 LLM 抽取实体/关系/声明 → 合并 → 社区检测 → LLM 社区摘要 → 实体/社区向量写入独立薄 collection → 单事务整图替换；`202` 返回 `job_id` / `entity_collection` / `community_collection`。非 corpus collection → 422 `invalid_request`；无叶子 → 400 `graph_empty` |
| `GET` | `/v1/databases/{db}/collections/{coll}/graph` | 图谱状态：实体/边/声明/社区计数 + 社区列表（id、序号、规模、摘要） |
| `DELETE` | `/v1/databases/{db}/collections/{coll}/graph` | 删除 corpus 全部图谱行并 drop `{safe}__graph_entities` / `{safe}__graph_communities`，缺失 collection 视为成功 |

检索请求可用 `graph.enabled=true` 打开实体 / 社区 ANN 检索。数据模型、
构建管线与参数详见 [graph.md](graph.md)。

## 检索反馈（corpus 知识库）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/feedback` | 提交一条反馈（up/down/click/adopt）：答案级（chunk_id 为空）或切片级，带管道版本快照，返回 `feedback_id` |
| `GET` | `/v1/feedback` | 列出反馈（新→旧），`database`/`collection`/`kind` 过滤 + `limit`/`offset` |

数据模型与快照语义详见 [feedback.md](feedback.md)。

## 评测集（corpus 知识库）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/evaluation/sets` | 建评测集（绑定 scope）；`201` |
| `GET` | `/v1/evaluation/sets` | 列评测集，scope 过滤 + `limit`/`offset` |
| `GET` | `/v1/evaluation/sets/{id}` | 查评测集（含 question_count） |
| `DELETE` | `/v1/evaluation/sets/{id}` | 删评测集（问题/运行/结果级联） |
| `POST` | `/v1/evaluation/sets/{id}/questions` | 批量加问题（期望切片/文档/答案至少一项）；`201` |
| `GET` | `/v1/evaluation/sets/{id}/questions` | 列问题 |
| `POST` | `/v1/evaluation/sets/{id}/versions` | 冻结当前问题为不可变版本；空集 422、tag 冲突 409；`201` |
| `GET` | `/v1/evaluation/sets/{id}/versions` | 列冻结版本（newest-first） |
| `POST` | `/v1/evaluation/sets/{id}/runs` | 提交批量检索/生成运行（`eval_run` 任务），可带 `version_id` 跑冻结问题；空集 422、版本不存在 404、`include_answer` 无 LLM 503；`202` 返回 `run_id` / `job_id` |
| `GET` | `/v1/evaluation/sets/{id}/runs` | 列运行 |
| `GET` | `/v1/evaluation/runs/{id}` | 查运行：每问切片 id / Recall·MRR·nDCG·doc_hit / rerank 前后对比 / 通道归因 / 生成答案 + 聚合摘要 |
| `POST` | `/v1/evaluation/gates` | 配置 scope 的回归门禁（版本 + 绝对下限/相对跌幅，至少一项）；重复 scope 409；`201` |
| `GET` | `/v1/evaluation/gates` | 列门禁，可按 database/collection 过滤 |
| `GET` | `/v1/evaluation/gates/{id}` | 查门禁 |
| `DELETE` | `/v1/evaluation/gates/{id}` | 删门禁（checks 级联） |
| `POST` | `/v1/evaluation/gates/{id}/checks` | 对 parked canary 跑冻结问题并判定（`gate_check` 任务，`index_ref` 强制钉 canary）；无 canary 409；`202` 返回 `check_id` / `run_id` / `job_id` |
| `GET` | `/v1/evaluation/gates/{id}/checks` | 列检查（passed/failed/cancelled + 报告） |

模板参数、版本化、门禁判定口径与重跑语义详见 [evaluation.md](evaluation.md)。

## 调用示例

```bash
# 1. 嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'

# 2. 创建 database
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -d '{"name": "tenant-a"}'

# 3. 在指定 database 下创建 collection
curl -X POST http://localhost:8080/v1/databases/tenant-a/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "primary_field": "id",
    "scalar_fields": [
      {"name": "id", "dtype": "varchar", "is_primary": true, "max_length": 64},
      {"name": "category", "dtype": "varchar", "max_length": 64},
      {"name": "price", "dtype": "float"}
    ],
    "vector_field": {"name": "vector", "dim": 1024, "metric_type": "cosine"},
    "index_params": [
      {"field_name": "vector", "metric_type": "cosine", "index_type": "HNSW",
       "params": {"M": 16, "efConstruction": 200}}
    ]
  }'

# 4. upsert（自动嵌入）
curl -X PUT http://localhost:8080/v1/databases/tenant-a/collections/products/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "ids": ["1"],
    "texts": ["a wireless mouse"],
    "fields": [{"category": "mouse", "price": 29.9}]
  }'

# 5. 检索
curl -X POST http://localhost:8080/v1/databases/tenant-a/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "query_text": "computer accessory",
    "top_k": 5,
    "filter_expr": "category == '\''mouse'\'' and price < 100"
  }'

# 6. 列 databases / 列 collection / drop
curl http://localhost:8080/v1/databases
curl http://localhost:8080/v1/databases/tenant-a/collections
curl -X DELETE http://localhost:8080/v1/databases/tenant-a/collections/products
curl -X DELETE http://localhost:8080/v1/databases/tenant-a
```
