# 分片检索（Retrieval）

知识库「分片检索」面板与 `POST /v1/retrieval` 端点对 `ingest` collection
执行多路检索，覆盖从基础向量召回到高阶 RAG 检索的完整栈。

## 管线

```
query
  → rewrite   HyDE / Multi-Query / Step-Back / Decomposition（需 LLM）
  → recall    dense ANN × N 条查询 ‖ BM25 全文 × N 条查询（并发 fan-out）
  → fuse      RRF（默认）/ weighted
  → mmr       可选，去冗余
  → rerank    可选，cross-encoder 精排
```

全链路可观测：结果之外还返回改写计划、每路原始 Top 命中与分数、融合分、
重排分和各阶段耗时（dashboard「检索追踪」区）。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/retrieval` | 多路检索，返回完整 JSON（结果 + plan + 各路命中 + traces） |
| `POST` | `/v1/retrieval/stream` | 同上，NDJSON 阶段事件流（stage → result/error） |
| `GET` | `/v1/retrieval/capabilities` | LLM 配置状态、ingest schema 版本、是否可迁移 |
| `POST` | `/v1/databases/{db}/collections/ingest/migrate` | ingest 集合 v1 → v2 一键迁移 |

## 模式预设

| 模式 | channels | 融合 | rewrite | MMR | rerank |
|------|----------|------|---------|-----|--------|
| 基础 basic | dense | — | – | – | – |
| 混合 hybrid（默认）| dense + bm25 | RRF | – | – | ✓ |
| 高阶 advanced | dense + bm25 | RRF | 全部方法 | – | ✓ |
| 自定义 custom | 任意 | 任意 | 任意 | 任意 | 任意 |

## 参数调优建议

- **RRF**：对分数尺度不敏感，首选；`rrf_k` 越小排名靠后的命中贡献衰减越快。
- **weighted**：需要明确 dense/词法权重（如关键词强信号的日志/法典场景可调高 bm25），
  每路先 min-max 归一化。
- **HyDE**：事实型问题收益最大；`hyde_alpha`（默认 0.7）是假设答案在混合向量中的占比，
  LLM 对领域不熟时调低。
- **Multi-Query**：缓解措辞不匹配，代价是召回腿数 ×n。
- **Step-Back / Decomposition**：多跳、综合型问题适用。
- **MMR**：结果需要去重/覆盖多角度时开启；λ 越低越多样。
- **rerank**：以融合后候选池（默认 25，上限 64）做 cross-encoder 精排，
  精度显著提升，成本随候选池线性增长。

## Schema v2 与迁移

BM25 全文检索要求 schema v2：`text`（varchar）启用 `chinese` 分析器，
新增 `sparse`（SPARSE_FLOAT_VECTOR）并注册 BM25 Function——写入时自动从
`text` 生成 sparse 向量。新库直接使用 v2；已有 v1 库执行一键迁移：

```
POST /v1/databases/{db}/collections/ingest/migrate
```

成功返回 `{"database", "collection", "rows", "schema_version": 2}`。
迁移会拷贝全部行（dense 向量原样携带，sparse 由 BM25 Function 重新生成），
校验行数后删旧并 rename；任一步失败都会删除临时集合
（`ingest_migrate_tmp`）并保留旧集合。

**故障窗口（数据丢失风险）**：在「旧集合已删除、rename 完成」之间存在一个
数据丢失窗口——此时 rename 若失败，清理逻辑会连同临时集合一起删除，新旧
两个集合都不存在（API 返回 503 `store_unavailable`）。唯一可能让临时集合
物理残留的子情形，是清理时执行的 drop 本身也失败；该 drop 的错误可能覆盖
原始错误（v1 不做异常链聚合）。因此建议：

- 在业务低峰、确认没有并发写入时执行迁移；
- 对大集合或重要集合，迁移前先在外部保留一份备份。

**分页上限（已知限制）**：迁移按 200 行/页分页拷贝，Milvus 要求
offset + limit < 16,384，因此行数超过约 16,200 的集合无法由本端点迁移——
会安全失败（旧集合保留，返回 503）。更大的集合需要等待未来基于迭代器的
迁移路径。

## NDJSON 事件流

`POST /v1/retrieval/stream` 返回 `application/x-ndjson`，事件类型：

```
{"type": "stage", "stage": "rewrite" | "recall" | "fuse" | "mmr" | "rerank"}
{"type": "result", ...RetrievalResponse 字段...}
{"type": "error", "status": ..., "error": {"code": ..., "message": ...}}
```

错误映射分两段：

- **预检（preflight）错误**——query 为空、参数校验失败（422
  `invalid_request`，FastAPI 校验信封）、embedder/reranker 未加载、LLM 未配置、
  集合不存在等——在流开启前返回，是普通 HTTP JSON 错误信封。
- **飞行中（post-flight）错误**——检索开跑后 store / embedder / reranker 的
  后端故障——以 NDJSON `error` 事件收尾并携带规范错误码，例如
  503 `store_unavailable` / `embedder_unavailable` / `reranker_not_loaded` /
  `reranker_error`，未预期故障为 500 `internal`。

## 能力探测

```
GET /v1/retrieval/capabilities?database=default
```

返回（实际字段仅此三个）：

```json
{
  "llm_configured": false,
  "schema_version": 1,
  "migration_available": true
}
```

- `llm_configured`：是否已配置 `VS_LLM__BASE_URL` 与 `VS_LLM__MODEL`；
- `schema_version`：该库 ingest 集合的 schema 版本（探测不到集合时为 1）；
- `migration_available`：当前为 v1、可执行迁移时为 `true`。

## 降级行为

- LLM 未配置（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`）：仅查询改写不可用
  （API 返回 503 `llm_unavailable`，UI 置灰），其余能力不受影响。
- 改写返回非法内容：该改写回退原查询并记 warning，不中断检索。

## 参考

- UTokyo-HitU, TREC 2025 RAG Track：HyDE 增强 sparse-dense 融合 + LLM 重排
  （RRF k=60，HyDE α=0.7）
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Cormack et al., *Reciprocal Rank Fusion*（RRF）
