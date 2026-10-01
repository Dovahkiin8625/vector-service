# 分片检索（Retrieval）

知识库「分片检索」面板与 `POST /v1/retrieval` 端点对 `ingest` collection
执行多路检索，覆盖从基础向量召回到高阶 RAG 检索的完整栈。

## 管线

```
query
  → route     可选，按查询意图自动选择通道与图谱/元数据路由
  → rewrite   HyDE / Multi-Query / Step-Back / Decomposition（需 LLM）
  → recall    dense ANN × N ‖ BM25 全文 × N ‖ 摘要 ANN × N（并发 fan-out）
  → graph     可选，实体 ANN + 子图扩展 + 社区 ANN（详见 GraphRAG 小节）
  → fuse      RRF（默认）/ weighted
  → mmr       可选，去冗余
  → rerank    可选，cross-encoder 精排
  → expand    可选，按 parent_id 上扩到 section / document 并折叠同父命中
```

全链路可观测：结果之外还返回改写计划、每路原始 Top 命中与分数、融合分、
重排分和各阶段耗时（dashboard「检索追踪」区）。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/retrieval` | 多路检索，返回完整 JSON（结果 + plan + 各路命中 + traces） |
| `POST` | `/v1/retrieval/stream` | 同上，NDJSON 阶段事件流（stage → result/error） |
| `GET` | `/v1/retrieval/capabilities` | LLM 配置状态 |

## 模式预设

| 模式 | channels | 融合 | rewrite | MMR | rerank |
|------|----------|------|---------|-----|--------|
| 基础 basic | dense | — | – | – | – |
| 混合 hybrid（默认）| dense + bm25 + summary | RRF | – | – | ✓ |
| 高阶 advanced | dense + bm25 + summary | RRF | 全部方法 | – | ✓ |
| 自定义 custom | 任意 | 任意 | 任意 | 任意 | 任意 |

> 选择 hybrid / advanced 时 dashboard 自动勾选摘要路，可手动取消。

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
- **summary**：摘要路是对分片「要点」而非原文的 ANN，适合提问与原文措辞
  差异大、但与摘要语义贴近的场景；摘要路按每条 dense 查询 fan-out，腿数与
  dense 一致。RRF 下勾选即参与融合；weighted 下用 `weights.summary` 调比例
  （dashboard 默认 0.3），设为 0.0 即静音该路。

## 查询意图路由

请求体加 `routing` 段打开（默认关闭），由路由器代替显式的通道选择，
在四类索引间按查询形态分流：**关键词**（BM25）、**语义**（dense /
summary ANN）、**图谱**（GraphRAG graph 阶段）、**结构化元数据**
（谓词编译到薄索引标量）。

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | false | 路由总开关 |
| `mode` | auto | `auto`（配置了 LLM 用 LLM 分类，否则启发式）、`heuristic`、`llm` |

- **启发式信号（确定性、零成本）**：错误码（`ERR-4042`）、hex 哈希、
  路径、版本号、驼峰/蛇形标识符等代码形态 → BM25 独占；引号短语与
  短词（≤3 英文词 / ≤6 汉字）→ BM25 + dense 兜底；疑问句与长文本
  → dense（+summary 候选）；关系措辞（"关系 / 之间 / 合作 /
  relationship / connected"）→ graph。
- **LLM 模式**：一次 JSON 分类调用返回 intents / filters / query；
  `mode=llm` 且未配置 LLM → 预检 503 `llm_unavailable`；调用或解析
  失败 → 静默回退启发式（signals 记 `llm_fallback`）。
- **可用性门控（best-effort）**：collection 无 sparse / summary
  字段、图谱未构建时，对应腿被丢弃并在 signals 记 `*_unavailable`，
  不会 422；通道被全丢时强制 dense（`dense_forced`）。
- **元数据谓词**：查询中内联的 `key op value`，支持
  `filename / title / author / status / mime`（`:`/`=` 为子串）、
  `page_count` 与薄索引 `chunk_index`（`>=` 等比较）、`doc_id`
  直取；文档级谓词在 corpus 解析为 doc_id 列表再编译，解析为空则
  跳过召回。识别为谓词的片段从后续各阶段的 query 中剥离。
- 结果保留**原始 query**（含谓词）；路由决策在响应 `routing` 段
  （router / intents / signals / 实际 channels / predicates /
  剥离后 query），route trace 排在所有阶段最前。

```json
{
  "query": "author:张三 page>=10 退款流程怎么处理？",
  "routing": {"enabled": true, "mode": "auto"}
}
```

## 摘要召回路

入库时勾选 **LLM 摘要**（`add_summary=true`）会让 LLM 为每个分片生成一段
摘要并独立持久化，检索时新增一路摘要向量 ANN：

- 集合携带 `summary`（VARCHAR 2048，**nullable**）与
  `summary_vector`（FLOAT_VECTOR，dim = embedder.dim，HNSW cosine，
  M=16 / efConstruction=200）。
- **`summary` 与 `context` 的区别**：`context`（Anthropic contextual
  retrieval）只在嵌入时拼到正文前形成一个向量，不单独存储；`summary` 是
  独立存储的产物，有自己的向量与召回路，并在结果中作为 `fields.summary`
  回显。
- 摘要路与 dense 路一样对查询改写产物 fan-out：HyDE 复用预设混合向量，
  multi-query / step-back / decompose 每条改写各跑一条摘要 ANN。
- **权重语义**：默认 RRF 忽略 `weights`，只要 `channels.summary=true` 就按
  reciprocal-rank 立即贡献；`weights.summary` 默认 0.0，仅在 weighted
  融合下起静音/调比作用。
- **兜底**：单个分片的摘要调用失败时 `summary` 为 null，`summary_vector`
  在嵌入阶段用该分片**原文**嵌入兜底——每行必有摘要向量（Milvus 向量
  字段不可空、无默认值）。
- **摘要向量总是写入**：即使入库请求未带 `add_summary`，也会嵌入
  `summary_vector`（未配置 LLM 时全部走原文兜底向量，不报错），只是
  `summary` 标量不产生。
- **缺少字段即拒绝**：在没有 `summary_vector` 字段的集合（如通过通用
  建集接口自建的集合）上开启摘要路，返回 **422
  `retrieval_channel_unsupported`**（`channels: ["summary"]`）。

BM25 全文能力在客户端完成：薄 collection 不存正文、不做服务端分词，只持
`sparse`（SPARSE_FLOAT_VECTOR）。摄取 / 重建时服务端用 jieba 词表的 BM25
统计（`data/corpus/bm25/*.bm25.json`，派生物）把切片正文编码成稀疏向量后
写入，查询时用同一套统计编码。统计在摄取后自动 refit；删除切片 /
collection 时走 discard，下次查询或一致性修复按 SQLite 叶子重建。在没有
`sparse` 字段的集合上开启 bm25 路返回 422（`channels: ["bm25"]`）。

## GraphRAG 图谱检索

需要先对 collection 跑过图谱构建（`POST .../graph/build`，详见
[graph.md](graph.md)）：corpus 中存有实体 / 边 / 声明 / 社区行，另有
`{safe}__graph_entities` / `{safe}__graph_communities` 两个薄向量
collection。请求体加 `graph` 段打开（默认关闭）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | false | graph 阶段总开关 |
| `entity_top_k` | 10 | 实体 ANN 种子数（1–50） |
| `community_top_k` | 3 | 社区 ANN 命中数；0 跳过社区检索 |
| `neighbor_depth` | 1 | 种子经 edges 的 BFS 深度；0 = 仅种子 |
| `include_claims` | true | 返回与图中实体相关的声明 |
| `claims_limit` | 20 | 返回声明条数上限 |

graph 阶段在 recall 之后、fuse 之前执行，产出与 `chunks` 并列的
`graph` 响应段，不影响切片的融合 / MMR / 重排：

- 实体 ANN 种子携带相似度分数；BFS 扩展节点（上限 500 个，防止枢纽
  实体拖入整图）的 `score` 为 null，排在种子之后；
- `edges` 给出子图内的边（端点、描述、权重）；
- 社区命中携带 `community_index`、摘要 `summary` 与成员实体 id 列表；
- 声明携带主体 / 客体实体 id、所在 `chunk_id`、类型、状态与陈述。

预检：图谱行缺失或任一向量 collection 不存在 → **422
`graph_not_built`**，提示先构建；构建 collection 无叶子 → 400
`graph_empty`。

## 上下文压缩（token 预算）

默认最终结果只按 `top_k` 截断；在请求体加 `context_budget.max_tokens` 后，
最终切片改为按 token 预算裁剪：

```json
{"context_budget": {"max_tokens": 2000}}
```

- 压缩在 rerank / top_k 截断**之后**、small-to-large 扩展之前执行，计数取
  每个切片 hydrated 的 `token_count`；按分数从高到低填充预算。
- 放不下的切片被跳过（后面更小的切片仍可补进剩余预算，不是到首个超限
  就停）。
- 最高分切片即使单独超预算也必保留——宁可返回超预算的最佳答案，不返回空。
- 未设置（默认）时该阶段不运行。

compress trace：`{"stage": "compress", "detail": {"max_tokens": ...,
"used_tokens": ..., "chunks": ...}}`。

## 引用回链与高亮定位

每个返回切片都携带引用所需的定位字段（hydrated content 的一部分）：

- `doc_id` / `filename`：文档身份；
- `chunk_index`：文档内切片序号；
- `page_number`：切片所在 PDF 页；
- `char_start` / `char_end`：切片在原文 Markdown 中的字符区间，前端可据此
  高亮/定位；
- section/document 扩展结果同样携带祖先自己的字符区间与页码。

字段为可空：非 PDF 文档 `page_number` 为 null，未产出字符区间的行
`char_start` / `char_end` 为 null。

## Small-to-large 扩展（parent-document retrieval）

请求参数 `context_level` 控制最终返回的粒度：

| 值 | 行为 |
|----|------|
| `chunk`（默认） | 命中保持在叶子切片 |
| `section` | 沿 `parent_id` 上扩到所属章节，返回章节全文 |
| `document` | 上扩到文档根，返回整篇 Markdown |

- **打分始终在叶子粒度进行**：expand 是管线最后一个阶段（mmr / rerank /
  `top_k` 截断之后），精排精度不受父级长文本稀释。
- **同父折叠 dedup**：多个叶子命中同一祖先时只输出一条结果——
  `fields.child_chunk_ids` 收集全部叶子 id，`fields.source_chunk_id`
  保留分数最高（rerank 分优先，否则融合分）的叶子，融合分 / 重排分取
  max，`matched_channels` 取并集；输出顺序按首条命中。
- 内容直接取自 SQLite 的祖先行（`text` / `section_header` /
  `page_number` / `char_start` / `char_end`），Milvus 不参与扩展。
- 链缺失（祖先行不存在）时退回到链上最深的现存行；完全无法解析的
  叶子原样透传。

expand 阶段 trace：`{"stage": "expand", "detail": {"level": "section",
"leaf_hits": N, "anchors": M}}`。

层级如何在摄取时产出见
[ingest-pipeline.md § 层级切片](ingest-pipeline.md#层级切片small-to-large)。

## 索引灰度与重建

逻辑 collection 名与物理 Milvus collection 名分离：`index_bindings` 表为每个
逻辑 collection 持有 `active_ref` / `canary_ref` / `canary_percent` / `model`；
没有 binding 行时物理名等于逻辑名。重建（`POST .../reindex`）把新向量写进
全新物理 collection（`{coll}__rebuild_{12hex}`），旧索引继续服务；提升
（`.../reindex/promote`）在单事务内切换 binding 与 `chunk_indexes` 注册表，
随后删除旧物理 collection。

检索侧选择物理 collection 的规则（`_resolve_physical`）：

1. **请求级 pin**：请求体带 `index_ref`，必须是该逻辑 collection 的 active 或
   canary，否则 422 `retrieval_unknown_index_ref`；灰度对照验证时用它固定
   两侧。
2. **稳定查询哈希桶**：未 pin 且 canary 在停时，
   `bucket = int.from_bytes(sha256(query utf-8) 的前 8 字节, "big") % 100`，
   `bucket < canary_percent` 走 canary，否则走 active。哈希只依赖查询文本：
   相同查询永远落在同一侧，灰度比例按查询而非按请求精确生效。
3. 无 binding 行：物理名 == 逻辑名。

recall trace 携带本次解析结果：`index_ref`（实际搜索的物理名）、
`served_canary`、`canary_percent`。

重建 / 提升 / 一致性端点见
[api.md § 索引重建与一致性](api.md#索引重建与一致性corpus-知识库)。

## NDJSON 事件流

`POST /v1/retrieval/stream` 返回 `application/x-ndjson`，事件类型：

```
{"type": "stage", "stage": "route" | "rewrite" | "recall" | "graph" | "fuse" | "mmr" | "rerank" | "expand"}
{"type": "result", ...RetrievalResponse 字段...}
{"type": "error", "status": ..., "error": {"code": ..., "message": ...}}
```

错误映射分两段：

- **预检（preflight）错误**——query 为空、参数校验失败（422
  `invalid_request`，FastAPI 校验信封）、embedder/reranker 未加载、LLM 未配置、
  集合不存在、集合缺少召回路所需字段（422 `retrieval_channel_unsupported`）等——
  在流开启前返回，是普通 HTTP JSON 错误信封。
- **飞行中（post-flight）错误**——检索开跑后 store / embedder / reranker 的
  后端故障——以 NDJSON `error` 事件收尾并携带规范错误码，例如
  503 `store_unavailable` / `embedder_unavailable` / `reranker_not_loaded` /
  `reranker_error`，未预期故障为 500 `internal`。

## 能力探测

```
GET /v1/retrieval/capabilities
```

返回：

```json
{
  "llm_configured": false
}
```

- `llm_configured`：是否已配置 `VS_LLM__BASE_URL` 与 `VS_LLM__MODEL`。

## 降级行为

- LLM 未配置（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`）：仅查询改写不可用
  （API 返回 503 `llm_unavailable`，UI 置灰），其余能力不受影响。
- 改写返回非法内容：该改写回退原查询并记 warning，不中断检索。

## 参考

- UTokyo-HitU, TREC 2025 RAG Track：HyDE 增强 sparse-dense 融合 + LLM 重排
  （RRF k=60，HyDE α=0.7）
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Cormack et al., *Reciprocal Rank Fusion*（RRF）
