# 分片检索（Retrieval Pipeline）设计文档

- 日期：2026-09-22
- 状态：已获批，待写实现计划
- 范围：知识库新增「分片检索」tab，涵盖基础 dense 召回、混合检索（dense + BM25）、
  RRF / 加权融合、cross-encoder 重排、查询改写（HyDE / Multi-Query / Step-Back /
  Decomposition）、MMR 多样性、元数据过滤，以及全链路检索追踪。

---

## 1. 背景与目标

现有「知识库」组包含四个面板：文档解析、分片测试、一键入库、入库浏览——覆盖了数据生产侧，
但缺少检索侧的调试能力。本次新增「分片检索」面板与配套后端管线，目标：

1. **基础检索**：dense 向量召回 + 元数据过滤；
2. **高阶检索**：dense + BM25 双路混合召回、RRF / WeightedRanker 融合、
   cross-encoder 精排；
3. **查询改写**：HyDE（含 2025 TREC RAG 冠军方案的 α-混合）、Multi-Query、
   Step-Back、Decomposition；
4. **多样性**：MMR 去冗余；
5. **可观测性**：改写后的查询、每路原始排名与分数、融合分、重排分、阶段耗时全部可追溯。

非目标（YAGNI）：

- 不做 SPLADE 等学习型稀疏模型（v1 稀疏路只有 Milvus 内置 BM25）；
- 不做服务端 `hybrid_search` 封装（融合在 Python 侧，理由见 §3）；
- 不做 Agentic / 迭代检索（Self-RAG、CRAG 留待后续）；
- 不做跨 collection 检索（v1 固定 `ingest` collection）。

## 2. 环境前提（已确认）

- Milvus server **2.5+**：支持 BM25 Function 全文检索；使用内置 `chinese` 分析器
  （jieba 分词 + cnalphanumonly 过滤器）；
- pymilvus 已锁定 **3.0.1**；
- LLM（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`）当前未配置，计划配置：查询改写功能照做，
  未配置时相关开关置灰并提示，不影响 dense / 混合 / 重排；
- 检索目标固定为 `ingest` collection，UI 只选 database。

## 3. 路线决策

采用 **Python 侧编排**（新 `retrieval/` 包内并发多路召回 + Python 侧融合），而非直接
封装 Milvus 服务端 `hybrid_search`：

- Multi-Query / Decomposition 要求「每个查询变体各跑一组召回腿」，服务端
  `hybrid_search` 的每条 AnnSearchRequest 只能携带单个查询，无法承载；
- Python 侧融合后每路原始排名/分数都在手，满足全链路 trace；
- 融合算法（RRF / 加权）为几十行纯函数，易于单元测试；
- 编排只依赖 `VectorStore` 通用接口，后端可移植性好。

代价：RPC 数多于服务端单次 hybrid_search，但所有召回并发 fan-out，且检索请求不是
超高频路径，可接受。

## 4. 后端架构

新包 `vector_service/retrieval/`：

```
retrieval/
├── __init__.py    # 公开 RetrievalPipeline
├── base.py        # 数据类（见 §4.1）
├── channels.py    # Channel ABC；DenseChannel / BM25Channel
├── fusion.py      # rrf_fuse() / weighted_fuse()
├── transforms.py  # HyDE / MultiQuery / StepBack / Decompose
├── diversity.py   # mmr()
└── pipeline.py    # RetrievalPipeline 编排器
```

### 4.1 数据类（base.py）

- `ChannelHit`：`chunk_id: str`、`score: float`、`rank: int`、`fields: dict`；
- `ChannelRun`：`channel: str`（dense/bm25）、`query: str`、`hits: list[ChannelHit]`；
- `StageTrace`：`stage: str`、`duration_ms: int`、`detail: dict`；
- `RetrievalPlan`：`original_query`、`dense_queries: list[str]`（HyDE 改写后用于向量路）、
  `lexical_queries: list[str]`（词法路）、`sub_queries: list[str]`（分解）；
- `RetrievedChunk`：`chunk_id`、`fields`、`fusion_score`、`rerank_score: float | None`、
  `matched_channels: list[str]`；
- `RetrievalResult`：`query`、`chunks: list[RetrievedChunk]`、`plan: RetrievalPlan`、
  `channel_runs: list[ChannelRun]`、`trace: list[StageTrace]`。

### 4.2 召回通道（channels.py）

`Channel` ABC：`name` 属性 + `recall(query, top_k, filter_expr) -> list[ChannelHit]`。

- **DenseChannel**：调用 embedder（`embed_query`）得到查询向量后调
  `store.search(...)`；
- **BM25Channel**：直接以原文调 `store.search_text(...)`（BM25 全文检索，
  Milvus 端运行分析器并打分）。

Channel 只依赖 embedder 与 `VectorStore` 接口，可用 fake 实现单测。

### 4.3 融合（fusion.py）

输入为多个 `ChannelRun`，输出按融合分降序的 `list[RetrievedChunk]`，按 chunk id 去重。

- **RRF（默认）**：`score(d) = Σ 1 / (rrf_k + rank_i(d))`，`rrf_k=60`，
  rank 从 1 开始；chunk 未在某路出现时该路贡献为 0；
- **WeightedRanker**：每路分数先 min-max 归一化到 [0,1]（单路全相等时该路置 0.5），
  `score(d) = Σ w_i · norm_i(d)`；权重默认 dense=0.5 / bm25=0.5，由请求传入并归一化
  （总和为 1）。

### 4.4 查询改写（transforms.py）

所有改写通过 `chat_fn`（复用 `chunking.llm_chunker.OpenAIChatClient`，
进程内缓存）。每个方法独立容错：LLM 返回非预期 JSON 时记录 warning 并回退为原查询，
不因改写失败中断检索。

- **HyDE**：要求 LLM 为原问题生成一段假设性答案文档 h（无 JSON 约束，取纯文本）。
  分别 embed 原查询 q 与 h，dense 路使用冠军方案的混合向量：
  `q' = norm((1−α)·q + α·h)`，`hyde_alpha` 默认 0.7；
- **Multi-Query**：要求 LLM 返回 JSON `{"queries": [ ... ]}`，生成 `n_variants`
  （默认 3）个不同角度的查询变体；每个变体分别走全部启用的 channel；
- **Step-Back**：要求 LLM 返回 JSON `{"query": "..."}`，把具体问题改写为一个
  更抽象的上位问题，与原查询并列召回；
- **Decomposition**：要求 LLM 返回 JSON `{"sub_queries": [ ... ]}`，将复杂/多跳
  问题拆为原子子问题，每个子问题分别走全部启用的 channel。

多查询场景下，各变体的 ChannelRun 全部保留进 trace，并共同参与融合。

### 4.5 MMR 多样性（diversity.py）

融合完成后可选执行：

1. 取融合排名前 `candidate_pool_mmr`（默认 50）的文本；
2. 通过 embedder `embed_documents` 批量编码（一次调用）；
3. 标准 MMR：首轮选融合分最高者，之后每轮选择
   `λ·sim(d,q) − (1−λ)·max sim(d, selected)` 最大的文档，`λ` 默认 0.7，
   直至选满 top_k；
4. 相似度使用 cosine（向量先归一化）。

### 4.6 编排管线（pipeline.py）

`RetrievalPipeline(settings, store, embedder_slot, reranker_getter, chat_getter).retrieve(req, emit)`。
全部阻塞调用（embed / search / chat / rerank）经 `loop.run_in_executor` 下发；
同组召回并发（`asyncio.gather`）。阶段：

1. **validate** —— query 非空；embedder 已加载（否则 503 `embedder_unavailable`）；
   rerank 开启时校验 reranker 已加载（否则 422/503 `reranker_not_loaded`）；
   rewrite 开启但 LLM 未配置 → 503 `llm_unavailable`；探测 collection schema，
   BM25 channel 开启但集合无 sparse 字段 → 422 `retrieval_channel_unsupported`
   （响应中带迁移提示标记）；
2. **rewrite** —— 按启用方法生成 `RetrievalPlan`；
3. **recall** —— 计划中每个查询 × 每个启用 channel 并发召回，filter 表达式下推；
   每路召回上限取 `max(rerank.candidate_pool, top_k)` 再向上取整（默认上限 50）；
4. **fuse** —— RRF / weighted 融合；
5. **mmr**（可选）—— 去冗余；
6. **rerank**（可选）—— 取融合/MMR 后候选池（默认 25，≤ reranker 上限 64），
   调 cross-encoder 精排并按精排分排序，截断 top_k；未开启时按融合分截断。

每个阶段产出 `StageTrace`（耗时与关键参数），随结果返回；`emit` 用于 stream
端点的阶段事件（见 §6）。

## 5. Store 接口扩展

`stores/base.py`：

- `FieldSpec` 增加字段：`dtype` 接受 `sparse_float_vector`；varchar 字段增加
  `enable_analyzer: bool = False`、`analyzer: dict | None = None`；
- 新增抽象方法：

  ```python
  def search_text(
      self, database, collection, sparse_field, query_text: str, top_k: int = 10,
      filter_expr: str | None = None, output_fields: list[str] | None = None,
  ) -> list[Hit]: ...
  ```

  语义：以原文对 BM25 sparse 字段做全文检索（服务端分析器分词 + BM25 打分）。

`stores/_milvus_adapter.py`：

- `create_collection` 支持 `SPARSE_FLOAT_VECTOR` 字段；varchar 字段透传
  `enable_analyzer=True` 与 `analyzer_params`；注册 `Function(BM25)`
  （`input_field_names=["text"]`，`output_field_names=["sparse"]`）；
  sparse 字段建 `SPARSE_INVERTED_INDEX`（metric `BM25`）；
- 新增 `search_text`：构造 sparse/BM25 检索（`param={"metric_type": "BM25"}`，
  data 为原始查询文本），结果 Hit 归一化与现有 `search` 一致；
- 新增 collection rename / 行拷贝所需的底层能力（复用现有 query/upsert 通道）。

`stores/milvus.py`：薄转发 + 参数校验，模式与现有方法一致。

### 5.1 Schema v2 与迁移

ingest collection schema：

- v1（现有）：dense `vector` + 10 个标量字段；
- v2：在 v1 基础上增加 `sparse`（SPARSE_FLOAT_VECTOR，SPARSE_INVERTED_INDEX /
  BM25），并在 `text`（VARCHAR）上启用分析器（`analyzer_params={"type": "chinese"}`，
  `enable_analyzer=True`），绑定 BM25 Function：写入/读取 `text` 时自动生成 sparse。

ingest 的 `_ensure_collection` 对新库直接创建 v2。对已存在的 v1 collection
（schema 不可变）提供**一键迁移**：

`POST /v1/databases/{db}/collections/ingest/migrate`：

1. 新建临时 collection（v2 schema）；
2. 分页读取旧 collection 全部行（主键、标量、已有 dense 向量）；
3. 批量写入临时 collection（sparse 由 BM25 Function 从 `text` 自动生成，
   dense 向量原样携带）；
4. 校验行数一致后删除旧 collection，并将临时 collection rename 为 `ingest`；
5. 任一步失败：删除临时 collection，旧集合保持不动，返回错误信封。

检索时 `collection_info` 自动探测 v1/v2（是否存在 sparse 字段），响应中携带
`migration_available` 标记，UI 显示迁移横幅。

## 6. API

`api/retrieval.py`（prefix `/v1`，tag `retrieval`），schema 在
`schemas/retrieval.py`：

- `POST /v1/retrieval` —— 经典 JSON 响应（`RetrievalResult`）；
- `POST /v1/retrieval/stream` —— NDJSON 事件流，契约与
  `/v1/ingest/stream` 同构：

  ```
  {"type": "stage", "stage": "rewrite" | "recall" | "fuse" | "mmr" | "rerank"}
  {"type": "result", ...RetrievalResult 字段...}
  {"type": "error", "status": ..., "error": { ... }}
  ```

  pre-flight 失败（参数非法 / embedder 未加载）在流开启前以普通 JSON 信封返回。

请求体：

```json
{
  "database": "default",
  "collection": "ingest",
  "query": "string，非空",
  "top_k": 10,
  "filter": { "doc_id": "", "filename": "" },
  "channels": { "dense": true, "bm25": true },
  "fusion": {
    "method": "rrf",
    "rrf_k": 60,
    "weights": { "dense": 0.5, "bm25": 0.5 }
  },
  "rewrite": {
    "enabled": false,
    "methods": ["hyde", "multi_query", "step_back", "decompose"],
    "hyde_alpha": 0.7,
    "n_variants": 3
  },
  "mmr": { "enabled": false, "lambda": 0.7 },
  "rerank": { "enabled": true, "candidate_pool": 25 }
}
```

过滤：`filter.doc_id`（精确）/ `filter.filename`（模糊）转换为 Milvus 表达式，
复用入库浏览的转义逻辑；两条件为空时不下发。

校验规则：`top_k ∈ [1,100]`；至少启用一个 channel；`fusion.method ∈ {rrf, weighted}`；
`rrf_k ∈ [1,200]`；权重非负且至少一个 > 0；`hyde_alpha ∈ [0,1]`；
`n_variants ∈ [1,5]`；`mmr.lambda ∈ [0,1]`；`rerank.candidate_pool ∈ [top_k,64]`。

新增错误码（见 `docs/errors.md` 同步）：

- `retrieval_empty_query`（422）、`retrieval_channel_unsupported`（422，
  带 `channels` 与 `migration_available`）、`retrieval_invalid_param`（422）；
- 复用：`embedder_unavailable`（503）、`reranker_not_loaded`（503）、
  `llm_unavailable`（503）、`collection_not_found`（404）。

## 7. 前端

- 侧栏「知识库」组新增第 5 个 nav item：view key `retrieval`，
  i18n key `nav.retrieval`（中：分片检索；英：Retrieval）；面包屑与
  `NAV_LABELS` 同步注册；
- 新组件 `components/retrieval.js`（独立文件，不改动 `knowledge-base.js`），
  在 `app.js` 中 import、挂载并加入 v-show 分组；
- **模式预设**：
  - 基础：仅 dense；
  - 混合（默认）：dense + bm25 + RRF + rerank；
  - 高阶：混合 + rewrite（全部方法）；
  - 自定义：全部参数可改；
  参数区分组折叠：channels & 融合、rewrite（LLM 未配置时整组置灰并显示配置提示）、
  rerank、MMR、过滤器；
- 请求走 `/v1/retrieval/stream`：头部展示阶段进度；v1 collection（无 sparse）时
  显示迁移横幅与「一键迁移」按钮（调迁移端点后刷新）；
- 结果卡片：排名、文本、`matched_channels` 徽章、融合分 / 重排分、
  doc_id、页码、章节等元信息；
- 「检索追踪」折叠区：展示 plan 中全部改写查询、每个 ChannelRun 的原始 Top 命中、
  各阶段耗时。

## 8. 测试策略

- **纯函数单测**：
  `fusion.rrf_fuse`（排名计算、k 敏感、多路叠加、去重）、
  `fusion.weighted_fuse`（归一化、权重）、`diversity.mmr`（选择顺序、λ 边界）；
- **transforms**：用 fake `chat_fn`，覆盖正常 JSON、包裹 Markdown 代码块、
  非法 JSON / 空响应时的回退行为；
- **pipeline**：fake channel / embedder / reranker，断言阶段顺序、
  多查询 fan-out 次数、trace 完整、各类资源缺失错误；
- **store**：schema v2 构建参数（sparse 字段 / BM25 Function / chinese 分析器）、
  能力探测、迁移流程（mock adapter，断言失败时临时集合被清理、旧集合不动）；
- **API contract**：校验错误（不启 channel、参数越界）、pre-flight 503、
  stream 事件顺序；沿用现有 `tests/unit` 模式，无需活 Milvus。

## 9. 文档

- 新增 `docs/retrieval.md`：能力总览、管线图、参数调优建议、模式预设说明、
  技术出处（TREC 2025 RAG 冠军方案、HyDE / RRF 原始文献）；
- 同步：`docs/api.md`（新端点）、`docs/errors.md`（新错误码）、
  `docs/README.md`（索引）、`.env.example`（LLM 配置说明，如缺）。

## 10. 技术参考

- UTokyo-HitU at TREC 2025 RAG Track: HyDE-Enhanced Sparse-Dense Retrieval
  Fusion with LLM Reranking（多路召回 + RRF(k=60) + α=0.7 HyDE 向量混合）；
- Milvus 官方文档：Full Text Search / BM25 Function（2.5+）、
  Hybrid Search、chinese(jieba) 分析器（2.6）；
- A Survey of Query Optimization in Large Language Models（arXiv 2412.17558）：
  expansion / decomposition / disambiguation / abstraction 四类原子改写。
