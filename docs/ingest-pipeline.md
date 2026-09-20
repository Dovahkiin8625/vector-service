# 摄取管线（Docling → Chunk → Embed → Milvus）

为知识库（RAG）场景设计的端到端文档摄取管线。Lumos 等上层应用只需 POST 一个文件 + 目标 collection，
即可在服务端串起 **Docling 解析 → 递归 markdown 分片 → BGE-M3 嵌入 → Milvus 写入** 四个步骤，
失败时按 `doc_id` 原子回滚已写入的向量。

> 该子系统与 [embedding-subsystems.md](embedding-subsystems.md) 共享已加载的 embedder；
> chunk 步骤独立于模型（纯文本操作），parser 步骤独立于模型（Docling 是文档转换库）。

---

## 端点速查

| 方法 | 路径 | 用途 |
|------|------|------|
| `POST` | `/v1/parse` | 文件 → Markdown + 元数据 |
| `POST` | `/v1/parse/stream` | 同上，但返回 NDJSON 事件流（上传 % + 逐页解析进度） |
| `POST` | `/v1/chunk` | Markdown → chunks（含 token 数 / 章节 / 页码） |
| `POST` | `/v1/ingest` | 一体化：文件 → 解析 → 分片 → 嵌入 → Milvus |
| `POST` | `/v1/ingest/stream` | 同上，但返回 NDJSON 阶段事件流（含逐页解析进度） |

三个端点均接收 `multipart/form-data`（`/v1/chunk` 是 `application/json`）。dashboard 在
**知识库** 导航组下实际调用的是两个流式端点（`/v1/parse/stream`、`/v1/ingest/stream`），
非流式变体保留给 API 客户端使用。

---

## `POST /v1/parse`

把任意支持的文档格式转成 Markdown + 结构化元数据。

### 请求

```
POST /v1/parse
Content-Type: multipart/form-data

file: <binary>      # 必填
```

支持 MIME：

| MIME | 解析器 | 备注 |
|------|--------|------|
| `application/pdf` | Docling | 含 OCR / 表格识别 / 阅读顺序 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (.docx) | Docling | |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` (.pptx) | Docling | |
| `text/html` | Docling | |
| `text/markdown` | MarkdownParser | passthrough |
| `text/plain` | MarkdownParser | passthrough |

最大文件大小由 `VS_PARSER__MAX_FILE_SIZE_MB` 控制（默认 100 MB）。

### 响应

```json
{
  "markdown": "# 文档标题\n\n第一段...",
  "metadata": {
    "page_count": 12,
    "title": "...",
    "author": "...",
    "mime_type": "application/pdf",
    "filename": "report.pdf"
  }
}
```

- `markdown` —— 完整 Markdown 文本，保留标题层级、表格、列表、code block。
- `metadata.page_count` —— 文档页数（PDF / PPT），其他格式可能为 `null`。
- Docling converter 为进程级单例：首次调用会触发懒加载（约 10-30 秒模型加载）；设置
  `VS_PARSER__AUTO_LOAD=true` 可在启动时预热（dashboard 部署默认开启），首个请求不再付冷启动代价。

错误码：`unsupported_mime`（415）/ `file_too_large`（413）/ `parser_failed`（500）/ `parser_unavailable`（503）。

---

## `POST /v1/parse/stream`

multipart 契约与 `/v1/parse` 完全相同，但响应是 `application/x-ndjson` —— 每行一个 JSON 事件，
dashboard 用它渲染「上传字节 % → 逐页解析进度」：

```
{"type": "stage", "stage": "parse"}
{"type": "progress", "stage": "parse", "page": 1, "total": 12}
{"type": "progress", "stage": "parse", "page": 2, "total": 12}
...
{"type": "result", "markdown": "# 文档标题\n\n...", "metadata": {"page_count": 12, ...}}
```

- `stage` —— 流开始时固定发一个 `{"stage": "parse"}`。
- `progress` —— 分页二进制格式（PDF / DOCX / PPTX / HTML）每完成一页发一个事件，
  `page` 为已完成页数、`total` 为总页数；文本格式（MD / TXT）不产生 `progress` 事件。
  事件严格有序：全部页码事件先于终态事件。
- 终态事件二选一：成功为 `result`（字段同 `ParseResponse`）；解析中失败为
  `{"type": "error", "status": 500, "error": {"code": "parser_failed", ...}}`，
  此时 HTTP 状态码仍是 200 —— 流已经开始。
- **预检失败**（MIME 不支持 / 文件超限 / 空文件）发生在流开启之前，仍按普通 JSON 错误信封返回
  （415 / 413 / 400，`Content-Type: application/json`），客户端需同时兼容两种响应。

curl 示例：

```bash
curl -N -F "file=@report.pdf" http://localhost:8080/v1/parse/stream
```

---

## `POST /v1/chunk`

把 Markdown 切成 chunks。可单独调试 chunk 策略而不必跑完整 ingest 流程。

### 请求

```json
POST /v1/chunk
Content-Type: application/json

{
  "markdown": "# 标题\n\n正文...",
  "chunk_size": 500,
  "chunk_overlap": 75,
  "metadata": {
    "filename": "report.pdf",
    "title": "示例"
  }
}
```

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `markdown` | string | 必填 | 待分片的 Markdown 文本 |
| `chunk_size` | int | 500 | 单 chunk 最大 token 数（cl100k_base） |
| `chunk_overlap` | int | 75 | 相邻 chunk 的重叠 token 数，必须 `< chunk_size` |
| `metadata` | object | — | 透传到每个 chunk 的字段；当前仅 `title` / `filename` / `page_count` 被消费为 chunk 内字段 |

约束（422 `invalid_request`）：
- `64 ≤ chunk_size ≤ 4096`
- `0 ≤ chunk_overlap < chunk_size`
- `chunk_overlap ≤ 512`

### 响应

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "text": "# 标题\n\n第一段...",
      "token_count": 487,
      "section_header": "1. Intro > 1.1 Background",
      "page_number": 1
    },
    {
      "chunk_index": 1,
      "text": "继续...",
      "token_count": 412,
      "section_header": "2. Details",
      "page_number": 2
    }
  ]
}
```

### 分片算法

递归 markdown-aware 分片器，按以下顺序尝试切分：

1. **Markdown header**（`#` / `##` / `###` ...）—— 顶层按 header 切分，header 路径作为 breadcrumb。
2. **段落**（双换行 `\n\n`）。
3. **句子**（句号 / 问号 / 感叹号 + 后置空白）。
4. **词**（兜底，确保即使无标点也能切）。

累积 tokens 达到 `chunk_size` 即输出当前 chunk，重叠区取上一个 chunk 末尾 `chunk_overlap` tokens。
**code block**（`` ``` `` 包围）视为原子单元，内部不再切分。

token 计数使用 `tiktoken.get_encoding("cl100k_base")`。

错误码：`invalid_request`（422）/ `chunk_failed`（500）。

---

## `POST /v1/ingest`

一键端到端：上传文件 → 解析 → 分片 → 嵌入 → Milvus 写入。
**任何步骤失败都会原子回滚** —— 已 upsert 的向量会按 `doc_id` filter 删除，避免半成品数据污染 collection。

### 请求

```
POST /v1/ingest
Content-Type: multipart/form-data

file:          <binary>            # 必填，表单字段
database:      lumos               # 表单字段；不存在时按固定 ingest schema 自动创建
collection:    kb_a1b2c3d4e5f6    # 表单字段；不存在时自动创建，已存在则校验 dim 必须匹配 embedder.dim
chunk_size:    500                 # 可选，表单字段，默认 500
chunk_overlap: 75                  # 可选，默认 75
embed_model:   bge-m3              # 必填，必须已加载
metadata:      {"title": "...", "author": "...", "filename": "..."}   # 可选，JSON 字符串
```

`embed_model` 必须是已通过 `POST /v1/models/{id}/load` 加载的 embedder；不指定 `is_query`
（上传路径走文档嵌入，前缀由服务端控制）。

### 响应

```json
{
  "doc_id": "a3f9c2e1-...-...-...-...",
  "chunk_count": 23,
  "page_count": 12,
  "tokens_used": 12453
}
```

### 流式变体：`POST /v1/ingest/stream`

multipart 契约与 `/v1/ingest` 完全相同，但响应是 `application/x-ndjson` —— 每行一个 JSON 事件，
dashboard 用它渲染「上传 → 解析 → 分片 → 嵌入 → 写入」阶段步进器：

```
{"type": "stage", "stage": "parse"}
{"type": "progress", "stage": "parse", "page": 1, "total": 12}
{"type": "progress", "stage": "parse", "page": 2, "total": 12}
...
{"type": "stage", "stage": "chunk"}
{"type": "stage", "stage": "embed"}
{"type": "stage", "stage": "upsert"}
{"type": "result", "doc_id": "...", "chunk_count": 23, "page_count": 12, "tokens_used": 12453}
```

- `stage` 事件共 4 个，名称固定（`parse` / `chunk` / `embed` / `upsert`），属于公开 API 契约。
- parse 阶段内可能穿插 `progress` 事件（逐页，字段同 `/v1/parse/stream`；仅 PDF / DOCX / PPTX /
  HTML，文本上传没有），全部位于 parse 与 chunk 两个 stage 事件之间。
  字节级上传进度不经过该流，由客户端从 multipart 上传本身取（dashboard 用 XHR `upload.onprogress`）。
- 终态事件二选一：成功为 `result`（字段同 `IngestResponse`）；管道内失败为
  `{"type": "error", "status": 503, "error": {"code": "...", "message": "..."}}`，
  此时 HTTP 状态码仍是 200 —— 流已经开始。
- **预检失败**（参数非法 / MIME 不支持 / 文件超限 / embedder 未加载）发生在流开启之前，
  仍按普通 JSON 错误信封返回（4xx/503，`Content-Type: application/json`），客户端需同时兼容两种响应。
- 文档解析后未产生任何分片时，只发出 `parse`、`chunk` 两个 stage 事件，随后直接 `result`
  （`chunk_count: 0`），不会进入嵌入 / 写入。

### Collection schema（写入约定）

目标 collection 不存在时按下列固定 schema 自动创建（database 也会自动创建）；
已存在则只校验向量维度，schema 必须与下面兼容：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `varchar` PK | 是 | 形如 `{doc_id}_{chunk_index}` 的确定性主键（重试不会产生重复） |
| `doc_id` | `varchar` | 是 | UUID4，用于过滤 / 回滚 |
| `chunk_index` | `int64` | 是 | 文档内 chunk 序号 |
| `text` | `varchar` | 是 | chunk 文本（用于 RAG 检索回显） |
| `section_header` | `varchar` | 是 | breadcrumb 章节路径 |
| `page_number` | `int64` | 否 | 来源页码 |
| `title` | `varchar` | 否 | 来自 metadata.title |
| `author` | `varchar` | 否 | 来自 metadata.author |
| `page_count` | `int64` | 否 | 文档总页数 |
| `filename` | `varchar` | 否 | 来自 metadata.filename |
| `token_count` | `int64` | 是 | chunk 的 token 数 |
| `vector` | `float_vector` | 是 | 维度 = embedder.dim（如 BGE-M3 = 1024） |

> Lumos 端会按此 schema 自动创建 collection（参见 lumos 项目的 `KnowledgeBaseConfig` + `VectorServiceClient.ensure_collection`）。
> 直接调用 `vector-service` 的运维可手动 `POST /v1/databases/{db}/collections` 创建。

### 原子回滚

服务端用 try/except 包裹整条管道。一旦以下任一步失败，已成功 upsert 的向量会按
`filter_expr = 'doc_id == "{new_doc_id}"'` 删除，向上抛出错误，collection 保持一致。

可能的失败点与对应错误码：

| 失败点 | 错误码 | HTTP |
|--------|--------|------|
| 文件解析 | `parser_failed` | 500 |
| 分片 | `chunk_failed` | 500 |
| 嵌入（模型未加载 / 推理失败） | `embedder_unavailable` | 503 |
| Milvus 写入 | `store_unavailable` | 503 |
| 目标 collection 不存在 | `collection_not_found` | 404 |
| 数据库不存在 | `database_not_found` | 404 |
| embedder.dim 与 collection 不匹配 | `dimension_mismatch` | 422 |

---

## Dashboard 调试面板

新加的 "知识库" 导航组（侧栏底部）下三个面板：

| 面板 | 端点 | 功能 |
|------|------|------|
| 文档解析 | `POST /v1/parse/stream` | multipart 上传（字节 %）+ 逐页解析进度条 → Markdown 预览 + 元数据 |
| 文本分片 | `POST /v1/chunk` | Markdown 输入 → 可折叠 chunk 列表（含 token / 页码 / 章节 / 原文） |
| 一体化摄取 | `POST /v1/ingest/stream` | database/embed_model 联动选择 + multipart 上传，5 阶段步进器（parse 阶段显示 n/total 页） → 摄取结果统计 |

---

## 典型工作流（Lumos 知识库场景）

```
Lumos 用户上传 PDF
  │
  ▼ Lumos 后端 BackgroundTask
  │
  ▼ 调用 vector-service /v1/ingest
  │
  ├─ 1. Docling 解析 PDF → Markdown
  ├─ 2. RecursiveChunker 切 Markdown → chunks
  ├─ 3. BGE-M3 嵌入每个 chunk text → 1024 维向量
  └─ 4. Milvus upsert 到 collection kb_xxx（doc_id 前缀）
      │
      ▼ 失败任一步：按 doc_id filter 删除已 upsert 的向量，向上抛错
      ▼ 成功：返回 {doc_id, chunk_count, page_count, tokens_used}
  │
  ▼ Lumos 写本地 SQLite + 落盘原文
```

检索时（用户查询）：

```
Lumos 用户 query
  │
  ▼ vector-service /v1/embeddings（embed query，BGE-M3 自动加检索前缀）
  │
  ▼ vector-service /v1/databases/.../search（k-NN，top_k=50）
  │
  ▼ vector-service /v1/rerank（cross-encoder 重排，top_n=8）
  │
  ▼ 返回带 filename / page / section 引用
```

---

## 实现细节

### Docling converter 单例

`parsers/docling_parser.py::DoclingParser` 维护一个进程级 `DocumentConverter` 单例（懒加载）。
首次调用 `parse()` 时实例化；后续调用复用实例，避免每次都重新初始化模型。模型权重
按需从 HuggingFace 下载（`VS_PARSER__AUTO_DOWNLOAD=true` 时）。

### Chunk row 主键

`{doc_id}_{chunk_index}` 是确定性的 —— 同一文件重复 ingest 会覆盖而非重复入库。
删除时按 `doc_id` filter 一次性清空文档所有 chunk，原子性强。

### Chunker 配置

分片器在 `chunking/recursive_chunker.py`。它是纯 Python 实现，不依赖 Docling，
因此可以单独被 `/v1/chunk` 端点直接调用，方便调参。

### 与 Lumos 的集成模式

Lumos 是薄客户端：

- **不引入 Docling/PyTorch 依赖** —— 这些只在 vector-service 进程内加载。
- **不实现 chunker** —— 通过 `/v1/chunk` 调参即可。
- **不直接调 Milvus** —— 所有 collection / vectors 操作都走 vector-service。
- **持久化 KB 元数据** —— 知识库名、chunk 配置、原文落盘路径等放在 Lumos 的 SQLite。

Lumos 后端通过 httpx 调用 vector-service，超时 300s（`/v1/ingest` 首次可能更久）。
详见 lumos 项目的 `KnowledgeManager` + `VectorServiceClient`。
