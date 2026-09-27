# 摄取管线（Docling → Chunk → Embed → Milvus）

为知识库（RAG）场景设计的端到端文档摄取管线。Lumos 等上层应用只需 POST 一个文件 + 目标 collection，
即可在服务端串起 **Docling 解析 → 文本分片（5 种可选策略）→ BGE-M3 嵌入 → Milvus 写入** 四个步骤，
失败时按 `doc_id` 原子回滚已写入的向量。默认分片策略为 `recursive`（递归 markdown-aware），
不带新字段的旧请求行为与重构前完全一致。

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
profile: auto       # 可选，auto | standard | native | vlm
```

支持 MIME：

| MIME | 解析器 | 备注 |
|------|--------|------|
| `application/pdf` | Docling | 含选择性 OCR / 表格识别 / 阅读顺序 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (.docx) | Docling | 始终走 SimplePipeline，profile 不生效 |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` (.pptx) | Docling | 同上 |
| `text/html` / `application/xhtml+xml` | Docling | 同上 |
| `image/jpeg`（.jpg/.jpeg） | Docling | 图片输入，走 OCR 管线 |
| `image/png` / `image/tiff`（.tif/.tiff） | Docling | 同上 |
| `image/webp` / `image/bmp` | Docling | 同上 |
| `text/markdown` | MarkdownParser | passthrough |
| `text/plain` | MarkdownParser | passthrough |

最大文件大小由 `VS_PARSER__MAX_FILE_SIZE_MB` 控制（默认 100 MB）。

### 解析方案（profile）

| profile | PDF 管线 | 适用场景 / 取舍 |
|---------|----------|----------------|
| `auto`（默认） | 按文档类型选择 | 带文本层的**数字 PDF → `native`**（亚秒级纯文字抽取）；**扫描/混合/无法探测的 PDF 与图片 → `standard`**（版面分析 + 选择性 OCR）；DOCX/PPTX/HTML 始终走 model-free SimplePipeline（复用 standard converter）。类型由上传前的 PyMuPDF 文本层探针（`parsers/pdf_probe.py`）判定 |
| `standard` | DocLayNet 版面分析 + TableFormer v2 表格 + **选择性 RapidOCR**（PP-OCRv4，ONNX Runtime） | 数字版页面的文字层直接可用、**不触发 OCR**；扫描页/图片区域只对未被文字层覆盖的版面簇做 OCR；混合 PDF 按区域分别处理。兼顾性能与效果（GPU 约 4 页/秒，OCR 页约 1 页/秒） |
| `native` | NativePdfPipeline（model-free，Rust/qpdf docling-parse） | 极快的纯文字抽取，但表格还原弱；仅对 PDF 生效（独立图片在此 profile 下仍走 Standard OCR） |
| `vlm` | VlmPipeline 端到端视觉模型（默认 `ibm-granite/granite-docling-258M`，`VS_PARSER__VLM_PRESET` 可换） | 复杂版面/手写/低质扫描的备选；**权重不预下载**，首次调用才拉取（约 1 GB），速度约 2 页/秒 |

OCR 语言由 `VS_PARSER__OCR_LANGS` 控制（默认 `["ch"]`，中文+英文 PP-OCR 模型）；
设备由 `VS_PARSER__DEVICE` 控制（`auto`/`cpu`/`cuda`）。

### 图片落盘

`VS_PARSER__SAVE_IMAGES=true`（默认）时，文档中的图片保存到
`VS_PARSER__ARTIFACTS_DIR/<文档标识>/images/`（默认 `./data/artifacts/<id>/images/`），
markdown 中以 HTTP URL 引用：

```markdown
![](/artifacts/0a1b2c3d…/images/image_000000_ab12cd.png)
```

服务把 `VS_PARSER__ARTIFACTS_URL_PREFIX`（默认 `/artifacts`）挂载为静态目录。
图片按内容哈希去重命名（`image_NNNNNN_<hash>.png`）；纯文本文档不会留下空目录。

### 响应

```json
{
  "markdown": "# 文档标题\n\n第一段...\n\n![](/artifacts/…/images/image_000000_ab12.png)",
  "metadata": {
    "page_count": 12,
    "title": "...",
    "author": "...",
    "mime_type": "application/pdf",
    "profile": "standard",
    "doc_kind": "digital",
    "scanned_pages": 0,
    "images_count": 1,
    "ocr_pages": 0,
    "tables_count": 2
  },
  "images": ["/artifacts/…/images/image_000000_ab12.png"]
}
```

- `markdown` —— 完整 Markdown 文本，保留标题层级、表格、列表、code block。
- `metadata.page_count` —— 文档页数（PDF / PPT），其他格式可能为 `null`。
- `metadata.doc_kind` —— PDF 文字层探测结果：`digital` / `scanned` / `mixed` / `unknown`（探测失败）。
- `metadata.profile` —— 实际执行的管线（`standard` / `native` / `vlm`）。
- `metadata.scanned_pages` —— 无可用文字层的页数；`metadata.images_count` —— 落盘图片数。
- `metadata.ocr_pages` —— 实际经过 OCR 阶段的页数（native/vlm/SimplePipeline 为 0，图片逐页 OCR）；
  `metadata.tables_count` —— 还原的表格数。
- `images` —— 与 markdown 中一致的图片 URL 列表（关闭落盘或纯文本文档时为空）。
- Docling converters 为进程级单例并按 profile 分别缓存：首次调用会触发懒加载（约 10-30 秒模型加载）；
  设置 `VS_PARSER__AUTO_LOAD=true` 可在启动时预热默认（standard）converter，首个请求不再付冷启动代价。

错误码：`unsupported_mime`（415）/ `file_too_large`（413）/ `invalid_profile`（400）/
`parser_failed`（500）/ `parser_unavailable`（503）。

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
- **预检失败**（MIME 不支持 / 文件超限 / 空文件 / profile 非法）发生在流开启之前，仍按普通
  JSON 错误信封返回（415 / 413 / 400，`Content-Type: application/json`），客户端需同时兼容两种响应。

curl 示例：

```bash
curl -N -F "file=@report.pdf" -F "profile=auto" http://localhost:8080/v1/parse/stream
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
  "strategy": "recursive",
  "options": {},
  "add_context": false,
  "chunk_size": 500,
  "chunk_overlap": 75,
  "page_numbers": null,
  "metadata": {
    "filename": "report.pdf",
    "title": "示例"
  }
}
```

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `markdown` | string | 必填 | 待分片的 Markdown 文本（至少 1 个字符） |
| `strategy` | string | `recursive` | `fixed` / `paragraph` / `recursive` / `semantic` / `llm`，详见[分片策略](#分片策略chunking-strategies) |
| `options` | object | `{}` | 策略参数：通用 `min_chunk_size`（默认 chunk_size 的 10%），以及各策略专属的 `breakpoint_percentile` / `protect_code` / `input_token_budget`；未知 key 会被忽略并记日志 |
| `add_context` | bool | `false` | 是否做 Anthropic contextual retrieval 式上下文增强（需配置 LLM），详见[上下文增强](#上下文增强contextual-retrieval) |
| `chunk_size` | int | 500 | 单 chunk 最大 token 数（cl100k_base） |
| `chunk_overlap` | int | 75 | 相邻 chunk 的重叠 token 数，必须 `< chunk_size` |
| `page_numbers` | array[int] \| null | null | 按源页给出的边界提示；缺省时按 markdown 中的页标记标注，均无则视为第 1 页 |
| `metadata` | object | — | 透传到每个 chunk 的字段；当前仅 `title` / `filename` / `page_count` 被消费为 chunk 内字段 |

请求体约束：
- `1 ≤ chunk_size ≤ 8192`、`0 ≤ chunk_overlap ≤ 4096`（pydantic 422）；
- `chunk_overlap ≥ chunk_size` → 400 `invalid_chunk_config`。

### 运行前依赖（pre-flight）

- `semantic` 需要已加载的 embedder（复用其 `embed_documents`，即 BGE-M3）；未加载 → **503 `embedder_unavailable`**。
- `llm` 与 `add_context` 需要配置外部 OpenAI 兼容 LLM（`VS_LLM__BASE_URL` + `VS_LLM__MODEL`）；
  未配置 → **503 `llm_unavailable`**。

### 响应

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "text": "# 标题\n\n第一段...",
      "token_count": 487,
      "section_header": "1. Intro > 1.1 Background",
      "page_number": 1,
      "context": null
    },
    {
      "chunk_index": 1,
      "text": "继续...",
      "token_count": 412,
      "section_header": "2. Details",
      "page_number": 2,
      "context": "本片段出自 2026 年季度报告的背景章节，讨论营收结构。"
    }
  ]
}
```

- `context` —— 仅 `add_context=true` 时可能非空：LLM 为该 chunk 生成的 1–2 句定位前缀。
  它只在后续嵌入时拼接到正文前，**不是 chunk 正文的一部分**。

错误码：请求体校验失败（422）/ `invalid_chunk_config`（400）/ `embedder_unavailable`（503）/
`llm_unavailable`（503）/ `chunk_failed`（500）。

---

## 分片策略（chunking strategies）

分片发生在 Docling 把文档转成 Markdown 之后、嵌入之前。五种策略统一由
`chunking/` 子系统的注册表 + factory 管理（`register_strategy` / `build_chunker`），
API 层只按名字构造，不直接依赖具体实现。

| 策略 | 切分依据 | 结构感知 | 额外成本 | 适合场景 |
|------|----------|----------|----------|----------|
| `fixed` | token 流硬窗口（可重叠） | 代码块保护（默认开） | 无 | 日志、流式文本、需要严格等长窗口的评测基线 |
| `paragraph` | 段落为原子单位贪心打包 | 标题分节 / 面包屑 / 页码 | 无 | 段落完整、格式规范的文章与文档（默认推荐起点） |
| `recursive` | header → 段落 → 句子 → 词 递归降级 | 同上，代码块原子 | 无 | 混合结构文档；行为最稳健，**服务默认策略** |
| `semantic` | 相邻句子嵌入距离的百分位断点 | 节内运行，保留面包屑 | 每篇一次额外的句子批量嵌入 | 主题边界明显、篇幅较长的文档 |
| `llm` | LLM 返回主题断点编号（LumberChunker 式） | 节内窗口化调用 | 每个分节窗口一次 LLM 调用，慢且有费用 | 高价值语料、单文档内精确定位要求高 |

### fixed

对 token 序列开滑窗：`step = chunk_size - chunk_overlap`，逐窗 encode→切片→decode，
任何文本都能严格按 token 数切开。默认 `options.protect_code=true`：fenced code block
（`` ``` ``）整体作为一个原子 chunk 输出，超长代码块只记 warning 不硬切；
置 `false` 则连代码一起按窗口切。末尾窗口短于 `min_chunk_size` 时直接并入前一窗口
（合并后的尾窗可能略微超过 chunk_size）。

### paragraph

先按 Markdown header 分节（header 路径作为 breadcrumb，首个 chunk 会带上字面标题前缀），
节内以空行拆段落，段落作为不可拆单元贪心打包到 `chunk_size`；放不下的段落先降级为句子、
再降级为词进行打包。支持 token overlap。与 recursive 的区别：不做跨层级递归，
优先保持段落完整，chunk 内文本更"整块"。

### recursive（默认）

按以下顺序递归尝试切分：

1. **Markdown header**（`#`–`######`）—— 顶层按 header 切分，header 路径作为 breadcrumb；
2. **段落**（双换行 `\n\n`）；
3. **句子**（句号 / 问号 / 感叹号 + 后置空白，含中文标点）；
4. **词**（兜底，确保即使无标点也能切）。

累积 tokens 达到 `chunk_size` 即输出当前 chunk，重叠区取上一个 chunk 末尾
`chunk_overlap` tokens；code block 视为原子单元。token 计数使用
`tiktoken.get_encoding("cl100k_base")`。

### semantic

LangChain `SemanticChunker` 的百分位断点法，节内运行：

1. 把节内文本拆成句子；
2. 用注入的 `embed_fn`（服务中即已加载的 BGE-M3）**一批**嵌入所有句子，无额外模型依赖；
3. 计算相邻句向量的 cosine 距离，取距离分布的第 `breakpoint_percentile` 百分位为阈值，
   **严格大于**阈值才切（均匀文本退化为整节一个组，不会误切）；
4. 各组句子按 `chunk_size` 打包，超限在句子边界硬切兜底。

选项：`options.breakpoint_percentile`（`(0, 100]`，默认 `95`——只在距离最大的 5% 处切；
`100` 等于不按语义切）。

### llm

LumberChunker 式 LLM 边界选择，节内运行：

1. 句子带编号，在 `input_token_budget`（默认 12000 tokens）预算内窗口化；
2. 每个窗口请求 LLM 只返回 JSON `{"break_after": [编号, ...]}`（解析失败时退化为正则抽数字，
   再失败则该窗口不切，**不阻断管线**）；
3. 按返回位置分组，组内按 `chunk_size` 打包（分隔符为空格）。

选项：`options.input_token_budget`（默认 12000，按所用模型的安全上下文调整）。

### 最小切片尺寸（min_chunk_size）

过小的切片（几个字 / 几个 token）携带的信息不足以构成有意义的检索命中。所有策略都接受
`options.min_chunk_size`：低于该 token 数的 chunk 会被并入相邻 chunk，而不是单独输出——

- 默认为 `chunk_size` 的 **10%**（如 chunk_size=500 → 50）；显式传 0 以上的值可覆盖；
- 前向合并：小 chunk 优先并入前一个 chunk（前提是合并后不超过 chunk_size），放不下则随
  下一个 chunk 到达时并入其后；
- 末尾小尾巴**无条件**并入前一个 chunk（宁可最后一个 chunk 略微超长，也不留下碎片）；
- `fixed` 策略在 token-id 区间层面合并尾窗口（不会因 overlap 产生重复文本）；
- semantic / llm 的边界是有意义的主题断点，但当断点两侧体量都低于下限时，检索价值仍然
  不足，故依旧合并——**下限优先于边界**。

实测（Docling 解析的 PDF，约 2.5 万 tokens，chunk_size=800 / overlap=80）：修复前 recursive
因把超大节逐段直出产生 **1463 个碎片（最小 1 token，1292 个不足 50 tokens）**；修复后
recursive 输出 32 个切片（最小 733 tokens），五种策略均无低于下限的切片。

### 选型建议（2025–2026 评测结论）

- 大规模 in-corpus 检索中，**结构型分片（paragraph/recursive）性价比最高**，"简单方法"仍然能打，
  应作为默认；
- **semantic** 只在主题切换明显的文档上稳定占优；均匀文本上它退化为尺寸打包，代价是一次额外嵌入；
- **llm** 对单文档内定位最好，但延迟与费用高，建议只用于高价值语料或离线精处理；
- **fixed** 主要作为基线与特殊文本（日志/代码）的选择。

参考：[Late Chunking (arXiv 2409.04701)](https://arxiv.org/abs/2409.04701)、
[Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)、
[Recursive Semantic Chunking (ICNLSP 2025)](https://preview.aclanthology.org/override-month/2025.icnlsp-1.15.pdf)、
[NVIDIA: Finding the best chunking strategy](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses)。

> **Late chunking**（先整体编码再按 span 池化）需要 token 级向量输出并改造嵌入阶段，
> 本轮未实现；注册表架构已为其预留扩展点（可新增不经过 `chunk()` 文本路径的策略）。

---

## 上下文增强（Contextual Retrieval）

`add_context=true` 时，分片完成后会按 [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
的做法，让 LLM 为**每个 chunk** 生成 1–2 句"这个片段在整篇文档中处于什么位置"的简短上下文：

- 使用 Anthropic cookbook 的精简 prompt（`<document>` / `<chunk>` 占位，要求只返回上下文、不复述片段）；
- 整篇文档按 `VS_LLM__MAX_CONCURRENCY`（默认 4）在有界线程池内并发请求；
- **单个 chunk 的请求失败 / 超时 / 返回空白只记日志，该 chunk 的 `context` 保持 `null`，绝不阻断摄取**；
- 嵌入文本 = `context + "\n\n" + text`（由 `embed_text()` 统一拼接）；
- **存入 Milvus 的 `text` 字段仍是原始 chunk 文本**，context 不入库、不回显给最终用户，
  Milvus schema 无需改动。

Anthropic 报告该技术（配合 reranking）可使 top-20 检索失败率降低约 35%；代价是每 chunk 一次
LLM 调用，建议与 `paragraph` / `recursive` 搭配用于高价值知识库。

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
strategy:      recursive           # 可选，fixed | paragraph | recursive | semantic | llm
chunk_options: {}                  # 可选，JSON 字符串（策略参数，同 /v1/chunk 的 options）
add_context:   false               # 可选，开启 LLM 上下文增强（详见上文）
chunk_size:    500                 # 可选，表单字段，默认 500
chunk_overlap: 75                  # 可选，默认 75
embed_model:   bge-m3              # 必填，必须已加载
profile:       auto                # 可选，auto | standard | native | vlm
metadata:      {"title": "...", "author": "...", "filename": "..."}   # 可选，JSON 字符串
```

`embed_model` 必须是已通过 `POST /v1/models/{id}/load` 加载的 embedder；不指定 `is_query`
（上传路径走文档嵌入，前缀由服务端控制）。

`doc_id` 在解析**之前**生成（UUID4），同时用作图片落盘的目录名：
`<artifacts_dir>/<doc_id>/images/`，markdown 中的图片 URL 即为
`/artifacts/<doc_id>/images/image_*.png`，因此入库后图片链接与文档长期对应。

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
- **预检失败**（参数非法 / strategy 或 chunk_options 非法 / MIME 不支持 / 文件超限 / profile 非法 /
  embedder 未加载 / LLM 策略未配置）发生在流开启之前，仍按普通 JSON 错误信封返回
  （4xx/503，`Content-Type: application/json`），客户端需同时兼容两种响应。
- 文档解析后未产生任何分片时，只发出 `parse`、`chunk` 两个 stage 事件，随后直接 `result`
  （`chunk_count: 0`），不会进入嵌入 / 写入；该文档的图片目录会一并清理。

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

### 原子回滚与产物清理

服务端用 try/except 包裹整条管道。一旦以下任一步失败，已成功 upsert 的向量会按
`filter_expr = 'doc_id == "{new_doc_id}"'` 删除，同时 best-effort 删除
`<artifacts_dir>/<doc_id>/` 整个图片目录（清理失败只记 warning，不影响回滚结果），
向上抛出错误，collection 与磁盘都不留半成品。

可能的失败点与对应错误码：

| 失败点 | 错误码 | HTTP |
|--------|--------|------|
| profile 非法（预检） | `invalid_profile` | 400 |
| strategy 非法（预检） | `invalid_strategy` | 400 |
| chunk_options 不是合法 JSON 对象（预检） | `invalid_chunk_options` | 400 |
| chunk_size / overlap 非法（预检） | `invalid_chunk_size` 等 | 400 |
| llm 策略 / add_context 但未配置 LLM（预检） | `llm_unavailable` | 503 |
| semantic 策略但 embedder 未加载（预检） | `embedder_unavailable` | 503 |
| 文件解析 | `parser_failed` | 500 |
| 分片（含零分片提前返回，仅清理产物） | `chunk_failed` | 500 |
| 嵌入（模型未加载 / 推理失败） | `embedder_unavailable` | 503 |
| Milvus 写入 | `store_unavailable` | 503 |
| 目标 collection 不存在 | `collection_not_found` | 404 |
| 数据库不存在 | `database_not_found` | 404 |
| embedder.dim 与 collection 不匹配 | `dimension_mismatch` | 422 |

---

## Dashboard 调试面板

"知识库" 导航组（侧栏底部）下四个面板：

| 面板 | 端点 | 功能 |
|------|------|------|
| 文档解析 | `POST /v1/parse/stream` | 解析方案下拉（自动按类型/标准/原生/VLM），选中即显示方案介绍 + multipart 上传（字节 %）+ 逐页解析进度条 → 统计区（字符/页数/图片/表格/OCR 页数/耗时）+ Markdown 预览（图片可点开） |
| 分片测试 | `POST /v1/chunk` | 策略下拉（5 种，semantic 显示百分位参数、显示上下文增强开关）+ Markdown 输入 → 可折叠 chunk 列表（含 token / 页码 / 章节 / 原文 / LLM 上下文前缀），不落库 |
| 一键入库 | `POST /v1/ingest/stream` | database/embed_model 联动选择 + 分片策略下拉（含策略参数与上下文增强）+ 解析方案下拉 + multipart 上传（支持图片文件），5 阶段步进器（parse 阶段显示 n/total 页） → 摄取结果统计 |
| 入库浏览 | `POST /v1/databases/{db}/collections/ingest/rows` | 分页浏览已入库分片，支持按 doc_id 精确匹配 / 文件名关键字过滤 |

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
  ├─ 2. build_chunker(strategy) 切 Markdown → chunks（默认 recursive；可选 LLM 上下文增强）
  ├─ 3. BGE-M3 嵌入每个 chunk（若有 context 则拼接后嵌入）→ 1024 维向量
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

### Docling converters：按 profile 缓存的进程级单例

`parsers/docling_parser.py::DoclingParser` 维护 `dict[profile, DocumentConverter]`，
在锁内双重检查懒加载：standard / native / vlm 各自只构建一次；`auto` 由
`_auto_profile(mime, probe)` 在触缓存前按文档类型解析（数字 PDF→native，其余→standard）。
format options 由 `_build_format_options(profile)` 生成：

- **standard**：PDF 与 IMAGE 均用 `StandardPdfPipeline`，`RapidOcrOptions(backend="onnxruntime",
  lang=VS_PARSER__OCR_LANGS, mode=OcrMode.DEFAULT)` —— 选择性 OCR，数字页不付 OCR 代价；
- **native**：PDF 用 `NativePdfFormatOption`（model-free），IMAGE 仍走 standard OCR；
- **vlm**：PDF 与 IMAGE 均用 `VlmPipeline`，`VlmConvertOptions.from_preset(VS_PARSER__VLM_PRESET)`。

DOCX / PPTX / HTML 未列入 format_options，由 DocumentConverter 的默认项自动补成
model-free `SimplePipeline`。GPU 设备经 `AcceleratorOptions(device=...)` 传入；
onnxruntime 的 CUDA DLL 由 `core/cuda_dlls.py` 在 Windows 上引导。

PDF 文字层探测（`parsers/pdf_probe.py`，PyMuPDF）只产出 `doc_kind` / `scanned_pages`
元数据，不决定是否 OCR —— 选择性 OCR 由 Docling 管线自身完成。

### 图片产物目录

布局固定为 `<artifacts_dir>/<stem>/images/image_NNNNNN_<hash>.png`：ingest 的 stem 即
`doc_id`（解析前生成），`/parse` 的 stem 为随机 hex（或调用方通过 `artifact_stem` 指定）。
图片经 StaticFiles 挂在 `/artifacts` 提供 HTTP 访问；无图片的文档会自动删除空目录。

### Chunk row 主键

`{doc_id}_{chunk_index}` 是确定性的 —— 同一文件重复 ingest 会覆盖而非重复入库。
删除时按 `doc_id` filter 一次性清空文档所有 chunk，原子性强。

### 可插拔分片子系统

分片器全部位于 `chunking/`：`base.py` 定义 `Chunk` / `Chunker` ABC 与名字注册表，
`tokens.py` 提供 tiktoken 计数/取尾，`structure.py` 提供共享的标题分节、页码标注、
代码块保护与打包原语；五种策略各自一个模块，经 `register_strategy` 自注册，
`factory.build_chunker()` 按名字 + `options` 构造。ingest 管线把分片工作整体放入
worker 线程（`_produce_chunks`），避免阻塞 NDJSON 事件刷新。

LLM 分片器与上下文增强共用一个进程级缓存的 OpenAI 兼容 httpx 客户端
（`OpenAIChatClient` → `{base_url}/chat/completions`），由 `VS_LLM__*` 配置；
服务自身不暴露 chat 接口。

### 与 Lumos 的集成模式

Lumos 是薄客户端：

- **不引入 Docling/PyTorch 依赖** —— 这些只在 vector-service 进程内加载。
- **不实现 chunker** —— 通过 `/v1/chunk` 调参即可。
- **不直接调 Milvus** —— 所有 collection / vectors 操作都走 vector-service。
- **持久化 KB 元数据** —— 知识库名、chunk 配置、原文落盘路径等放在 Lumos 的 SQLite。

Lumos 后端通过 httpx 调用 vector-service，超时 300s（`/v1/ingest` 首次可能更久）。
详见 lumos 项目的 `KnowledgeManager` + `VectorServiceClient`。
