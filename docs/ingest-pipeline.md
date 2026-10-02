# 摄取管线（Docling → Chunk → Embed → Milvus）

为知识库（RAG）场景设计的端到端文档摄取管线。Lumos 等上层应用通过 `POST /v1/jobs/ingest`
提交文件 + 目标 collection，后台 worker 即串起 **Docling 解析 → 文本分片（5 种可选策略）→
BGE-M3 嵌入 → Milvus 写入** 四个步骤，失败时按 `doc_id` 原子回滚。默认分片策略为
`recursive`（递归 markdown-aware）。任务进度可通过 `GET /v1/jobs/{id}/events` SSE 实时订阅。

> 该子系统与 [embedding-subsystems.md](embedding-subsystems.md) 共享已加载的 embedder；
> chunk 步骤独立于模型（纯文本操作），parser 步骤独立于模型（Docling 是文档转换库）。

---

## 端点速查

| 方法 | 路径 | 用途 |
|------|------|------|
| `POST` | `/v1/parse` | 文件 → Markdown + 元数据 |
| `POST` | `/v1/parse/stream` | 同上，但返回 NDJSON 事件流（上传 % + 逐页解析进度） |
| `POST` | `/v1/chunk` | Markdown → chunks（含 token 数 / 章节 / 页码） |
| `POST` | `/v1/jobs/ingest` | 提交异步摄取任务（multipart）：校验 + spool 字节，`202` 返回 `job_id` |
| `GET` | `/v1/jobs` · `/v1/jobs/{id}` | 任务列表 / 单个任务状态 |
| `GET` | `/v1/jobs/{id}/events` | SSE 进度推送（见下文） |
| `POST` | `/v1/jobs/{id}/cancel` | 请求取消任务 |

摄取请求接收 `multipart/form-data`（`/v1/chunk` 是 `application/json`）。文档处理完全异步：
请求只做预检与落 spool，解析/分片/嵌入/写入由进程内后台 worker 执行。

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
  "add_summary": false,
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
| `add_summary` | bool | `false` | 是否让 LLM 为每个分片生成摘要（独立存储、独立嵌入，检索时新增摘要召回路，需配置 LLM），详见[LLM 摘要](#llm-摘要summary) |
| `chunk_size` | int | 500 | 单 chunk 最大 token 数（cl100k_base） |
| `chunk_overlap` | int | 75 | 相邻 chunk 的重叠 token 数，必须 `< chunk_size` |
| `page_numbers` | array[int] \| null | null | 按源页给出的边界提示；缺省时按 markdown 中的页标记标注，均无则视为第 1 页 |
| `metadata` | object | — | 透传到每个 chunk 的字段；当前仅 `title` / `filename` / `page_count` 被消费为 chunk 内字段 |

请求体约束：
- `1 ≤ chunk_size ≤ 8192`、`0 ≤ chunk_overlap ≤ 4096`（pydantic 422）；
- `chunk_overlap ≥ chunk_size` → 400 `invalid_chunk_config`。

### 运行前依赖（pre-flight）

- `semantic` 需要已加载的 embedder（复用其 `embed_documents`，即 BGE-M3）；未加载 → **503 `embedder_unavailable`**。
- `llm`、`add_context` 与 `add_summary` 需要配置外部 OpenAI 兼容 LLM（`VS_LLM__BASE_URL` + `VS_LLM__MODEL`）；
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
      "context": null,
      "summary": null,
      "level": "chunk",
      "char_start": 0,
      "char_end": 512
    },
    {
      "chunk_index": 1,
      "text": "继续...",
      "token_count": 412,
      "section_header": "2. Details",
      "page_number": 2,
      "context": "本片段出自 2026 年季度报告的背景章节，讨论营收结构。",
      "summary": "2026 年季度报告背景章节：营收结构及变化原因。"
    }
  ]
}
```

- `context` —— 仅 `add_context=true` 时可能非空：LLM 为该 chunk 生成的 1–2 句定位前缀。
  它只在后续嵌入时拼接到正文前，**不是 chunk 正文的一部分**。
- `summary` —— 仅 `add_summary=true` 时可能非空：LLM 为该 chunk 生成的
  事实性摘要。它是**独立存储的产物**，有自己的嵌入与检索召回路，与只用于
  嵌入拼接的 `context` 不同。单个分片摘要失败时为 `null`。
- `level` / `char_start` / `char_end` —— `/v1/chunk` 只返回叶子，故
  `level` 恒为 `"chunk"`；后两者为该叶子在输入 Markdown 中的字符区间
  （层级结构中的父行只在摄取时产生，详见[层级切片](#层级切片small-to-large)）。

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

## 层级切片（small-to-large）

chunker 产出的每个 chunk 是**叶子**；摄取时由 `chunking/hierarchy.py`
包成三级结构一并写入 `chunks` 表：

| level | key（文档局部链接） | 内容 | 是否进 ANN |
|-------|--------------------|------|-----------|
| `document` | `"document"` | 整篇解析后 Markdown（1 行，根） | 否 |
| `section` | `"section:{ord}"` | 每个非空分节：字面 header 前缀 + 节正文 | 否 |
| `chunk` | `"chunk:{i}"` | chunker 叶子，源顺序 | 是 |

- 叶子只在节内产生——chunk 绝不跨节（节边界强制 flush，
  `merge_small_chunks` 也禁止跨 `section_ord`），所以每个叶子有且只有
  一个父节；无 header 的结构块同样有 ordinal，也会成为父行。
- 持久化时局部 key 经 `id_for_key` 解析为真实 `chunk_id =
  f"{doc_id}_{flat_index}"`，`parent_key` 解析为父行 `chunk_id`
  （写入 `parent_id`）；扁平顺序为 document → sections → leaves，
  `chunk_index` 跨整列稠密编号。
- `char_start` / `char_end` 记录该行在解析 Markdown 中的字符区间：
  section 的坐标在分节时精确计算（header 起、raw body 止），叶子的
  span 是其打包 units 的区间并集，支持检索结果回链高亮。
- **父子索引分离**：只有 level=`chunk` 的行进入 Milvus upsert，
  BM25 统计（`leaf_chunk_texts`）也只拟合叶子语料——父级文本不会
  重复计数、不会污染召回；父级是纯内容行，丢失后可随时从... （父级
  本身即 corpus 内容，Milvus 整体可重建）。

检索时如何上扩到父级见
[retrieval.md § Small-to-large 扩展](retrieval.md#small-to-large-扩展parent-document-retrieval)。

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

## LLM 摘要（summary）

`add_summary=true` 时，分片完成后让 LLM 为**每个 chunk** 生成一段事实性
摘要：

- prompt 携带 chunk 所属章节（`section_header`，缺失时省略）与 chunk 原文，
  要求只输出精炼摘要；
- 按 `VS_LLM__MAX_CONCURRENCY`（默认 4）在有界线程池内并发请求；
- **单个 chunk 的请求失败 / 超时 / 返回空白只记日志（`summarize_call_failed`），
  该 chunk 的 `summary` 保持 `null`，绝不阻断摄取**；
- 摘要入库为 `summary` 字段，并再做一次批量嵌入写入
  `summary_vector`；**摘要失败的 chunk 用原文嵌入兜底**，保证每行都有
  摘要向量（Milvus 向量字段不可空、无默认值）；
- 与 `add_context` 可同时开启，两者互不影响：context 只拼接到正文嵌入前、
  不入库；summary 独立存储、独立成路。

检索侧行为（摘要 ANN 召回路、权重语义、缺字段拒绝）详见
[retrieval.md § 摘要召回路](retrieval.md#摘要召回路)。

---

## 异步任务：`POST /v1/jobs/ingest`

文档摄取完全异步：提交请求只执行预检并把上传字节落 spool，立即返回 `202`；
解析 → 分片 → 嵌入 → 写入由进程内后台 worker 执行。**任何步骤失败都会原子回滚** ——
已写入的 corpus 行与已 upsert 的向量会按 `doc_id` 删除，避免半成品数据污染。

### 提交请求

```
POST /v1/jobs/ingest
Content-Type: multipart/form-data

file:          <binary>            # 必填，表单字段
database:      lumos               # 表单字段；不存在时按固定 ingest schema 自动创建
collection:    kb_a1b2c3d4e5f6    # 表单字段；不存在时自动创建，已存在则校验 dim 必须匹配 embedder.dim
strategy:      recursive           # 可选，fixed | paragraph | recursive | semantic | llm
chunk_options: {}                  # 可选，JSON 字符串（策略参数，同 /v1/chunk 的 options）
add_context:   false               # 可选，开启 LLM 上下文增强（详见上文）
add_summary:   false               # 可选，开启 LLM 摘要（检索新增摘要召回路）
chunk_size:    500                 # 可选，表单字段，默认 500
chunk_overlap: 75                  # 可选，默认 75
embed_model:   bge-m3              # 提交时不要求已加载；worker 执行时检查
profile:       auto                # 可选，auto | standard | native | vlm
metadata:      {"title": "...", "author": "...", "filename": "..."}   # 可选，JSON 字符串
```

提交成功响应（202）：

```json
{"job_id": "a3f9c2e1…", "status": "queued"}
```

`doc_id` 在提交时生成（UUID4），同时用作图片落盘的目录名：
`<artifacts_dir>/<doc_id>/images/`，markdown 中的图片 URL 即为
`/artifacts/<doc_id>/images/image_*.png`，因此入库后图片链接与文档长期对应。

### worker：执行、重试、取消

- 单进程后台 worker 逐个消费 queued 任务，任务阶段（`parsing` / `chunking` / `embedding` /
  `upserting`）与进度落 `ingest_jobs` 行。
- **重试**：4xx（参数/状态错误）立即终态失败；5xx（parser/store/embedder 不可用）按指数退避
  重排（`not_before_ts`），直到 `max_attempts`（默认 3）耗尽。
- **取消**：`POST /v1/jobs/{id}/cancel` 置 cancel 标记，worker 在阶段边界观察后回滚并置
  `cancelled`；终态任务返回 409。
- **启动恢复**：进程重启时，lifespan 在 worker 启动前把遗留的阶段状态任务逐行处理——清理
  半成品（corpus / 薄索引 / 产物）后 attempts+1 重排；spool 已丢失 → `failed/interrupted`；
  遗留 cancel 标记 → `cancelled`。

### 状态查询：`GET /v1/jobs/{id}`

```json
{
  "job_id": "a3f9c2e1…",
  "doc_id": "…",
  "status": "embedding",          // queued | parsing | … | done | failed | cancelled
  "stage": "embedding",           // 运行中与 status 相同；queued/终态为 null
  "database": "lumos",
  "collection": "kb_a1b2c3d4e5f6",
  "filename": "doc.pdf",
  "mime": "application/pdf",
  "attempts": 1,
  "max_attempts": 3,
  "cancel_requested": false,
  "progress": {"current": 3, "total": 12},
  "chunk_count": 0,
  "page_count": null,
  "tokens_used": 0,
  "error": null,
  "created_ts": 1769…,
  "updated_ts": 1769…,
  "finished_ts": null
}
```

终态 `done` 时 `chunk_count` / `page_count` / `tokens_used` 为最终统计；`failed` 时
`error = {"code", "message"}`。

### SSE 进度推送：`GET /v1/jobs/{id}/events`

`text/event-stream` 长连接，适合 dashboard / 前端实时展示，无需轮询：

```
event: job
data: {"job_id":"…","status":"queued", … }

event: job
data: {"job_id":"…","status":"parsing","stage":"parsing", … }
```

- **连接即发**当前完整状态（与 `GET /v1/jobs/{id}` 同形），之后**每次变更**推送一帧：
  阶段切换、parse 逐页 progress tick（仅 PDF/DOCX/PPTX/HTML）、重试重排、cancel 标记、终态。
- 帧数据始终是**完整 JobStatus**，客户端整体替换即可，天然幂等；重复 nudge 按行签名去重。
- **心跳**：`VS_JOBS__HEARTBEAT_SECONDS`（默认 15s）发送 `: keepalive` 注释帧，防止代理断连。
- **兜底 resync**：`VS_JOBS__SSE_RESYNC_SECONDS`（默认 2s）即使没有 nudge 也重读一次行，
  丢失/满队列 nudge 不可能让视图长期过期。
- 任务不存在 → 连接开启前直接 404 `job_not_found`（普通 JSON 信封）。
- **终态帧（done/failed/cancelled）发送后流立即关闭**；客户端按需重连只会重复收到终态快照。

浏览器消费：

```js
const es = new EventSource(`/v1/jobs/${jobId}/events`);
es.addEventListener("job", (e) => renderJob(JSON.parse(e.data)));
```

预检失败（参数非法 / strategy 或 chunk_options 非法 / MIME 不支持 / 文件超限 / profile 非法 /
LLM 策略未配置）发生在 spool 之前，按普通 JSON 错误信封返回（4xx/503）。

### 写入约定：SQLite 事实来源 + Milvus 薄索引

**内容先落 SQLite，向量再写 Milvus。** 文档与全部 chunk 在单个事务内
写入 corpus 后，才开始编码与 upsert——Milvus 的每一行都是 corpus 中某条
chunk 的派生物。整体定位见 [architecture.md](architecture.md)。

SQLite（`data/corpus/corpus.db`）：

| 表 | 内容 |
|----|------|
| `documents` | `doc_id`、database/collection、filename、mime、content_hash、title、author、page_count、status |
| `chunks` | `chunk_id`、`doc_id`、`chunk_index`、`text`、`section_header`、`page_number`、`token_count`、`summary`，以及层级列 `parent_id` / `level`（`document` / `section` / `chunk` 三级）/ `char_start` / `char_end`，见[层级切片](#层级切片small-to-large) |
| `chunk_indexes` | 每个 chunk 的派生索引登记：`index_kind`（dense/sparse/summary）、`model`、`index_ref` |
| `ingest_jobs` | 本次摄取任务的状态与统计 |

目标 Milvus collection 不存在时按下列**薄 schema** 自动创建（database 同
理）；已存在则只校验向量维度：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `varchar` PK | 形如 `{doc_id}_{chunk_index}` 的确定性主键（重试不会产生重复） |
| `doc_id` | `varchar` | 用于过滤 / 回滚 |
| `chunk_index` | `int64` | 文档内 chunk 序号 |
| `vector` | `float_vector` | 正文向量，维度 = embedder.dim（如 BGE-M3 = 1024），HNSW cosine |
| `sparse` | `sparse_float_vector` | 客户端 BM25 编码（jieba），`SPARSE_INVERTED_INDEX` / IP |
| `summary_vector` | `float_vector` | 摘要向量，HNSW cosine；无摘要时用 chunk 原文嵌入兜底 |

Milvus **不存** text / summary / 章节路径 / 文档元数据——这些只在 SQLite，
检索命中后经 hydrate 阶段组装（见 [retrieval.md](retrieval.md)）。
**只有叶子行（level=`chunk`）进入 Milvus**：document / section 父级是
SQLite 中的纯内容行，upsert 与 BM25 拟合都跳过它们（父子索引分离）。

`summary_vector` 每次入库都会写入：即使未带 `add_summary`，也会嵌入摘要
向量（未配置 LLM 时全部走原文兜底，不报错），只是 SQLite 中的 `summary`
列为空。

> Lumos 端会按此 schema 自动创建 collection（参见 lumos 项目的 `KnowledgeBaseConfig` + `VectorServiceClient.ensure_collection`）。
> 直接调用 `vector-service` 的运维可手动 `POST /v1/databases/{db}/collections` 创建。

### 原子回滚与产物清理

服务端用 try/except 包裹整条管道。一旦以下任一步失败，且文档已写入 corpus：
先按 `doc_id` 删除 SQLite 全部行（documents/chunks/chunk_indexes），再按
`filter_expr = 'doc_id == "{new_doc_id}"'` best-effort 删除已 upsert 的
Milvus 行，同时 best-effort 删除 `<artifacts_dir>/<doc_id>/` 整个图片目录
（清理失败只记 warning，不影响回滚结果）。向上抛出错误，corpus、collection
与磁盘都不留半成品。

可能的失败点与对应错误码：

| 失败点 | 错误码 | HTTP |
|--------|--------|------|
| profile 非法（预检） | `invalid_profile` | 400 |
| strategy 非法（预检） | `invalid_strategy` | 400 |
| chunk_options 不是合法 JSON 对象（预检） | `invalid_chunk_options` | 400 |
| chunk_size / overlap 非法（预检） | `invalid_chunk_size` 等 | 400 |
| llm 策略 / add_context / add_summary 但未配置 LLM（预检） | `llm_unavailable` | 503 |
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
| 分片测试 | `POST /v1/chunk` | 策略下拉（5 种，semantic 显示百分位参数、显示上下文/摘要增强开关）+ Markdown 输入 → 可折叠 chunk 列表（含 token / 页码 / 章节 / 原文 / LLM 上下文前缀与摘要），不落库 |
| 一键入库 | `POST /v1/jobs/ingest` + `GET /v1/jobs/{id}/events` | database/embed_model 联动选择 + 分片策略下拉（含策略参数与上下文/摘要增强）+ 解析方案下拉 + multipart 上传（支持图片文件，字节 %），提交后订阅 SSE `job` 帧渲染阶段进度 → 摄取结果统计；取消走 `POST /v1/jobs/{id}/cancel`（协作式，worker 在阶段边界停止） |
| 入库浏览 | `POST /v1/databases/{db}/collections/ingest/rows` | 分页浏览已入库分片，支持按 doc_id 精确匹配 / 文件名关键字过滤 |

---

## 典型工作流（Lumos 知识库场景）

```
Lumos 用户上传 PDF
  │
  ▼ Lumos 后端调 POST /v1/jobs/ingest（202 → job_id），订阅 SSE
  │
  ▼ vector-service 后台 worker：
  ├─ 1. Docling 解析 PDF → Markdown
  ├─ 2. build_chunker(strategy) 切 Markdown → chunks（默认 recursive；可选 LLM 上下文增强 / LLM 摘要）
  ├─ 3. BGE-M3 嵌入每个 chunk（若有 context 则拼接后嵌入）→ 1024 维向量；
  │      开启摘要时再批量嵌入摘要（缺失摘要走原文兜底）→ summary_vector
  └─ 4. Milvus upsert 到 collection kb_xxx（doc_id 前缀，含正文/摘要双向量）
      │
      ▼ 失败任一步：原子回滚（4xx 立即 failed，5xx 退避重排），SSE 推送阶段/终态
      ▼ 成功：GET /v1/jobs/{id} 终态为 {doc_id, chunk_count, page_count, tokens_used}
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

Lumos 后端通过 httpx 调用 vector-service 提交任务（`/v1/jobs/ingest`，立即 202 返回），
执行耗时（大文件首次解析可能较久）通过 SSE / 状态查询观察，不需要长超时阻塞请求。
详见 lumos 项目的 `KnowledgeManager` + `VectorServiceClient`。
