# 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件。pydantic-settings 使用 `__`（双下划线）作为嵌套分隔符，因此子系统的字段形如 `VS_RERANKER__BACKEND`。

字段全集见 [`.env.example`](../.env.example)。

## 子系统配置块

| 前缀 | 适用子系统 |
|------|------------|
| `VS_EMBEDDING_*` | 文本嵌入（BGE-M3） |
| `VS_IMAGE_EMBEDDING__*` | 图像嵌入（OpenCLIP） |
| `VS_MULTIMODAL_EMBEDDING__*` | 跨模态嵌入（Chinese-CLIP） |
| `VS_RERANKER__*` | 重排序（BGE-Reranker-v2-M3） |
| `VS_PARSER__*` | 文档解析（Docling） |
| `VS_CHUNKING__*` | 文本分片（RecursiveChunker） |
| `VS_RUNTIME__*` | 阻塞调用线程池隔离（store / model / sqlite） |
| `VS_MAINTENANCE__*` | 存储维护（VACUUM / blob 清扫） |
| `VS_BACKUP__*` | 语料库定时备份（sqlite 在线快照） |
| `VS_TRACING__*` | OpenTelemetry 分布式追踪 |
| `VS_MILVUS_*` | 向量库连接 |
| `VS_VECTOR_STORE_BACKEND` | 向量库后端选择（目前只支持 `milvus`） |

## 解析器与分片器

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_PARSER__AUTO_LOAD` | `false` | 启动时自动加载 Docling converter |
| `VS_PARSER__AUTO_DOWNLOAD` | `true` | 首次使用时按需下载 Docling 模型 |
| `VS_PARSER__MAX_FILE_SIZE_MB` | `100` | `/v1/parse` 与 `/v1/jobs/ingest` 单文件上限 |
| `VS_CHUNKING__DEFAULT_CHUNK_SIZE` | `500` | 默认 chunk token 数 |
| `VS_CHUNKING__DEFAULT_CHUNK_OVERLAP` | `75` | 默认 chunk 重叠 token 数 |

> Docling 体积较大（PyTorch + 模型权重），建议按需懒加载（`AUTO_LOAD=false`），首次调用 `/v1/parse` 时再加载。

## 关键开关速查

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_HOST` / `VS_PORT` | `0.0.0.0` / `8080` | 监听地址 |
| `VS_LOG_LEVEL` / `VS_LOG_FORMAT` | `INFO` / `json` | 日志级别 / 格式（`json` 或 `console`） |
| `VS_DEBUG` | `false` | 是否开放 `/backend/raw/call` debug 透传 |
| `VS_EMBEDDING_AUTO_LOAD` | `false` | 进程启动时是否自动加载文本嵌入器 |
| `VS_IMAGE_EMBEDDING__AUTO_LOAD` | `false` | 进程启动时是否自动加载图像嵌入器 |
| `VS_MULTIMODAL_EMBEDDING__AUTO_LOAD` | `false` | 进程启动时是否自动加载跨模态嵌入器 |
| `VS_RERANKER__AUTO_LOAD` | `false` | 进程启动时是否自动加载 reranker |
| `VS_PARSER__AUTO_LOAD` | `false` | 进程启动时是否自动加载 Docling converter |
| `VS_EMBEDDING_AUTO_DOWNLOAD` | `true` | 首次启动自动下载模型权重 |
| `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 OpenCLIP 权重 |
| `VS_MULTIMODAL_EMBEDDING__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 Chinese-CLIP 权重 |
| `VS_RERANKER__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 BGE-Reranker 权重 |
| `VS_PARSER__AUTO_DOWNLOAD` | `true` | 首次使用 Docling 时按需下载 |

> **生产默认行为**：所有模型族 `AUTO_LOAD=false`，启动时不构造、不加载。详见 [model-lifecycle.md](model-lifecycle.md)。

## 线程池隔离

所有阻塞调用按性质进入三个互相独立的线程池，避免 Milvus RPC、本地模型推理、SQLite 读写在同一个默认 executor 里相互饿死。每个池有独立 worker 数与准入上限（运行中 + 已排队的提交数），超限时调用方在信号量处排队等待（背压），而不是堆积无界任务。

| 池 | 承载调用 |
|----|----------|
| `store` | 向量库 RPC（search / upsert / delete / collection 管理等网络等待） |
| `model` | 本地 CPU 推理：文本/图像嵌入、rerank、BM25 拟合、Docling 解析、层级构建、社区检测 |
| `sqlite` | 语料库（CorpusRepository）全部读写 |

LLM 外呼（OpenAI 兼容接口）与文件系统操作（artifact / spool / originals）仍走 asyncio 默认 executor：它们以网络/IO 等待为主，不占隔离池名额。

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_RUNTIME__STORE_WORKERS` | `8` | store 池线程数（1–64） |
| `VS_RUNTIME__STORE_MAX_PENDING` | `64` | store 池准入上限（1–4096，须 ≥ workers） |
| `VS_RUNTIME__MODEL_WORKERS` | `2` | model 池线程数 |
| `VS_RUNTIME__MODEL_MAX_PENDING` | `16` | model 池准入上限 |
| `VS_RUNTIME__SQLITE_WORKERS` | `4` | sqlite 池线程数 |
| `VS_RUNTIME__SQLITE_MAX_PENDING` | `32` | sqlite 池准入上限 |

> 未装配线程池时（单元测试、独立脚本），`run_in_store` / `run_in_model` / `run_in_sqlite` 自动回退到 `asyncio.to_thread`。实时计数器（`inflight` / `waiting` / `scheduled`）见 `GET /v1/system/status` 的 `thread_pools` 字段。

## 语料存储与维护

原始上传二进制**从不入库**：语料库只存文本切片与元数据，原文以 SHA-256 内容寻址存盘，`documents.content_hash` 作为引用——磁盘路径可由哈希直接推出（`<originals_dir>/<hash[:2]>/<hash>`），无需额外列。相同内容的上传共享一个 blob；最后一个引用该哈希的文档被删除时，blob 随之删除。

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_CORPUS_DB_PATH` | `./data/corpus/corpus.db` | SQLite 语料库（事实来源） |
| `VS_BM25_STATE_DIR` | `./data/corpus/bm25` | BM25 统计派生物 |
| `VS_ORIGINALS_DIR` | `./data/corpus/originals` | 原文内容寻址仓 |
| `VS_INSTANCE_LOCK_PATH` | `./data/corpus/instance.lock` | 进程级排他锁（单实例闸） |

后台维护 worker（`jobs/maintenance.py`）按固定间隔：当数据库空闲页比例达到阈值时执行 `VACUUM` 并以 `wal_checkpoint(TRUNCATE)` 截断 WAL；随后扫描原文仓，删除无文档引用的 blob（删除路由即时 GC 的崩溃兜底）。

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_MAINTENANCE__ENABLED` | `true` | 启停维护 worker |
| `VS_MAINTENANCE__INTERVAL_SECONDS` | `86400` | 维护周期（60–604800） |
| `VS_MAINTENANCE__VACUUM_MIN_FREE_RATIO` | `0.2` | 空闲页占比达到该值才 VACUUM |
| `VS_MAINTENANCE__BLOB_SWEEP_ENABLED` | `true` | 是否清扫无引用 blob |

> VACUUM 与维护查询走 sqlite 池；blob 的文件操作走默认 executor。VACUUM 失败只记录日志，下一周期重试。

### 备份

备份只针对事实来源：维护周期先于 VACUUM 用 **sqlite 在线 backup API** 写出一份活库一致快照（写入期间安全，产物是单个可独立打开的 `.db` 文件——无需同时拷贝 WAL），再按数量修剪旧文件。默认快照目录 `data/backups/corpus-<UTC时间戳>.db`。Milvus 向量是可重建的派生索引，**不备份**。

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_BACKUP__ENABLED` | `true` | 是否在维护周期写快照 |
| `VS_BACKUP__DIR` | `./data/backups` | 快照目录 |
| `VS_BACKUP__RETAIN` | `7` | 保留最新快照数（1–365） |

> 快照恢复即把该 `.db` 拷回 `VS_CORPUS_DB_PATH`（进程停止状态下），随后按需重建 Milvus / BM25 派生物。需要持续归档（异地、分钟级）时可再引入 LiteStream；当前内置快照覆盖"定期备份"需求。

### 单实例约束

摄取/维护 worker 在进程内运行，SQLite 不提供跨进程行锁，因此**同一语料目录只允许一个服务实例**。启动时进程对 `VS_INSTANCE_LOCK_PATH` 取 OS 级字节范围排他锁（Windows `msvcrt` / POSIX `fcntl`）：第二个实例取不到锁即 fail fast，uvicorn 直接退出。锁由 OS 持有，进程崩溃后自动释放，不存在陈旧锁文件清理问题。同一主机跑多个独立语料库时，让各实例指向不同的语料目录（或至少不同 lock 路径）即可。

> 需要水平扩展时再迁移：repository 接口保持不变，只把实现从 SQLite 换成 Postgres（WAL / 连接池 / 行锁），届时移除该锁。在此之前不支持多实例。

## 分布式追踪

OpenTelemetry，默认关闭。启用后一次 `POST /v1/retrieval` 产生一棵 span 树：根 span `retrieval` 记录 database / collection / query 长度 / top_k，下挂各阶段 span——`rewrite`（含 HyDE 嵌入）、`recall`（每通道腿各一个 `recall.dense` / `recall.bm25` / `recall.summary` 子 span，带 query_length 属性）、`fuse`（method / chunks）、`hydrate`、`mmr`、`rerank`（model / candidates / input_chars）以及按需的 `graph` / `compress` / `expand`。任何阶段抛错都会记录 exception 事件并把根 span 置为 ERROR。

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_TRACING__ENABLED` | `false` | 安装 SDK tracer provider |
| `VS_TRACING__SERVICE_NAME` | `vector-service` | Resource `service.name` |
| `VS_TRACING__ENDPOINT` | _空_ | OTLP/HTTP traces 完整 URL，如 `http://otel-collector:4318/v1/traces`；留空则在进程内记录但不导出 |
| `VS_TRACING__SAMPLE_RATIO` | `1.0` | 采样率（ParentBased + TraceIdRatioBased） |

> 导出走 OTLP/HTTP + 批量 span processor；关闭时管线 span 命中默认 proxy，为零成本 no-op。上游服务传入 W3C `traceparent` 请求头时，检索 span 自动接进同一条 trace（Jaeger / Tempo / SigNoz 等后端均可）。

## Milvus 连接

`VS_MILVUS_TOKEN` 优先级高于 `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD`。`VS_MILVUS_TIMEOUT` 同时作用于连接与 RPC。完整的 Milvus 配置语义见 [vector-store.md](vector-store.md)。
