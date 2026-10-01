# 架构定位

## 拓扑

```
client ──HTTP──▶ vector-service ──gRPC──▶ Milvus server
                  ├─ BGE-M3 文本嵌入
                  ├─ OpenCLIP 图像嵌入
                  ├─ Chinese-CLIP 跨模态嵌入
                  ├─ BGE-Reranker 重排序
                  ├─ Docling 文档解析
                  ├─ 递归 markdown 分片
                  └─ Milvus store 客户端（pymilvus 直连）
```

`vector-service` 自己的职责有五块：

1. **文本嵌入**：用 BGE-M3 把文本变成 `list[float]`。
2. **图像嵌入**：用 OpenCLIP 把 base64 图片变成 `list[float]`。
3. **跨模态嵌入**：用 Chinese-CLIP 同时支持中文文本与图片，输出同一空间向量。
4. **向量库 CRUD**：通过 `pymilvus` 直接对 Milvus 做 database / collection / 向量管理。
5. **知识库摄取管线**（Docling → 分片 → 嵌入 → 写入）：把任意文档一键转成可检索的 chunks。
   详见 [ingest-pipeline.md](ingest-pipeline.md)。

### 阻塞调用隔离

所有外部依赖都是同步阻塞接口，事件循环上绝不直接调用。阻塞调用按性质进入三个互相独立的有界线程池（`core/threadpools.py`，lifespan 装配）：

- **store 池** — Milvus RPC；
- **model 池** — 本地模型/CPU 推理（嵌入、rerank、BM25、解析、社区检测）；
- **sqlite 池** — 语料库读写（SQLite 是事实来源）。

每池独立 worker 数与准入信号量，超限排队产生背压，三类流量不会互相饿死。LLM 外呼与文件系统清理保留在 asyncio 默认 executor。配置见 [configuration.md](configuration.md#线程池隔离)。

### 语料存储

SQLite（`data/corpus/corpus.db`）是唯一事实来源；Milvus 只存向量与最少标量，BM25 统计与原文都是可重建/可重取的派生物。原始上传二进制**不入库**：摄取成功后 spool 原子提升为 SHA-256 内容寻址文件（`data/corpus/originals/<hash[:2]>/<hash>`），`documents.content_hash` 持有引用，相同内容全局去重。blob 删除分两层：删除路由按 repository 返回的孤立 hash 即时 GC，后台维护 worker（`jobs/maintenance.py`）定期兜底清扫；同一 worker 按空闲页比例阈值执行 VACUUM + WAL 截断，并在周期开头用 sqlite 在线 backup API 写语料库快照（保留最新 N 份；Milvus 不备份，可重建）。摄取/维护 worker 在进程内运行，同一语料目录只允许一个实例：启动时对 `instance.lock` 取 OS 级排他锁，第二个实例 fail fast；多实例需求出现时再将 repository 实现替换为 Postgres（接口不变）。配置见 [configuration.md](configuration.md#语料存储与维护)。

## 源码结构

```
vector-service/
├── pyproject.toml              # 项目元数据 + 依赖 + pytest 配置
├── README.md                   # 项目简介与入口
├── .env.example                # VS_* 环境变量示例
├── .gitignore
├── src/vector_service/
│   ├── main.py                 # FastAPI app 工厂 + 异常处理 + lifespan
│   ├── api/                    # FastAPI 路由
│   │   ├── health.py           # /healthz /readyz /metrics
│   │   ├── models.py           # /v1/models /v1/models/{id}
│   │   ├── embeddings.py       # /v1/embeddings
│   │   ├── image_embeddings.py # /v1/image_embeddings
│   │   ├── multimodal_embeddings.py # /v1/multimodal_embeddings
│   │   ├── rerank.py           # /v1/rerank
│   │   ├── management.py       # /v1/databases + /v1/databases/{db}/collections/*
│   │   ├── backend.py          # /backend/raw（运维诊断）
│   │   └── dashboard.py       # /dashboard 内嵌调试页面
│   ├── core/                   # 基础设施
│   │   ├── config.py           # pydantic-settings：VS_* + 子系统嵌套设置
│   │   ├── errors.py           # 异常体系（EmbedderError / StoreError / RerankerError / ...）
│   │   ├── lifespan.py         # 启动期加载模型、连接 Milvus
│   │   ├── logging.py          # structlog 结构化日志
│   │   ├── metrics.py          # Prometheus 指标
│   │   ├── middleware.py       # RequestID 中间件
│   │   └── model_lifecycle.py  # ModelSlot：每个模型族的常驻实例 + load/unload
│   ├── embeddings/             # 文本/图像/跨模态嵌入器抽象与实现
│   │   ├── base.py             # Embedder ABC
│   │   ├── bge_m3.py           # BGE-M3 文本嵌入
│   │   ├── image_base.py       # ImageEmbedder ABC
│   │   ├── image_decoding.py   # base64 + MIME 校验
│   │   ├── openclip_vit_l14.py # OpenCLIP 图像嵌入
│   │   ├── multimodal_base.py  # MultimodalEmbedder ABC
│   │   ├── chinese_clip_multimodal.py  # Chinese-CLIP 跨模态
│   │   ├── registry.py         # EMBEDDER_REGISTRY
│   │   ├── image_registry.py   # IMAGE_EMBEDDER_REGISTRY
│   │   └── multimodal_registry.py      # MULTIMODAL_EMBEDDER_REGISTRY
│   ├── rerankers/              # 重排序
│   │   ├── base.py             # Reranker ABC
│   │   ├── cross_encoder.py    # BGE-Reranker-v2-M3
│   │   └── registry.py         # RERANKER_REGISTRY
│   ├── parsers/                # 文档解析（Docling + passthrough）
│   │   ├── base.py             # DocumentParser ABC
│   │   ├── docling_parser.py   # Docling 单例：PDF/DOCX/PPTX/HTML
│   │   └── markdown_parser.py  # Markdown/Text passthrough
│   ├── chunking/               # 文本分片
│   │   └── recursive_chunker.py  # 递归 markdown-aware 分片器
│   ├── schemas/                # Pydantic 请求/响应模型
│   │   ├── openai.py           # Embedding / EmbeddingResponse / Model
│   │   ├── image_embeddings.py
│   │   ├── multimodal_embeddings.py
│   │   ├── rerank.py
│   │   ├── management.py       # database/collection/vector CRUD
│   │   ├── ingest.py           # shared pipeline core (/v1/parse · /v1/chunk)
│   │   ├── jobs.py             # /v1/jobs/ingest · /v1/jobs/{id}(/events)
│   │   └── errors.py           # ErrorEnvelope
│   └── stores/                 # 向量库抽象与实现
│       ├── base.py             # VectorStore ABC
│       ├── milvus.py           # MilvusStore（直连 pymilvus）
│       ├── _milvus_adapter.py  # pymilvus 适配层（内部实现细节）
│       └── registry.py         # build_store(settings)
├── tests/                      # 单元 + contract 测试
│   ├── unit/                   # 不依赖外部服务的快速测试
│   └── contract/               # 需要真实 Milvus / 模型（标记 contract）
├── docs/                       # 用户文档（本目录）
│   ├── README.md               # 文档目录索引
│   ├── quickstart.md
│   ├── architecture.md         # 本文件
│   ├── configuration.md
│   ├── api.md
│   ├── embedding-subsystems.md
│   ├── vector-store.md
│   ├── ingest-pipeline.md
│   ├── model-lifecycle.md
│   ├── errors.md
│   ├── testing.md
│   ├── extending.md
│   ├── dashboard-overview.png
│   └── dashboard-text-similarity.png
└── models/                     # 模型权重目录（运行时下载；已在 .gitignore）
```

## 扩展点

- **新增嵌入器**（文本 / 图像 / 跨模态）：在 `embeddings/` 下新建适配文件，并在对应的 `*_registry.py` 注册。详见 [extending.md](extending.md)。
- **新增 reranker**：在 `rerankers/` 下实现 `Reranker` ABC 并在 `RERANKER_REGISTRY` 注册。
- **新增向量库后端**：在 `stores/` 下实现 `VectorStore` 接口，并在 `stores/registry.py::build_store()` 加分支。
- **新增文档解析器**：在 `parsers/` 下实现 `DocumentParser.parse_bytes()`，在 `parsers/base.py` 注册。详见 [extending.md](extending.md)。
