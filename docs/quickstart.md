# 快速上手

5 分钟把 `vector-service` 跑起来。

> 详细配置、API、扩展说明都在 [README.md 索引](README.md) 列出。

## 0. 前置条件

- Python ≥ 3.11
- 一台可访问的 Milvus server（standalone / cluster），版本 ≥ 3.0（native database）
- 可选：GPU + CUDA（如需用 torch fp16 后端跑 BGE-M3；否则自动落到 ONNX int8 CPU）

## 1. 准备 `.env`

```bash
cp .env.example .env
```

至少确认：

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `VS_HOST` / `VS_PORT` | 监听地址 | `0.0.0.0` / `8080` |
| `VS_LOG_LEVEL` / `VS_LOG_FORMAT` | 日志级别 / 格式（`json` / `console`） | `INFO` / `json` |
| `VS_EMBEDDING_BACKEND` | 文本嵌入器后端 | `bge-m3` |
| `VS_EMBEDDING_MODEL_DIR` | 模型本地目录 | `./models/bge-m3` |
| `VS_EMBEDDING_AUTO_DOWNLOAD` | 首次启动自动下载模型权重 | `true` |
| `VS_VECTOR_STORE_BACKEND` | 向量库后端（目前只支持 `milvus`） | `milvus` |
| `VS_MILVUS_URI` | Milvus server 地址（gRPC） | `http://localhost:19530` |
| `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD` | 账号密码（启用 auth 时填） | （空） |
| `VS_MILVUS_TOKEN` | 可选 token（优先级高于 user/password） | （空） |
| `VS_MILVUS_TIMEOUT` | 连接 / RPC 超时（秒） | `30` |
| `VS_RERANKER__BACKEND` | reranker 后端 | `bge-reranker-v2-m3` |
| `VS_IMAGE_EMBEDDING__BACKEND` | 图像嵌入器后端 | `openclip-vit-l-14` |
| `VS_MULTIMODAL_EMBEDDING__BACKEND` | 跨模态嵌入器后端 | `chinese-clip-vit-base-patch16` |

完整字段见 [`.env.example`](../.env.example)。字段语义详见 [configuration.md](configuration.md)。

## 2. 启动 Milvus server

参考 [Milvus 官方文档](https://milvus.io/docs/install_standalone-docker.md) 起一个 standalone：

```bash
wget https://github.com/milvus-io/milvus/releases/download/v3.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

确认 `localhost:19530` 可达。

## 3. 安装依赖

```bash
uv sync --python 3.11
```

所有运行时依赖都是必装项（pymilvus、BGE-M3、open_clip_torch、transformers）。
在 win32/linux 上 torch 与 onnxruntime 自动使用官方 cu130 GPU wheel（见 `pyproject.toml` 的 `[tool.uv.sources]`），cuDNN 来自 torch/lib，不再需要手工安装 GPU torch；macOS 使用 CPU 版。
`dev` 依赖组（pytest / ruff / mypy）默认一并安装。

## 4. 启动服务

```bash
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
```

或直接装 console_script：

```bash
vector-service
```

## 5. 访问路径

| 路径 | 内容 |
|------|------|
| `http://127.0.0.1:8080/dashboard` | 自带调试页面（每个 tab 都能点按钮调接口） |
| `http://127.0.0.1:8080/docs` | Swagger UI |
| `http://127.0.0.1:8080/redoc` | ReDoc |
| `http://127.0.0.1:8080/openapi.json` | OpenAPI 文档 |
| `http://127.0.0.1:8080/healthz` | 进程存活 |
| `http://127.0.0.1:8080/readyz` | 嵌入器加载 + Milvus 可达 |
| `http://127.0.0.1:8080/metrics` | Prometheus |

![Dashboard 总览首页](dashboard-overview.png)
*Dashboard 默认落地页：KPI 总览 + 已加载能力与资源占用 + 文档解析引擎 + 向量库连接状态。*

![文本相似度调试面板](dashboard-text-similarity.png)
*「相似度」调试面板的文本模式（侧栏 模型 → 相似度）：选模型 + 度量，填查询与候选（每行一条），运行后按所选度量排序展示结果行。*

侧栏「运维」组提供四个运维操作面：

| 面板 | 对应端点 | 能做什么 |
|------|----------|----------|
| 索引重建 | `POST .../reindex`、`.../reindex/promote` | 从 SQLite 全量重算向量建 canary，跟踪重建任务与门禁检查，通过后一键提升 |
| 任务队列 | `GET /v1/jobs`、`/v1/jobs/{id}/events` | 按状态过滤、分页浏览全部任务；点开任务经 SSE 实时跟踪进度、请求取消 |
| 一致性校验 | `POST .../consistency` | 对比 SQLite 叶子与 Milvus 行，展示缺失 / 孤儿明细，一键修复（需 embedder 已加载） |
| 评测结果 | `/v1/evaluation/*` | 浏览评测集 / 问题 / 版本 / 运行，run 详情含聚合 KPI、rerank 前后对比与通道归因；查看门禁阈值与检查记录 |

## 下一步

- API 用法：[api.md](api.md)
- 嵌入/重排子系统详细参数与错误码：[embedding-subsystems.md](embedding-subsystems.md)
- 模型热加载与生命周期：[model-lifecycle.md](model-lifecycle.md)
- 知识库摄取管线（Docling → 分片 → 嵌入 → 写入）：[ingest-pipeline.md](ingest-pipeline.md)
- 跑测试：[testing.md](testing.md)
