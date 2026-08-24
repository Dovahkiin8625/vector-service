# vector-service

生产级 FastAPI 向量服务：

- **OpenAI 兼容嵌入 API**：`/v1/embeddings`、`/v1/models`
- **可扩展向量库管理**：`/collections`、`/collections/{name}/vectors`、`/collections/{name}/search`
- **可观测性**：`/metrics`（Prometheus）+ 结构化 JSON 日志
- **可扩展抽象**：新增嵌入器/向量库只需新增适配文件并注册

## 第一阶段支持

| 组件 | 实现 |
|------|------|
| Embedder | BGE-M3 (GPU: torch fp16 / CPU: ONNX int8) |
| VectorStore | Milvus Lite |

## 快速开始

### 安装

```bash
pip install -e ".[embed,store,dev]"
cp .env.example .env
# 编辑 .env，至少设置 VS_MILVUS_URI 与 VS_EMBEDDING_MODEL_DIR
```

### 启动

```bash
# 首次启动会自动从 HuggingFace 下载 BGE-M3（若模型不存在）
make run
# 或
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
```

### 调用

```bash
# 1. 嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'

# 2. 创建集合
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "products"}'

# 3. upsert（用文本，自动嵌入）
curl -X PUT http://localhost:8080/collections/products/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": ["1"], "texts": ["a wireless mouse"], "metadatas": [{"price": 29.9}]}'

# 4. 检索
curl -X POST http://localhost:8080/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "computer accessory", "top_k": 5}'
```

### 健康检查

```bash
curl http://localhost:8080/healthz   # 进程存活
curl http://localhost:8080/readyz    # 模型 + store 就绪
curl http://localhost:8080/metrics   # Prometheus
```

## 测试

```bash
make test              # 全部（默认排除 slow）
make test-unit         # 仅单元
make test-contract     # 仅契约
make test-integration  # 集成（不含 slow）
make test-slow         # 真实 BGE-M3 + Milvus Lite（需要可选依赖）
```

> **Windows 注意**：`milvus-lite` 没有 Windows wheel。Milvus 相关契约测试
> 在 Windows 上自动跳过。要运行完整 Milvus 测试，请使用 WSL、Docker 或 Linux。

## 环境管理（uv）

```bash
# 一次性同步依赖（含 embed、store、dev 全套可选依赖）
uv venv --python 3.11                # 在 .venv 创建虚拟环境
uv pip install -e ".[embed,store]"    # 安装项目 + 可选依赖
uv pip install --group dev            # 安装开发依赖（pytest/ruff/mypy 等）

# 运行测试 / smoke
uv run pytest -q                      # 全部测试（默认排除 slow）
uv run pytest -m slow                 # 跑真实模型集成测试
uv run python scripts/smoke.py        # 端到端 smoke
```

## 添加新嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `Embedder` ABC
2. 在 `embeddings/registry.py` 注册
3. 加配置项到 `core/config.py`
4. 写契约测试 `tests/contract/test_embedder_contract.py`

## 添加新向量库

参见 `src/vector_service/stores/`：

1. 新建 `stores/<backend>.py`，实现 `VectorStore` ABC
2. 在 `stores/registry.py` 注册
3. 加配置项到 `core/config.py`
4. 跑契约测试确保通过

## 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件。详见 `.env.example`。

## 架构

```
api/         FastAPI 路由（OpenAI 协议 + 管理）
embeddings/  Embedder 抽象 + 实现
stores/      VectorStore 抽象 + 实现
core/        配置 / 日志 / 指标 / 错误 / 生命周期 / 中间件
schemas/     Pydantic 请求/响应模型
testing/     测试用 FakeEmbedder / FakeStore
```

## 文档

- 设计 spec: `docs/superpowers/specs/2026-08-22-vector-service-design.md`
- 实施计划: `docs/superpowers/plans/2026-08-22-vector-service-plan.md`