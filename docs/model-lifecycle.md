# 模型热加载 / 热卸载

服务对每个模型族（text embedder / image embedder / multimodal embedder / reranker）只保留一个常驻实例挂在 `app.state.<family>` 上。无需重启进程就能换模型或腾显存。

## 启动策略：默认不加载

生产默认行为：**进程启动时所有模型族都不构造、不加载**。`app.state.<family>` 全为 `None`，推理路由（`/v1/embeddings`、`/v1/image_embeddings`、`/v1/multimodal_embeddings`、`/v1/rerank`）立即返回 503。运维在启动后通过 dashboard 面板或 API 显式加载需要的模型。

`/readyz` 的 gate 已收敛到**仅 store 可达性**——模型未加载时 `/readyz` 仍然返回 200，body 中按家族报告 `not_loaded`，方便 K8s probe 不会因为运维还没点 Load 而把流量切走。仅当 vector store 不可达时 `/readyz` 才返 503 `degraded`。

如需保留旧的 eager-load 行为，在 `.env` 设：

```bash
VS_EMBEDDING_AUTO_LOAD=true             # 文本嵌入
VS_IMAGE_EMBEDDING__AUTO_LOAD=true      # 图像嵌入
VS_MULTIMODAL_EMBEDDING__AUTO_LOAD=true  # 图文嵌入
VS_RERANKER__AUTO_LOAD=true             # 重排
```

每个开关独立，不开 eager 的族保持默认的「未加载」状态。

## 加载 / 卸载端点

- `POST /v1/models/{model_id}/load` — 构造并加载该 id 对应的后端实例，替换同族当前实例。返回 `{id, type, status: "loaded", dimensions}`。同 id 重发是幂等操作（不会重建实例）。
- `POST /v1/models/{model_id}/unload` — 释放同族实例并把 `app.state.<family>` 置 `None`，后续该族推理返回 503。返回 `{id, type, status: "unloaded"}`。

```bash
# 加载 BGE-M3
curl -X POST localhost:8080/v1/models/bge-m3/load
# {"id":"bge-m3","type":"embedder","status":"loaded","dimensions":1024}

# 推理现在可用
curl -X POST localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "hello", "model": "bge-m3"}'

# 卸载后内存即释放，再次推理会得到 503 embedder_unavailable
curl -X POST localhost:8080/v1/models/bge-m3/unload
# {"id":"bge-m3","type":"embedder","status":"unloaded"}

# 重新加载
curl -X POST localhost:8080/v1/models/bge-m3/load
```

## Dashboard 面板

打开 <http://localhost:8080/dashboard>，切到「模型 → 加载/卸载」子标签：

- 顶部「刷新状态」按钮调 `GET /v1/models` 拉已注册 id 并按家族分组。
- 每行展示：模型 id + 类型徽标 + 维度、`已加载`/`未加载` pill、Load / Unload 按钮。
- 「自动刷新」开关（默认开）每 5s 轮询一次，仅在当前 tab 可见时拉接口。
- Load/Unload 后自动同步刷新「嵌入」「图像嵌入」「图文嵌入」「重排」各页签下的模型下拉。

## 错误码

- 同族并发 load/unload → 409 `model_busy`（不排队，路由立即返回）。
- 同族已加载了不同 id → 409 `conflict_loaded`（先 unload 再换）。
- load 路由的 factory 或实例 `load()` 抛异常 → 503 `model_load_failed`，slot 保持原状。
- unload 一个空 slot → 409 `not_loaded`。
- 未知 model_id → 404 `model_not_found`。
- 未加载即推理 → 503 `embedder_unavailable` / `image_embedder_unavailable` / `multimodal_embedder_unavailable` / `reranker_not_loaded`。

## 实现细节

每个模型族对应 `core/model_lifecycle.py` 里的 `ModelSlot`，挂到 `app.state._slot_<family>`；lifespan 在启动后按 `auto_load` 决定是否构造 + 加载实例并通过 `set_instance()` 注入对应 slot，热加载/卸载路由与推理路由都基于该 slot 同步状态。卸载在子类里释放原生资源（`_impl`/`_model`/`_preprocess`/`tokenizer` 等）并 best-effort 调用 `torch.cuda.empty_cache()`，base class 提供 no-op 默认实现。
