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
VS_PARSER__AUTO_LOAD=true                # Docling 解析器
```

每个开关独立，不开 eager 的族保持默认的「未加载」状态。

> 异步摄取 worker（`/v1/jobs/ingest` 任务的执行）依赖文本 embedder 已加载；提交任务本身不要求 —— 通常运维需要先 `POST /v1/models/bge-m3/load`，
> 再触发 ingest。`/v1/parse` 与 `/v1/chunk` 不依赖模型，但 Docling 首次加载较慢（30-60 秒）。

## 加载 / 卸载端点

- `POST /v1/models/{model_id}/load` — **异步**。请求只做非阻塞的锁检查，随后在后台线程构造并加载该 id 对应的后端实例，**立即返回 HTTP 202** `{id, type, status: "loading", dimensions: null}`。客户端轮询 `GET /v1/models`，观察该 id 的 `load_status` 走到 `loaded`（此时 `dimensions` 出现、推理路由立即可用）或 `failed`（`load_error` 给出错误信息）。加载失败不再有同步 503 响应。
  - 同 id 已在 slot 上时重发是幂等操作：不重建实例，直接返回 **HTTP 200** `{status: "loaded", dimensions}`。
  - 同族已载不同 id → 409 `conflict_loaded`；加载进行中再次 load/unload → 409 `model_busy`（均为同步返回）。
- `POST /v1/models/{model_id}/unload` — 同步。释放同族实例并把 `app.state.<family>` 置 `None`，后续该族推理返回 503。返回 `{id, type, status: "unloaded"}`。

`GET /v1/models` 与 `GET /v1/models/{id}` 每行包含：

| 字段 | 说明 |
| --- | --- |
| `loaded` (bool) | 实例已在进程上、可立即推理 |
| `load_status` | `unloaded` / `loading`（后台加载中）/ `loaded` / `failed`（最近一次后台加载失败） |
| `load_error` | 失败原因文本；成功或卸载后为 `null` |
| `model_info` (object\|null) | 仅 loaded 行存在；运行时资源信息，见下表。未加载 / 加载中 / 失败行为 `null` |

`model_info` 子字段（全部 best-effort，取不到为 `null`）：

| 字段 | 说明 |
| --- | --- |
| `device` | 实例推理设备（`cuda` / `cpu`，可能带序号如 `cuda:0`），取自后端 wrapper 的 `_device` |
| `dtype` | 主导参数精度，如 `float16` / `bfloat16` / `float32` |
| `param_count` | 实例内所有可达 `torch.nn.Module` 的**去重后**参数总数（共享 / 父子模块不重复计数） |
| `memory_bytes` | 参数 + buffer 的权重字节数（去重）。GPU 模型即确定性的**显存档位下限**，不含 CUDA context、allocator 预留和推理激活；CPU 模型为常驻内存占用 |
| `load_duration_seconds` | 最近一次成功构造 + `load()`（含其中的下载 / 预热）实际墙钟耗时；lifespan 未计时注入时为 `null` |

注意 `loading`/`failed` 只标记**本次被操作的那个 model id**；同族其他已注册 id 仍显示 `unloaded`。

```bash
# 加载 BGE-M3 —— 立即返回 202
curl -X POST localhost:8080/v1/models/bge-m3/load
# HTTP/1.1 202 Accepted
# {"id":"bge-m3","type":"embedder","status":"loading","dimensions":null}

# 轮询直到 load_status 变成 loaded（dashboard 空闲时 5s 一次、有加载中卡片时 2s 一次）
curl -s localhost:8080/v1/models | jq '.data[] | select(.id=="bge-m3")'
# {"id":"bge-m3","type":"embedder","loaded":true,"load_status":"loaded",
#  "load_error":null,"dimensions":1024,...}

# 幂等重发：已在 slot 上，直接 200
curl -X POST localhost:8080/v1/models/bge-m3/load
# HTTP/1.1 200 OK
# {"id":"bge-m3","type":"embedder","status":"loaded","dimensions":1024}

# 推理现在可用
curl -X POST localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "hello", "model": "bge-m3"}'

# 卸载后内存即释放，再次推理会得到 503 embedder_unavailable
curl -X POST localhost:8080/v1/models/bge-m3/unload
# {"id":"bge-m3","type":"embedder","status":"unloaded"}

# 重新加载（再次 202 + 轮询）
curl -X POST localhost:8080/v1/models/bge-m3/load
```

## Dashboard 面板

打开 <http://localhost:8080/dashboard>，切到「模型 → 加载/卸载」子标签：

- 顶部「刷新状态」按钮调 `GET /v1/models` 拉已注册 id 并按家族分组。
- 每行展示：模型 id + 类型徽标 + 维度、状态 pill、Load / Unload 按钮。加载完成后卡片额外展示设备/精度徽标（如 `CUDA · FP16`）与资源信息：参数量、显存占用（CPU 模型显示「内存占用」）、加载耗时，数据来自 `GET /v1/models` 的 `model_info`。
- 点 Load 后卡片在**同一次点击内**乐观切换为琥珀色「加载中…」态（状态 pill 旋转点 + 动作区替换为带转圈的禁用「加载中…」按钮，无需等 POST/GET 往返）；POST 立即返回（202），后台完成后静默翻转为「已加载」；失败则显示红色「加载失败」pill + 错误文案，Load 按钮恢复可点用于重试（同时弹一次失败提示）。Unload 同理，点击瞬间显示禁用的「卸载中…」。乐观标记只在轮询观测到与动作一致的终态后才解除，因此点击前一刻发出的旧轮询回包不会把卡片闪回可点击态。
- 「自动刷新」开关（默认开）轮询 `GET /v1/models`：空闲 5s 一次，任一卡片处于加载中时自动加密到 2s，加载结束后恢复 5s。
- Load/Unload 后自动同步刷新「嵌入」「图像嵌入」「图文嵌入」「重排」各页签下的模型下拉。

## 错误码

- 同族并发 load/unload → 409 `model_busy`（不排队，路由立即返回；后台加载进行中再发 load/unload 也是此错误）。
- 同族已加载了不同 id → 409 `conflict_loaded`（先 unload 再换）。
- 后台加载中 factory 或实例 `load()` 抛异常 → slot 保持为空，`GET /v1/models` 中该 id 为 `load_status="failed"` + `load_error`（服务端日志记录 `model_load_failed`，含 exception_type）。**不再同步返回 503。**
- unload 一个空 slot → 409 `not_loaded`。
- 未知 model_id → 404 `model_not_found`。
- 未加载即推理 → 503 `embedder_unavailable` / `image_embedder_unavailable` / `multimodal_embedder_unavailable` / `reranker_not_loaded`。

## 实现细节

每个模型族对应 `core/model_lifecycle.py` 里的 `ModelSlot`，挂到 `app.state._slot_<family>`；lifespan 在启动后按 `auto_load` 决定是否构造 + 加载实例并通过 `set_instance()` 注入对应 slot，热加载/卸载路由与推理路由都基于该 slot 同步状态。热加载走两阶段 API：`begin_load()` 在事件循环线程做非阻塞抢锁 + 幂等/冲突判定并置状态为 `loading`（锁保持持有），`finish_load()` 在线程池里跑 factory + `load()` 并在结束时释放锁、翻转 `loaded`/`failed` 状态；路由收到 202 后用 `asyncio.create_task` 调度第二阶段，完成后把实例镜像到 `app.state.<family>` 并更新 `MODEL_LOADED` 指标（任务引用保存在 `app.state._model_tasks` 防止被 GC）。卸载在子类里释放原生资源（`_impl`/`_model`/`_preprocess`/`tokenizer` 等）并 best-effort 调用 `torch.cuda.empty_cache()`，base class 提供 no-op 默认实现。`model_info` 的资源字段由 `core/model_info.py` 以有限深度、循环安全的属性遍历发现实例内全部 `nn.Module`，参数按对象 id 去重后统计；结果按实例缓存在 `WeakKeyDictionary` 里，卸载即随实例一起回收。加载耗时由 `ModelSlot` 在 `_build_and_install()` / `finish_load()` 内计时（lifespan eager load 则由 lifespan 计时后传入 `set_instance(load_duration=...)`），卸载或失败时清零。
