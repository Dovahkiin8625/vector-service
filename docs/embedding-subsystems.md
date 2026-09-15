# 嵌入与重排子系统

服务内置 4 个推理子系统：文本嵌入、图像嵌入、图文跨模态嵌入、cross-encoder 重排。每个子系统都可以通过 `load` / `unload` 端点独立热加载/释放显存，详见 [model-lifecycle.md](model-lifecycle.md)。通用配置见 [configuration.md](configuration.md)。

---

## 文本嵌入（BGE-M3）

`POST /v1/embeddings`：OpenAI 兼容协议，接受字符串或字符串列表，自动选择 BGE-M3（GPU 走 torch fp16，CPU 走 ONNX int8）。

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'
```

返回示例：

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

错误码：`model_not_found`（404）/ `too_many_texts` / `text_too_long` / `invalid_request`（422）/ `embedder_unavailable`（503）。

---

## 图像嵌入（OpenCLIP）

`POST /v1/image_embeddings`：当前内置 `openclip-vit-l-14`（OpenCLIP ViT-L/14，openai 预训练权重，768 维）。

```bash
curl -X POST localhost:8080/v1/image_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openclip-vit-l-14",
    "input": {
      "data": "'$(base64 -w0 cat.png)'",
      "mime": "image/png"
    }
  }'
```

返回示例：

```json
{
  "object": "list",
  "data": [
    {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]}
  ],
  "model": "openclip-vit-l-14",
  "usage": {"prompt_tokens": 1, "total_tokens": 1}
}
```

### 启动

首次启动会自动从 open_clip 的 CDN（`openaipublic.azureedge.net`）下载 openai 预训练的 `ViT-L-14` 权重到 `./models/openclip-vit-l-14/`（`OPEN_CLIP_DOWNLOAD_PATH` 环境变量也会被指向同一目录）。离线环境把 `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_IMAGE_EMBEDDING__MODEL_DIR` 指定的目录。

所有 `VS_IMAGE_EMBEDDING__*` 配置项见 `.env.example` 的 `Image embedding` 段。

### 图搜图：upsert + search 接入

`PUT /v1/databases/{db}/collections/{coll}/vectors` 和 `POST .../search` 同时接受文本、向量和图像三种输入（三选一）。图像模式下：

- `PUT`：body 用 `images`（base64 列表）+ `image_mimes`（并行）+ `model`（图像嵌入器 id）代替 `texts`。
- `search`：body 用 `query_image`（base64 单张）+ `query_image_mime` + `model` 代替 `query_text`。

```bash
# 1) upsert 图片（collection dim 必须等于 768）
curl -X PUT localhost:8080/v1/databases/tenant-a/collections/products/vectors \
  -H 'Content-Type: application/json' \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "ids": ["sku-1"],
    "images": ["'"$(base64 -w0 mouse.png)"'"],
    "image_mimes": ["image/png"],
    "model": "openclip-vit-l-14"
  }'

# 2) 图搜图
curl -X POST localhost:8080/v1/databases/tenant-a/collections/products/search \
  -H 'Content-Type: application/json' \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "query_image": "'"$(base64 -w0 query.png)"'",
    "query_image_mime": "image/png",
    "model": "openclip-vit-l-14",
    "top_k": 5
  }'
```

错误码：`model_not_found`（404）/ `image_decode_failed` / `image_too_large` / `unsupported_mime` / `too_many_images`（422）/ `image_embedder_unavailable`（503）。

---

## 图文跨模态嵌入（Chinese-CLIP）

`POST /v1/multimodal_embeddings`：跨模态嵌入子系统同时接受中文文本和 base64 图片，统一在 **512 维共享投影空间**中输出向量。两端的向量可以直接做 cos similarity，是 **文搜图** 和 **图搜文** 检索的基础。当前内置 `chinese-clip-vit-base-patch16`（Chinese-CLIP ViT-B/16，OFA-Sys 预训练权重）。

> 与图像嵌入 (`openclip-vit-l-14`, 768 维) 不同，跨模态模型的两个塔必须使用同一个 `MultimodalEmbedder` 实例、产生同维度向量，才能保证跨模态相似度有意义。

```bash
# 纯文本
curl -X POST localhost:8080/v1/multimodal_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chinese-clip-vit-base-patch16",
    "input": [{"text": "一只猫"}, {"text": "一只狗"}]
  }'

# 混合输入 — 返回顺序与请求顺序一一对应
curl -X POST localhost:8080/v1/multimodal_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chinese-clip-vit-base-patch16",
    "input": [
      {"text": "一只小猫在窗台上晒太阳"},
      {"image": {"data": "'$(base64 -w0 cat.png)'", "mime": "image/png"}},
      {"text": "一只小狗在草地上奔跑"}
    ]
  }'
```

返回示例：

```json
{
  "object": "list",
  "data": [
    {"object": "multimodal_embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]},
    {"object": "multimodal_embedding", "index": 1, "embedding": [0.0789, -0.1011, ...]},
    {"object": "multimodal_embedding", "index": 2, "embedding": [0.1314, -0.1718, ...]}
  ],
  "model": "chinese-clip-vit-base-patch16",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

### 启动

首次启动会从 HuggingFace（默认 `OFA-Sys/chinese-clip-vit-base-patch16`）下载权重到 `./models/chinese-clip-vit-base-patch16/`。离线环境把 `VS_MULTIMODAL_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_MULTIMODAL_EMBEDDING__MODEL_DIR` 指定的目录。`VS_MULTIMODAL_EMBEDDING__*` 配置项见 `.env.example` 的 `Multimodal embedding` 段。

### 文搜图 / 图搜文

由于文本塔和图像塔输出同空间向量，把文本向量与图片向量存进同一个 `dim=512` 的 Milvus collection 之后，就可以直接做 cos 相似度检索（Milvus 端无需区分 query 端是文本还是图片 — 上层把文本 query 通过同一 embedder 转成向量即可）。Milvus upsert / search 端点打通跨模态路由在后续版本提供；本轮先把图文嵌入能力上线。

错误码：`model_not_found`（404）/ `image_decode_failed` / `image_too_large` / `unsupported_mime` / `text_too_long` / `too_many_items`（422）/ `multimodal_embedder_unavailable`（503）。

---

## 重排序（cross-encoder）

`POST /v1/rerank`：把向量检索回来的候选 documents 交给 cross-encoder 重排，按相关性得分降序输出 `{index, score}` 列表。当前内置 `bge-reranker-v2-m3`（`BAAI/bge-reranker-v2-m3`）。

```bash
curl -X POST localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "中国首都",
    "documents": ["巴黎是法国首都", "苹果是一种水果", "北京是中华人民共和国的首都"],
    "top_n": 3
  }'
```

返回示例：

```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 2, "score": 0.9876},
    {"index": 0, "score": 0.0432}
  ],
  "request_id": "8f4e1c2a-9b1d-4f0e-9c1a-2b3c4d5e6f70"
}
```

### 启动

reranker 默认随主进程一起启动。在 `.env` 里设好 `VS_RERANKER__BACKEND`，然后像往常一样跑：

```bash
VS_RERANKER__BACKEND=bge-reranker-v2-m3 vector-service
```

首次启动会自动从 ModelScope 下载 `BAAI/bge-reranker-v2-m3` 权重到 `./models/bge-reranker-v2-m3/`；离线环境可以把 `VS_RERANKER__AUTO_DOWNLOAD=false` 然后手工把权重放到 `VS_RERANKER__MODEL_DIR` 指定的目录。

错误码：`model_not_found`（404）/ `invalid_request` / `too_many_documents` / `document_too_long` / `query_too_long` / `invalid_top_n`（422）/ `reranker_not_loaded` / `reranker_error`（503）。
