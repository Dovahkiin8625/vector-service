# Image Embeddings — Design Spec

- **Status:** Draft, pending review
- **Date:** 2026-09-12
- **Scope:** Add image vectorization to `vector-service` so callers can submit images and run k-NN image-to-image search.

---

## 1. Context & Problem

`vector-service` currently embeds **text only** (BGE-M3). Callers want to provide
**images** and search an image collection by image (图搜图).

The vector store layer (Milvus) is modality-agnostic — vectors are `list[float]`.
The missing piece is an image embedder + API surface for image inputs.

Cross-modal text↔image is **out of scope** for this spec (the user explicitly
scoped image-only). The embedder ABI is designed so a future cross-modal model
can be added without breaking changes.

---

## 2. Goals & Non-Goals

### Goals
1. Add an image embedding model (default: OpenCLIP ViT-L/14, 768d) loaded at startup.
2. Expose `POST /v1/image_embeddings` accepting base64 images.
3. Extend `PUT .../vectors` to accept `images` so callers can populate image collections directly.
4. Extend `POST .../search` to accept `query_image` so callers can search image collections with an image.
5. List image embedders in `GET /v1/models` under `type=image_embedder`.
6. Follow existing patterns: separate ABC + registry, nested settings, lifecycle hook, error envelope, metrics.
7. Backward-compatible: zero changes to existing text embedder behavior or `/v1/embeddings`.

### Non-Goals
- Cross-modal text↔image embedding (can be added later via the same `ImageEmbedder` ABC if a cross-modal model is registered).
- Image storage / thumbnail serving / image-search UI — only embedding + vector ops.
- URL-based image fetching (callers send base64).
- Multipart upload (callers send JSON).

---

## 3. Architecture

Mirror the existing `Embedder`/`Reranker` split. New image subsystem is a sibling, not a replacement.

```
api/             ├─ embeddings.py      (text, unchanged)
                 ├─ image_embeddings.py (NEW)
                 ├─ management.py      (upsert/search extended)
                 ├─ models.py          (lists image_embedder type)
embeddings/      ├─ base.py            (text ABC, unchanged)
                 ├─ image_base.py      (NEW — ImageEmbedder ABC + ImageInput)
                 ├─ image_registry.py  (NEW)
                 ├─ openclip_vit_l14.py (NEW — concrete)
                 └─ ...
schemas/         ├─ openai.py          (Model.type adds "image_embedder")
                 ├─ image_embeddings.py (NEW)
                 └─ management.py      (upsert/search extended)
core/            ├─ config.py          (ImageEmbeddingSettings nested)
                 ├─ lifespan.py        (image_embedder load hook)
                 ├─ metrics.py         (IMAGE_EMBEDDING_* labels)
                 └─ errors.py          (existing envelope reused)
```

App state after `lifespan`:

```python
app.state.embedder        # text, existing
app.state.image_embedder  # NEW
```

Routes dispatch between them by inspecting the `model` field on the request body
(when images are involved) or by the endpoint itself (`/v1/embeddings` is always
text; `/v1/image_embeddings` is always image).

---

## 4. Components

### 4.1 `ImageEmbedder` ABC — `embeddings/image_base.py`

```python
@dataclass(frozen=True)
class ImageInput:
    data: bytes   # raw image bytes (post base64 decode)
    mime: str     # e.g. "image/png"

class ImageEmbedder(ABC):
    dim: int
    model_name: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        """Batch-embed. Return one vector per input, in order."""

    @abstractmethod
    def embed_query_image(self, image: ImageInput) -> list[float]:
        """Embed one query. May differ from embed_images (e.g. model-specific
        prompt / preprocessing). For OpenCLIP this is identical."""
```

`ImageInput` is intentionally small: bytes + mime. Bytes→PIL conversion lives in
helper functions (see §4.3); per-model preprocessing stays in the subclass.

### 4.2 Registry — `embeddings/image_registry.py`

```python
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

IMAGE_EMBEDDER_REGISTRY: dict[str, type[ImageEmbedder]] = {
    "openclip-vit-l-14": OpenCLIPVitL14ImageEmbedder,
}

def get_image_embedder_class(name: str) -> type[ImageEmbedder]:
    if name not in IMAGE_EMBEDDER_REGISTRY:
        raise ImageEmbedderError(f"unknown image embedding model: {name!r}")
    return IMAGE_EMBEDDER_REGISTRY[name]

def list_image_embedder_names() -> list[str]:
    return list(IMAGE_EMBEDDER_REGISTRY.keys())
```

Mirror of `embeddings/registry.py`. `ImageEmbedderError` is a new exception
class in `core/errors.py`, parallel to `EmbedderError`.

### 4.3 Concrete embedder — `embeddings/openclip_vit_l14.py`

- Class: `OpenCLIPVitL14ImageEmbedder(ImageEmbedder)`, `dim = 768`, `model_name = "openclip-vit-l-14"`.
- Loads via `open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", ...)`.
- `device` and `use_fp16` resolved the same way as `BGEM3Embedder` (`auto|cpu|cuda`).
- `load()` is idempotent; calls `embed_images([...warmup image...])` once for warmup (warmup failure is logged, not fatal — same pattern as BGE-M3).
- `embed_images(images)`:
  - For each `ImageInput`, decode bytes → `PIL.Image` (using `PIL.Image.open(BytesIO(...))`).
  - Apply `self._preprocess(pil)` (the transform returned by `create_model_and_transforms`).
  - `self._model.encode_image(batch.to(device))`.
  - Return list of `list[float]` (move to CPU, cast to float).
- `embed_query_image(image)`:
  - Single-image path; equivalent to `embed_images([image])[0]`. OpenCLIP has no distinct query mode.
- Auto-download from HuggingFace (open_clip's catalog is HF-only); `model_dir` defaults to `./models/openclip-vit-l-14`. `snapshot_download(repo_id="openai/ViT-L-14", ...)` plus `open_clip`'s weights.

### 4.4 Image decoding helper — `embeddings/image_decoding.py`

```python
def decode_image(b64: str, mime: str, *, max_bytes: int, allowed_mime: set[str]) -> ImageInput:
    """Decode base64 + validate MIME + size.

    Raises the exception family defined in §4.13:
      - UnsupportedMime        → maps to 422 `unsupported_mime`
      - ImageTooLarge          → maps to 422 `image_too_large`
      - ImageDecodeError       → maps to 422 `image_decode_failed`
    """
    if mime not in allowed_mime:
        raise UnsupportedMime(f"unsupported mime: {mime!r}", got=mime, allowed=sorted(allowed_mime))
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ImageTooLarge(f"image too large: {len(raw)} > {max_bytes}", got=len(raw), max=max_bytes)
    return ImageInput(data=raw, mime=mime)
```

`allowed_mime` is read from `Settings.image_embedding.allowed_mime` (default
`{"image/jpeg", "image/png", "image/webp"}`). `max_bytes` is
`Settings.image_embedding.max_image_bytes`.

### 4.5 Lifespan — `core/lifespan.py`

Add a new block after the existing `embedder.load()`:

```python
img_settings = settings.image_embedding
img_cls = get_image_embedder_class(img_settings.backend)
img_embedder = img_cls(img_settings)
try:
    img_embedder.load()
except (ImageEmbedderError, ModelNotLoaded) as e:
    log.error("image_embedder_load_failed", error=str(e))
    raise
app.state.image_embedder = img_embedder
```

`/readyz` is extended to require `image_embedder` is loaded (it already requires
`embedder`); the existing 503 path is unchanged in shape, just additional check.

### 4.6 Config — `core/config.py`

```python
class ImageEmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VS_IMAGE_EMBEDDING__", extra="ignore")
    backend: str = "openclip-vit-l-14"
    model_dir: str = "./models/openclip-vit-l-14"
    auto_download: bool = True
    download_source: Literal["huggingface"] = "huggingface"
    hf_repo: str = "openai/ViT-L-14"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(16, ge=1, le=512)
    max_images_per_request: int = Field(64, ge=1, le=1024)
    max_image_bytes: int = Field(10 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    allowed_mime: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

class Settings(BaseSettings):
    ...
    image_embedding: ImageEmbeddingSettings = Field(default_factory=ImageEmbeddingSettings)
```

### 4.7 Schemas — `schemas/image_embeddings.py`

```python
class ImageInputItem(BaseModel):
    data: str = Field(description="Base64-encoded image bytes (no data: URI prefix).")
    mime: str = Field(description="Image MIME type; must be one of allowed_mime.")

class ImageEmbeddingRequest(BaseModel):
    model: str = Field(description="Image embedder model id (see GET /v1/models).")
    input: ImageInputItem | list[ImageInputItem] = Field(
        description="One or more base64-encoded images."
    )
    encoding_format: Literal["float"] = "float"
    user: str | None = None

class ImageEmbeddingData(BaseModel):
    object: Literal["image_embedding"] = "image_embedding"
    index: int
    embedding: list[float]

class ImageEmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ImageEmbeddingData]
    model: str
    usage: EmbeddingUsage   # prompt_tokens := image_count, total_tokens := image_count
```

### 4.8 `Model` schema — `schemas/openai.py`

`type` field widens to:

```python
type: Literal["embedder", "reranker", "image_embedder"]
```

`/v1/models` joins text embedders, image embedders, and rerankers under a single
list. `dimensions` is populated for both `embedder` and `image_embedder`, `null`
for `reranker`.

### 4.9 Management schemas — `schemas/management.py`

```python
class UpsertVectorsRequest(BaseModel):
    primary_field: str
    vector_field: str
    ids: list[str] = Field(min_length=1)

    texts: list[str] | None = None
    vectors: list[list[float]] | None = None
    images: list[str] | None = Field(
        default=None,
        description="Base64-encoded image bytes (parallel to ids).",
    )
    image_mimes: list[str] | None = Field(
        default=None,
        description="MIME per image (parallel to ids). Required when images is provided.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Embedder model id. Required when images is provided (selects the "
            "image embedder). Optional otherwise (uses the default text embedder)."
        ),
    )
    fields: list[dict] | None = None

    @model_validator(mode="after")
    def _xor_three_way(self):
        has_t = self.texts is not None
        has_v = self.vectors is not None
        has_i = self.images is not None
        if sum([has_t, has_v, has_i]) != 1:
            raise ValueError("provide exactly one of texts, vectors, images")
        n = len(self.ids)
        if has_t and len(self.texts) != n: ...
        if has_v and len(self.vectors) != n: ...
        if has_i:
            if self.image_mimes is None or len(self.image_mimes) != n:
                raise ValueError("image_mimes required, same length as ids")
            if self.model is None:
                raise ValueError("model is required when images is provided")
        if self.fields is not None and len(self.fields) != n: ...
        return self
```

`SearchRequest` extends identically:

```python
class SearchRequest(BaseModel):
    primary_field: str
    vector_field: str
    query_text: str | None = None
    query_vector: list[float] | None = None
    query_image: str | None = None
    query_image_mime: str | None = None
    model: str | None = None
    top_k: int = 10
    filter_expr: str | None = None
    output_fields: list[str] | None = None

    @model_validator(mode="after")
    def _xor_three_way(self):
        ...
        if has_i and (self.model is None or self.query_image_mime is None):
            raise ValueError("model and query_image_mime required when query_image is provided")
        return self
```

### 4.10 Routes — `api/image_embeddings.py`

Mirror of `api/embeddings.py`. Differences:

- Pulls `image_embedder` from `request.app.state.image_embedder`.
- Validates `model` via `get_image_embedder_class()`.
- Decodes each input via `decode_image(b64, mime, max_bytes=..., allowed_mime=...)` — collects decode failures into a single 422 with `failed_indices: list[int]` (one envelope per error code: `image_decode_failed`, `image_too_large`, `unsupported_mime`).
- Calls `embedder.embed_images([...])` inside `loop.run_in_executor`.
- Records metrics under `IMAGE_EMBEDDING_*` labels (see §4.12).

### 4.11 Routes — `api/management.py` (extended)

`upsert_vectors` dispatch:

```python
if body.images is not None:
    cls = get_image_embedder_class(body.model)
    image_embedder = request.app.state.image_embedder
    inputs = [
        decode_image(
            b64, mime,
            max_bytes=settings.image_embedding.max_image_bytes,
            allowed_mime=set(settings.image_embedding.allowed_mime),
        )
        for b64, mime in zip(body.images, body.image_mimes)
    ]
    vectors = await loop.run_in_executor(None, image_embedder.embed_images, inputs)
elif body.texts is not None:
    ...
else:
    vectors = body.vectors or []
```

`search` dispatch: identical pattern with `embed_query_image`.

### 4.12 Metrics — `core/metrics.py`

Add parallel series to `EMBEDDING_*`:

```python
IMAGE_EMBEDDING_REQUESTS_TOTAL = Counter(...)
IMAGE_EMBEDDING_DURATION_SECONDS = Histogram(...)
IMAGE_EMBEDDING_INPUTS_TOTAL = Counter(... labels=["model"])
```

Per-store upsert/search get a new `op` value `upsert_image` and `search_image`
for distinct dashboards.

### 4.13 Errors — `core/errors.py` + envelope

| code | HTTP | trigger |
|------|------|---------|
| `image_embedder_unavailable` | 503 | image model not loaded / inference error |
| `image_decode_failed` | 422 | base64 / MIME invalid |
| `image_too_large` | 422 | decoded bytes > `max_image_bytes` |
| `too_many_images` | 422 | count > `max_images_per_request` |
| `unsupported_mime` | 422 | mime not in `allowed_mime` |

New exception classes in `core/errors.py`:

```python
class ImageEmbedderError(VectorServiceError): ...
class ModelNotLoadedForImages(ImageEmbedderError): ...   # mirrors ModelNotLoaded

class ImageDecodeError(VectorServiceError):
    """Generic base64 / image-bytes failure (422 image_decode_failed)."""

class UnsupportedMime(ImageDecodeError):
    got: str
    allowed: list[str]

class ImageTooLarge(ImageDecodeError):
    got: int
    max: int
```

Routes catch each subclass and raise `HTTPException` with the matching `code`
from the table above, with the canonical `detail={"error": {...}}` envelope.
The global handler in `main.py` does not need changes — it already converts
`HTTPException` to the canonical envelope.

Decode failures inside an `/v1/image_embeddings` request that contains multiple
images are aggregated into a single 422 response with
`extra.failed_indices: list[int]` so the caller knows which inputs were bad
without retrying the whole batch.

---

## 5. Data Flow

### 5.1 Embed-only path (image → vectors)
```
client → POST /v1/image_embeddings
  → decode_image per item (422 on failure)
  → get_image_embedder_class(model) (404 on miss)
  → image_embedder.embed_images([...]) in executor
  → ImageEmbeddingResponse { data: [...], model, usage }
```

### 5.2 Image upsert
```
client → PUT /v1/databases/{db}/collections/{coll}/vectors { images: [...], image_mimes: [...], model, ids, fields }
  → decode_image per item
  → image_embedder.embed_images([...])
  → store.upsert(db, coll, ids, vectors, fields)
  → { upserted: N }
```

### 5.3 Image search
```
client → POST /v1/databases/{db}/collections/{coll}/search { query_image, query_image_mime, model, top_k, filter_expr }
  → decode_image(query_image, query_image_mime)
  → image_embedder.embed_query_image(input)
  → store.search(db, coll, vector_field, qvec, top_k, filter_expr, output_fields)
  → SearchResponse { hits: [...] }
```

---

## 6. Testing

### Unit — `tests/unit/test_image_embedder_openclip.py`
- Stub `open_clip.create_model_and_transforms` returning a fake model + preprocess.
- Assert `embed_images([a, b, c])` returns 3 vectors of length 768.
- Assert `embed_query_image` is functionally equivalent to `embed_images([x])[0]`.
- `load()` warmup is called and is idempotent.
- Bytes-decode failure (bad base64, bad MIME) raises `ImageDecodeError`.

### Unit — `tests/unit/test_image_embedding_route.py`
- `TestClient` with stubbed `image_embedder`.
- Happy path: 1 image and N images → response shape (`object="image_embedding"`, indices in order, `usage.prompt_tokens == N`).
- Error paths: `image_decode_failed` (bad base64), `image_too_large` (> max bytes), `too_many_images` (> N), `unsupported_mime` (e.g. `image/bmp`), `image_embedder_unavailable` (stub raises), `model_not_found` (unknown `model`).
- Limits: `max_images_per_request`, `max_image_bytes` come from settings and are honored.

### Unit — `tests/unit/test_management_image_extension.py`
- `PUT .../vectors` with `images` and `image_mimes` flows through to a fake store with correct dim.
- 3-way XOR validator: zero inputs rejected; two inputs rejected; mismatched lengths rejected; missing `model` rejected when `images` present.
- `POST .../search` with `query_image` produces a k-NN against a fake image store.
- `model` field selects between text and image embedder.

### Contract — `tests/contract/test_image_models_listing.py`
- `GET /v1/models` exposes image embedders with `type="image_embedder"`, `dimensions=768` for `openclip-vit-l-14`.

No live Milvus tests (matches existing convention).

---

## 7. Dependencies

- Add `open_clip_torch>=2.24` to a new optional extra `[image-embed]` in `pyproject.toml`.
- `Pillow`, `torch` already required by the BGE-M3 path — no new transitive deps.
- README gets a new section "图像嵌入（Image Embeddings）" mirroring Rerank.

`.env.example` gets:

```bash
VS_IMAGE_EMBEDDING__BACKEND=openclip-vit-l-14
VS_IMAGE_EMBEDDING__MODEL_DIR=./models/openclip-vit-l-14
VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=true
VS_IMAGE_EMBEDDING__DEVICE=auto
VS_IMAGE_EMBEDDING__BATCH_SIZE=16
VS_IMAGE_EMBEDDING__MAX_IMAGES_PER_REQUEST=64
VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES=10485760
```

---

## 8. Open Questions

None at spec time. The two intentional design choices to confirm at review:

1. **`image_mimes` is required** (not defaulted to `image/jpeg`) — strict, avoids wrong-format corruption.
2. **Image embedding share the same `GET /v1/models` shape** as text and rerank — one endpoint, one `Model` schema, discriminated by `type`. Callers filter with `?type=image_embedder` (extend the optional filter if desired, or filter client-side).

Both are minor; flag in review if you want them changed.