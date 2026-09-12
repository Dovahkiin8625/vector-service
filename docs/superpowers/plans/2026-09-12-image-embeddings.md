# Image Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image vectorization to `vector-service` so callers can submit images (base64) and run k-NN image-to-image search.

**Architecture:** Mirror the existing text `Embedder` subsystem. New `ImageEmbedder` ABC + `IMAGE_EMBEDDER_REGISTRY` + new `POST /v1/image_embeddings` route. Extend `PUT .../vectors` and `POST .../search` to accept `images` / `query_image` inputs. Add `OpenCLIPVitL14ImageEmbedder` (768d) as the first concrete backend.

**Tech Stack:** FastAPI, pydantic v2, pydantic-settings, open_clip_torch (new), Pillow (existing), pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-image-embeddings-design.md`

---

## Global Constraints

These apply to every task. Each task's requirements implicitly include this section.

- **Spec is authoritative.** If a step seems to contradict the spec, the spec wins — flag the conflict and ask before deviating.
- **Python ≥ 3.11**, project name `vector-service`, src layout at `src/vector_service/`, tests at `tests/`.
- **Naming & style** — match existing code:
  - Module docstrings start with `"""... """` and the first line is a single sentence.
  - Type hints use `from __future__ import annotations` and modern syntax (`list[str]`, `dict`, `X | None`).
  - Tests use `pytest`, plain `assert`, no unittest class hierarchy.
  - Use `monkeypatch` (not `unittest.mock.patch`) for stubbing.
- **HTTP errors** use the canonical envelope already wired in `src/vector_service/main.py:38-71`. Routes raise `HTTPException(status_code=..., detail={"error": {...}})`; the global handler in `main.py:229-247` rewrites it.
- **No external calls in unit tests.** `open_clip`, `huggingface_hub`, `PIL.Image.open(BytesIO(...))` on real bytes — all stubbed via `monkeypatch`.
- **Commit cadence.** Every task ends with `git commit`. Tasks are not merged across commits.
- **Commit message prefix**: `feat:`, `fix:`, `test:`, `chore:`, `docs:` — conventional commits.
- **Test command:** `pytest tests/unit -q` (and `pytest tests/contract -q` where applicable). Run from repo root.
- **Backward compatibility:** zero changes to existing text `Embedder` behavior or `POST /v1/embeddings`. Existing tests must continue to pass.

---

## File Structure (target end-state)

### New files
| Path | Purpose |
|------|---------|
| `src/vector_service/embeddings/image_base.py` | `ImageInput` dataclass + `ImageEmbedder` ABC |
| `src/vector_service/embeddings/image_registry.py` | `IMAGE_EMBEDDER_REGISTRY` + lookup helpers |
| `src/vector_service/embeddings/image_decoding.py` | `decode_image(b64, mime, …)` helper |
| `src/vector_service/embeddings/openclip_vit_l14.py` | `OpenCLIPVitL14ImageEmbedder` concrete class |
| `src/vector_service/schemas/image_embeddings.py` | request/response schemas for `POST /v1/image_embeddings` |
| `src/vector_service/api/image_embeddings.py` | `POST /v1/image_embeddings` route |
| `tests/unit/test_image_embedder_base.py` | ABC contract tests |
| `tests/unit/test_image_decoding.py` | decode helper tests (all error paths) |
| `tests/unit/test_image_settings.py` | config env-var parsing |
| `tests/unit/test_image_registry.py` | registry lookup |
| `tests/unit/test_image_embedder_openclip.py` | concrete embedder w/ mocked `open_clip` |
| `tests/unit/test_image_embedding_schemas.py` | pydantic schema validation |
| `tests/unit/test_image_embedding_route.py` | route behavior via `TestClient` |
| `tests/unit/test_management_image_schemas.py` | extended `UpsertVectorsRequest` / `SearchRequest` |
| `tests/unit/test_management_image_routes.py` | extended upsert/search behavior |
| `tests/unit/test_image_lifespan.py` | lifespan wiring + `/readyz` |
| `tests/unit/test_image_models_listing.py` | `GET /v1/models` includes image embedders |

### Modified files
| Path | Change |
|------|--------|
| `src/vector_service/core/errors.py` | Add `ImageEmbedderError`, `ModelNotLoadedForImages`, `ImageDecodeError`, `UnsupportedMime`, `ImageTooLarge` |
| `src/vector_service/core/config.py` | Add `ImageEmbeddingSettings` nested under `Settings.image_embedding` |
| `src/vector_service/core/metrics.py` | Add `IMAGE_EMBEDDING_REQUESTS_TOTAL`, `IMAGE_EMBEDDING_DURATION_SECONDS`, `IMAGE_EMBEDDING_INPUTS_TOTAL`; init `MODEL_LOADED.labels(kind="image_embedder").set(0)` |
| `src/vector_service/core/lifespan.py` | Add `build_image_embedder()`, lifespan block that loads it (mirrors text embedder block) |
| `src/vector_service/api/health.py` | `_image_embedder_loaded()`, `/readyz` body includes `"image_embedder"` key |
| `src/vector_service/api/models.py` | Include image embedders in the model list (filterable by `type`) |
| `src/vector_service/schemas/openai.py` | `Model.type` widens to `Literal["embedder", "reranker", "image_embedder"]` |
| `src/vector_service/schemas/management.py` | `UpsertVectorsRequest`: add `images`, `image_mimes`, `model`, plus 3-way XOR validator. `SearchRequest`: add `query_image`, `query_image_mime`, `model`, plus 3-way XOR validator |
| `src/vector_service/api/management.py` | Extend `upsert_vectors` + `search` to dispatch on `images` / `query_image` |
| `pyproject.toml` | Add `image-embed` optional extra |
| `.env.example` | Add `VS_IMAGE_EMBEDDING__*` block |
| `README.md` | Add "图像嵌入（Image Embeddings）" section |

---

### Task 1: Image-embedding exception hierarchy

**Files:**
- Modify: `src/vector_service/core/errors.py:7-65` (append new exception classes)
- Create: `tests/unit/test_image_errors.py`

**Interfaces:**
- Consumes: existing `VectorServiceError` base
- Produces: `ImageEmbedderError`, `ModelNotLoadedForImages`, `ImageDecodeError`, `UnsupportedMime(got, allowed)`, `ImageTooLarge(got, max)`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_errors.py`:

```python
"""Exception classes for image embedding subsystem."""
from __future__ import annotations

import pytest

from vector_service.core.errors import (
    EmbedderError,
    ImageDecodeError,
    ImageEmbedderError,
    ImageTooLarge,
    ModelNotLoaded,
    ModelNotLoadedForImages,
    UnsupportedMime,
    VectorServiceError,
)


def test_image_embedder_error_inherits_vector_service_error():
    assert issubclass(ImageEmbedderError, VectorServiceError)


def test_model_not_loaded_for_images_inherits_image_embedder_error():
    assert issubclass(ModelNotLoadedForImages, ImageEmbedderError)
    assert issubclass(ModelNotLoadedForImages, ModelNotLoaded) is False  # distinct from text


def test_image_decode_error_inherits_vector_service_error():
    assert issubclass(ImageDecodeError, VectorServiceError)


def test_unsupported_mime_carries_got_and_allowed():
    e = UnsupportedMime("nope", got="image/bmp", allowed=["image/jpeg", "image/png"])
    assert e.got == "image/bmp"
    assert e.allowed == ["image/jpeg", "image/png"]
    assert str(e) == "nope"


def test_image_too_large_carries_got_and_max():
    e = ImageTooLarge("big", got=2_000_000, max=1_000_000)
    assert e.got == 2_000_000
    assert e.max == 1_000_000


def test_image_embedder_error_caught_as_embedder_error_for_dispatch():
    """Image embedder errors are siblings, NOT children of EmbedderError —
    routes dispatch on the specific subclass."""
    assert not issubclass(ImageEmbedderError, EmbedderError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_errors.py -q`
Expected: `ImportError` or `ModuleNotFoundError` for the new exception names.

- [ ] **Step 3: Implement the exception classes**

Append to `src/vector_service/core/errors.py` (after the existing `RerankerNotLoaded` class):

```python
class ImageEmbedderError(VectorServiceError):
    """Base class for image embedding failures (sibling of EmbedderError)."""


class ModelNotLoadedForImages(ImageEmbedderError):
    """Image model failed to load at startup or is no longer available.

    Intentionally NOT a subclass of ModelNotLoaded — the text and image
    embedder lifecycles are independent, and a missing image model should
    not look like a text-embedder failure to dispatchers.
    """


class ImageDecodeError(VectorServiceError):
    """Base class for image decoding failures (422 image_decode_failed family)."""


class UnsupportedMime(ImageDecodeError):
    """MIME type was not in the allow-list."""

    def __init__(self, message: str, *, got: str, allowed: list[str]):
        super().__init__(message)
        self.got = got
        self.allowed = allowed


class ImageTooLarge(ImageDecodeError):
    """Decoded image bytes exceeded the configured maximum."""

    def __init__(self, message: str, *, got: int, max: int):
        super().__init__(message)
        self.got = got
        self.max = max
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_errors.py -q`
Expected: all 6 tests pass.

- [ ] **Step 5: Run full unit suite to confirm no regression**

Run: `pytest tests/unit -q`
Expected: all existing tests pass plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add src/vector_service/core/errors.py tests/unit/test_image_errors.py
git commit -m "feat(image): add image embedding exception hierarchy"
```

---

### Task 2: Image decoding helper

**Files:**
- Create: `src/vector_service/embeddings/image_decoding.py`
- Create: `tests/unit/test_image_decoding.py`

**Interfaces:**
- Consumes: `ImageDecodeError`, `UnsupportedMime`, `ImageTooLarge` from Task 1
- Produces: `decode_image(b64, mime, *, max_bytes, allowed_mime) -> ImageInput` — but `ImageInput` is defined in Task 3; for now return a 2-tuple of `(bytes, mime)` and Task 3 will refactor the signature.

**Note to implementer:** Write `decode_image` returning `tuple[bytes, str]` in this task. Task 3 will rename the return type to `ImageInput` and update the helper at the same time. Keep the import boundary clean.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_decoding.py`:

```python
"""decode_image helper: validation, MIME/size limits, error mapping."""
from __future__ import annotations

import base64

import pytest

from vector_service.core.errors import ImageDecodeError, ImageTooLarge, UnsupportedMime
from vector_service.embeddings.image_decoding import decode_image


ALLOWED = {"image/jpeg", "image/png", "image/webp"}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_decode_image_happy_path_returns_bytes_and_mime():
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    data, mime = decode_image(_b64(raw), "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    assert data == raw
    assert mime == "image/png"


def test_decode_image_rejects_unsupported_mime():
    with pytest.raises(UnsupportedMime) as ei:
        decode_image(_b64(b"x"), "image/bmp", max_bytes=1024, allowed_mime=ALLOWED)
    assert ei.value.got == "image/bmp"
    assert "image/png" in ei.value.allowed


def test_decode_image_rejects_oversize():
    raw = b"\x00" * 2048
    with pytest.raises(ImageTooLarge) as ei:
        decode_image(_b64(raw), "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    assert ei.value.got == 2048
    assert ei.value.max == 1024


def test_decode_image_rejects_invalid_base64():
    with pytest.raises(ImageDecodeError) as ei:
        decode_image("!!!not-base64!!!", "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    # Must be ImageDecodeError but NOT UnsupportedMime or ImageTooLarge
    assert not isinstance(ei.value, UnsupportedMime)
    assert not isinstance(ei.value, ImageTooLarge)


def test_decode_image_does_not_touch_pil():
    """The helper must NOT decode bytes into a PIL.Image — that's the
    embedder's job. It only validates and returns raw bytes."""
    raw = b"raw-bytes"
    data, _ = decode_image(_b64(raw), "image/jpeg", max_bytes=1024, allowed_mime=ALLOWED)
    assert data is raw  # identity preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_decoding.py -q`
Expected: `ModuleNotFoundError: No module named 'vector_service.embeddings.image_decoding'`.

- [ ] **Step 3: Implement `decode_image`**

Create `src/vector_service/embeddings/image_decoding.py`:

```python
"""Validate and decode base64 image payloads.

Returns raw bytes; per-model decoding to PIL.Image lives in the embedder
subclass (each model has its own preprocess pipeline).
"""
from __future__ import annotations

import base64
import binascii

from vector_service.core.errors import ImageDecodeError, ImageTooLarge, UnsupportedMime


def decode_image(
    b64: str,
    mime: str,
    *,
    max_bytes: int,
    allowed_mime: set[str],
) -> tuple[bytes, str]:
    """Decode base64 + validate MIME + size.

    Args:
        b64: Base64-encoded image bytes (no data: URI prefix).
        mime: MIME type claim from the caller.
        max_bytes: Maximum decoded byte length.
        allowed_mime: Set of permitted MIME types.

    Returns:
        ``(raw_bytes, mime)`` tuple — ``mime`` is echoed back unchanged
        after the allow-list check.

    Raises:
        UnsupportedMime: ``mime`` not in ``allowed_mime``.
        ImageTooLarge: decoded bytes exceed ``max_bytes``.
        ImageDecodeError: base64 decoding failed (other reasons).
    """
    if mime not in allowed_mime:
        raise UnsupportedMime(
            f"unsupported mime: {mime!r}",
            got=mime,
            allowed=sorted(allowed_mime),
        )
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ImageTooLarge(
            f"image too large: {len(raw)} > {max_bytes}",
            got=len(raw),
            max=max_bytes,
        )
    return raw, mime
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_decoding.py -q`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vector_service/embeddings/image_decoding.py tests/unit/test_image_decoding.py
git commit -m "feat(image): base64/MIME/size decoder with error mapping"
```

---

### Task 3: ImageEmbedder ABC + ImageInput dataclass

**Files:**
- Create: `src/vector_service/embeddings/image_base.py`
- Create: `tests/unit/test_image_embedder_base.py`
- Modify: `src/vector_service/embeddings/image_decoding.py` (update `decode_image` to return `ImageInput`)

**Interfaces:**
- Consumes: `ImageEmbedderError`, `ModelNotLoadedForImages` from Task 1
- Produces: `ImageInput` dataclass (frozen), `ImageEmbedder` ABC with `load()`, `embed_images()`, `embed_query_image()`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_embedder_base.py`:

```python
"""ABC contract for ImageEmbedder implementations."""
from __future__ import annotations

import pytest

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


def test_image_input_is_immutable():
    item = ImageInput(data=b"\x00", mime="image/png")
    with pytest.raises((AttributeError, TypeError)):
        item.data = b"\x01"  # type: ignore[misc]


def test_image_input_carries_bytes_and_mime():
    item = ImageInput(data=b"abc", mime="image/jpeg")
    assert item.data == b"abc"
    assert item.mime == "image/jpeg"


def test_image_embedder_base_load_is_abstract():
    with pytest.raises(TypeError):
        ImageEmbedder()  # type: ignore[abstract]


def test_image_embedder_base_subclass_must_implement_methods():
    class _Half(ImageEmbedder):
        dim = 768
        model_name = "half"

        def load(self):
            return None

        # Missing embed_images and embed_query_image

    with pytest.raises(TypeError):
        _Half()  # type: ignore[abstract]


def test_subclass_can_be_loaded_and_called():
    class _Fake(ImageEmbedder):
        dim = 768
        model_name = "fake"

        def __init__(self):
            self.load_called = 0

        def load(self):
            self.load_called += 1

        def embed_images(self, images):
            return [[0.0] * 768 for _ in images]

        def embed_query_image(self, image):
            return [0.0] * 768

    e = _Fake()
    e.load()
    assert e.load_called == 1
    assert len(e.embed_images([ImageInput(b"x", "image/png"), ImageInput(b"y", "image/png")])) == 2
    assert len(e.embed_query_image(ImageInput(b"x", "image/png"))) == 768


def test_subclass_load_failure_propagates_model_not_loaded_for_images():
    class _Boom(ImageEmbedder):
        dim = 768
        model_name = "boom"

        def load(self):
            raise ModelNotLoadedForImages("weights missing")

        def embed_images(self, images): return []
        def embed_query_image(self, image): return []

    with pytest.raises(ModelNotLoadedForImages):
        _Boom().load()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_embedder_base.py -q`
Expected: `ModuleNotFoundError: No module named 'vector_service.embeddings.image_base'`.

- [ ] **Step 3: Implement the ABC**

Create `src/vector_service/embeddings/image_base.py`:

```python
"""Image embedder abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageInput:
    """A single image ready to embed.

    ``data`` is raw decoded bytes (NOT base64). Per-model preprocessing
    (PIL decode, resize, normalize) happens inside the embedder.
    """

    data: bytes
    mime: str


class ImageEmbedder(ABC):
    """Abstract base for image embedding models.

    Concrete subclasses must:
    - Set ``dim`` (int) and ``model_name`` (str).
    - Implement ``load``, ``embed_images``, and ``embed_query_image``.

    All methods are synchronous. Async dispatch is the caller's job
    (use ``loop.run_in_executor`` from FastAPI routes).
    """

    dim: int
    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Eagerly load the model into memory and run any one-time warmup.

        Called once during application startup (see ``core.lifespan``).
        Subclasses that lazily defer heavy work to first use must
        implement this so the load happens up front and readiness
        probes can observe it. May raise ``ModelNotLoadedForImages``.
        """

    @abstractmethod
    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        """Embed a batch of images. Return one vector per input, in order."""

    @abstractmethod
    def embed_query_image(self, image: ImageInput) -> list[float]:
        """Embed a single query image. May differ from ``embed_images``.

        For models without a distinct query mode (e.g. OpenCLIP), this
        is identical to ``embed_images([image])[0]``.
        """
```

- [ ] **Step 4: Update `decode_image` to return `ImageInput`**

In `src/vector_service/embeddings/image_decoding.py`, change the function signature and return value to use `ImageInput`:

```python
from vector_service.core.errors import ImageDecodeError, ImageTooLarge, UnsupportedMime
from vector_service.embeddings.image_base import ImageInput


def decode_image(
    b64: str,
    mime: str,
    *,
    max_bytes: int,
    allowed_mime: set[str],
) -> ImageInput:
    """Decode base64 + validate MIME + size.

    ... (docstring unchanged)

    Returns:
        ``ImageInput`` carrying raw bytes and the verified MIME type.
    """
    if mime not in allowed_mime:
        raise UnsupportedMime(...)
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(...) from e
    if len(raw) > max_bytes:
        raise ImageTooLarge(...)
    return ImageInput(data=raw, mime=mime)
```

Update `tests/unit/test_image_decoding.py` to use the new return type:

```python
from vector_service.embeddings.image_base import ImageInput
# ...
def test_decode_image_happy_path_returns_image_input():
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    out = decode_image(_b64(raw), "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    assert isinstance(out, ImageInput)
    assert out.data == raw
    assert out.mime == "image/png"


def test_decode_image_does_not_touch_pil():
    raw = b"raw-bytes"
    out = decode_image(_b64(raw), "image/jpeg", max_bytes=1024, allowed_mime=ALLOWED)
    assert out.data is raw  # identity preserved
```

- [ ] **Step 5: Run all image tests to verify they pass**

Run: `pytest tests/unit/test_image_decoding.py tests/unit/test_image_embedder_base.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vector_service/embeddings/image_base.py \
        src/vector_service/embeddings/image_decoding.py \
        tests/unit/test_image_embedder_base.py \
        tests/unit/test_image_decoding.py
git commit -m "feat(image): ImageEmbedder ABC + ImageInput dataclass"
```

---

### Task 4: Image embedding config

**Files:**
- Modify: `src/vector_service/core/config.py` (add `ImageEmbeddingSettings`, attach to `Settings`)
- Create: `tests/unit/test_image_settings.py`

**Interfaces:**
- Consumes: `BaseSettings` from pydantic-settings
- Produces: `Settings.image_embedding: ImageEmbeddingSettings` with all `VS_IMAGE_EMBEDDING__*` env vars parsed

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_settings.py`:

```python
"""ImageEmbeddingSettings env-var parsing + nested Settings wiring."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.core.config import ImageEmbeddingSettings, Settings


def test_image_embedding_settings_defaults():
    s = ImageEmbeddingSettings()
    assert s.backend == "openclip-vit-l-14"
    assert s.device == "auto"
    assert s.batch_size == 16
    assert s.max_images_per_request == 64
    assert s.max_image_bytes == 10 * 1024 * 1024
    assert "image/jpeg" in s.allowed_mime
    assert "image/png" in s.allowed_mime
    assert "image/webp" in s.allowed_mime


def test_image_embedding_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BACKEND", "openclip-vit-l-14")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__DEVICE", "cpu")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BATCH_SIZE", "8")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES", "5242880")
    s = ImageEmbeddingSettings()
    assert s.device == "cpu"
    assert s.batch_size == 8
    assert s.max_image_bytes == 5_242_880


def test_image_embedding_settings_batch_size_bounds():
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(batch_size=0)
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(batch_size=10_000)


def test_image_embedding_settings_max_image_bytes_bounds():
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(max_image_bytes=10)


def test_settings_exposes_image_embedding():
    s = Settings()
    assert hasattr(s, "image_embedding")
    assert isinstance(s.image_embedding, ImageEmbeddingSettings)


def test_settings_env_nested_image_embedding(monkeypatch):
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BATCH_SIZE", "4")
    s = Settings()
    assert s.image_embedding.batch_size == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_settings.py -q`
Expected: `ImportError` or `AttributeError` on `ImageEmbeddingSettings`.

- [ ] **Step 3: Implement `ImageEmbeddingSettings`**

In `src/vector_service/core/config.py`, add the new class above `class Settings`:

```python
class ImageEmbeddingSettings(BaseSettings):
    """Image embedding subsystem configuration.

    Env prefix: ``VS_IMAGE_EMBEDDING__`` (double underscore — pydantic-settings
    nested-field separator).
    """

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

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {v!r}")
        return v
```

Then attach to `Settings` — add a new field alongside `reranker`:

```python
class Settings(BaseSettings):
    ...
    # Image embedding (nested; env prefix VS_IMAGE_EMBEDDING__)
    image_embedding: ImageEmbeddingSettings = Field(default_factory=ImageEmbeddingSettings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_settings.py -q`
Expected: all tests pass.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/unit -q`
Expected: no regression.

- [ ] **Step 6: Commit**

```bash
git add src/vector_service/core/config.py tests/unit/test_image_settings.py
git commit -m "feat(image): ImageEmbeddingSettings with VS_IMAGE_EMBEDDING__ env prefix"
```

---

### Task 5: OpenCLIP ViT-L/14 concrete embedder

**Files:**
- Create: `src/vector_service/embeddings/openclip_vit_l14.py`
- Create: `tests/unit/test_image_embedder_openclip.py`

**Interfaces:**
- Consumes: `ImageEmbedder` ABC, `ImageInput`, `Settings.image_embedding`, `ModelNotLoadedForImages`
- Produces: `OpenCLIPVitL14ImageEmbedder` class with `dim=768`, `model_name="openclip-vit-l-14"`, lazy import of `open_clip`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_embedder_openclip.py`:

```python
"""OpenCLIPVitL14ImageEmbedder with mocked open_clip."""
from __future__ import annotations

from pathlib import Path

import pytest

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageInput


class _FakeImageSettings:
    """Minimal stand-in for ImageEmbeddingSettings — only the attrs the embedder reads."""

    def __init__(
        self,
        *,
        model_dir="./models/openclip-vit-l-14",
        device="cpu",
        batch_size=4,
        auto_download=True,
        download_source="huggingface",
        hf_repo="openai/ViT-L-14",
    ):
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.auto_download = auto_download
        self.download_source = download_source
        self.hf_repo = hf_repo


def _install_open_clip_stub(monkeypatch, encode_return=None):
    """Patch open_clip + huggingface_hub so no real model loads."""
    encode_return = encode_return if encode_return is not None else [[0.0] * 768]

    encode_calls: list[tuple] = []

    def _encode_image(tensor):
        encode_calls.append(tensor)
        # Mimic open_clip: one row per item in the batch.
        n = tensor.shape[0] if hasattr(tensor, "shape") else len(tensor)
        return encode_return * n if isinstance(encode_return, list) and len(encode_return) == n else encode_return[:n]

    class _FakeTensor:
        def __init__(self, n): self.shape = (n,)
        def to(self, device): return self

    fake_model = type("M", (), {"encode_image": lambda self, t: _encode_image(t)})()
    fake_preprocess = lambda pil: object()  # noqa: E731
    fake_tokenize = lambda texts: None  # noqa: E731

    def _create_model_and_transforms(name, pretrained, **kwargs):
        assert name == "ViT-L-14"
        assert pretrained == "openai"
        return fake_model, fake_preprocess, fake_tokenize

    monkeypatch.setattr("vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create_model_and_transforms)
    return encode_calls


def test_openclip_class_metadata():
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    assert OpenCLIPVitL14ImageEmbedder.dim == 768
    assert OpenCLIPVitL14ImageEmbedder.model_name == "openclip-vit-l-14"


def test_openclip_load_calls_create_model_and_warmup(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    encode_calls = _install_open_clip_stub(monkeypatch)
    s = _FakeImageSettings(model_dir=str(tmp_path))
    embedder = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]

    embedder.load()
    # Warmup issues exactly one encode_image call.
    assert len(encode_calls) >= 1


def test_openclip_load_is_idempotent(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    create_calls = []
    def _create(name, pretrained, **kwargs):
        create_calls.append((name, pretrained))
        class _M:
            def encode_image(self, t): return [[0.0] * 768]
        return _M(), lambda pil: object(), lambda texts: None

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create
    )

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()
    e.load()
    assert len(create_calls) == 1


def test_openclip_load_propagates_model_not_loaded_for_images(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    def _boom(*a, **kw):
        raise RuntimeError("disk error")

    monkeypatch.setattr("vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _boom)

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    with pytest.raises(ModelNotLoadedForImages):
        e.load()


def test_embed_images_returns_one_vector_per_input(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    encode_calls = []
    def _create(name, pretrained, **kwargs):
        class _M:
            def encode_image(self, t):
                encode_calls.append(t)
                n = t.shape[0]
                return [[0.1 * i] * 768 for i in range(n)]
        return _M(), lambda pil: object(), lambda texts: None

    monkeypatch.setattr("vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create)

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    inputs = [ImageInput(data=b"\x00", mime="image/png") for _ in range(3)]
    vecs = e.embed_images(inputs)
    assert len(vecs) == 3
    assert all(len(v) == 768 for v in vecs)


def test_embed_query_image_equals_embed_images_of_one(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    def _create(name, pretrained, **kwargs):
        class _M:
            def encode_image(self, t):
                n = t.shape[0]
                return [[float(i)] * 768 for i in range(n)]
        return _M(), lambda pil: object(), lambda texts: None

    monkeypatch.setattr("vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create)

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    img = ImageInput(data=b"\x00", mime="image/png")
    q = e.embed_query_image(img)
    assert q == e.embed_images([img])[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_embedder_openclip.py -q`
Expected: `ModuleNotFoundError: No module named 'vector_service.embeddings.openclip_vit_l14'`.

- [ ] **Step 3: Implement the embedder**

Create `src/vector_service/embeddings/openclip_vit_l14.py`:

```python
"""OpenCLIP ViT-L/14 image embedder (openai pretrained weights, 768d)."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from vector_service.core.config import ImageEmbeddingSettings
from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


# Indirection so tests can monkeypatch without us importing open_clip at
# module load (we want a clean ImportError path if open_clip_torch isn't
# installed in production).
def _create_model_and_transforms(name: str, pretrained: str, **kwargs):
    import open_clip  # lazy

    return open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)


def _snapshot_download(repo_id: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download  # lazy

    return snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)


def _dir_has_model(p: Path) -> bool:
    if not p.exists():
        return False
    return (p / "config.json").exists() or any(p.glob("*.bin")) or any(p.glob("*.safetensors"))


class OpenCLIPVitL14ImageEmbedder(ImageEmbedder):
    dim = 768
    model_name = "openclip-vit-l-14"

    def __init__(self, settings: ImageEmbeddingSettings | None = None):
        if settings is None:
            # Lazy import to avoid pulling core.config when the embedder is
            # used in tests that pass their own settings stub.
            from vector_service.core.config import get_settings

            s = get_settings().image_embedding
        else:
            s = settings
        self._settings = s
        self._device = self._resolve_device(s.device)
        self._batch_size = s.batch_size
        self._model_dir = Path(s.model_dir)
        self._model = None
        self._preprocess = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cuda":
            return _check_cuda()
        if requested == "cpu":
            return "cpu"
        # auto
        try:
            return _check_cuda()
        except Exception:
            return "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._load_internal()

    def load(self) -> None:
        if self._model is not None:
            return
        self._load_internal()

    def _load_internal(self) -> None:
        self._ensure_model_dir()
        try:
            self._model, self._preprocess, _ = _create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=self._device
            )
        except Exception as e:
            raise ModelNotLoadedForImages(
                f"failed to load OpenCLIP ViT-L/14: {e}"
            ) from e

        # Warmup: single image to materialize kernels.
        try:
            self.embed_images([ImageInput(data=b"\x89PNG\r\n\x1a\n", mime="image/png")])
        except Exception as e:  # warmup failures are non-fatal
            import structlog

            structlog.get_logger(__name__).warning("openclip_warmup_failed", error=str(e))

    def _ensure_model_dir(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if _dir_has_model(self._model_dir):
            return
        if not self._settings.auto_download:
            raise ModelNotLoadedForImages(
                f"model dir {self._model_dir} has no OpenCLIP weights and auto_download is off"
            )
        _snapshot_download(self._settings.hf_repo, str(self._model_dir))

    def _preprocess_one(self, image: ImageInput):
        pil = Image.open(io.BytesIO(image.data))
        return self._preprocess(pil)

    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        if not images:
            return []
        self._ensure_loaded()
        import torch  # local import; only needed at inference time

        batched = [self._preprocess_one(img) for img in images]
        tensor = torch.stack(batched).to(self._device)
        with torch.no_grad():
            out = self._model.encode_image(tensor)
        return [list(map(float, row)) for row in out.cpu().tolist()]

    def embed_query_image(self, image: ImageInput) -> list[float]:
        return self.embed_images([image])[0]


def _check_cuda() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError as e:
        raise ModelNotLoadedForImages(f"torch not available: {e}") from e
    raise ModelNotLoadedForImages("CUDA not available")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_embedder_openclip.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vector_service/embeddings/openclip_vit_l14.py \
        tests/unit/test_image_embedder_openclip.py
git commit -m "feat(image): OpenCLIP ViT-L/14 embedder (768d, mocked open_clip)"
```

---

### Task 6: Image embedder registry

**Files:**
- Create: `src/vector_service/embeddings/image_registry.py`
- Create: `tests/unit/test_image_registry.py`

**Interfaces:**
- Consumes: `ImageEmbedderError`, `OpenCLIPVitL14ImageEmbedder`
- Produces: `IMAGE_EMBEDDER_REGISTRY: dict[str, type[ImageEmbedder]]`, `get_image_embedder_class(name)`, `list_image_embedder_names()`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_registry.py`:

```python
"""Image embedder registry lookup."""
from __future__ import annotations

import pytest

from vector_service.core.errors import ImageEmbedderError
from vector_service.embeddings.image_registry import (
    IMAGE_EMBEDDER_REGISTRY,
    get_image_embedder_class,
    list_image_embedder_names,
)


def test_registry_contains_openclip_vit_l_14():
    assert "openclip-vit-l-14" in IMAGE_EMBEDDER_REGISTRY


def test_get_image_embedder_class_returns_registered_class():
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    cls = get_image_embedder_class("openclip-vit-l-14")
    assert cls is OpenCLIPVitL14ImageEmbedder


def test_get_image_embedder_class_unknown_raises():
    with pytest.raises(ImageEmbedderError) as ei:
        get_image_embedder_class("does-not-exist")
    assert "does-not-exist" in str(ei.value)


def test_list_image_embedder_names_is_non_empty():
    names = list_image_embedder_names()
    assert isinstance(names, list)
    assert "openclip-vit-l-14" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_registry.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the registry**

Create `src/vector_service/embeddings/image_registry.py`:

```python
"""Image embedder registry."""
from __future__ import annotations

from vector_service.core.errors import ImageEmbedderError
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_registry.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vector_service/embeddings/image_registry.py tests/unit/test_image_registry.py
git commit -m "feat(image): image embedder registry"
```

---

### Task 7: Image embedding metrics

**Files:**
- Modify: `src/vector_service/core/metrics.py` (add `IMAGE_EMBEDDING_*`, init `MODEL_LOADED.labels(kind="image_embedder")`)
- Modify: `tests/unit/test_image_metrics.py` (new file)

**Interfaces:**
- Consumes: `prometheus_client` (existing)
- Produces: `IMAGE_EMBEDDING_REQUESTS_TOTAL`, `IMAGE_EMBEDDING_DURATION_SECONDS`, `IMAGE_EMBEDDING_INPUTS_TOTAL`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_metrics.py`:

```python
"""Image embedding metrics are registered and labelable."""
from __future__ import annotations

from vector_service.core import metrics


def test_image_embedding_requests_total_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_REQUESTS_TOTAL")
    m = metrics.IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(model="openclip-vit-l-14", status="ok")
    assert m is not None


def test_image_embedding_duration_seconds_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_DURATION_SECONDS")
    h = metrics.IMAGE_EMBEDDING_DURATION_SECONDS.labels(model="openclip-vit-l-14", status="ok")
    h.observe(0.01)


def test_image_embedding_inputs_total_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_INPUTS_TOTAL")
    c = metrics.IMAGE_EMBEDDING_INPUTS_TOTAL.labels(model="openclip-vit-l-14")
    c.inc(3)


def test_model_loaded_kind_image_embedder_initialized():
    """The gauge must have a 0 sentinel for kind=image_embedder so
    /readyz reports 'not_loaded' until the lifespan sets it to 1."""
    sample = metrics.MODEL_LOADED.labels(kind="image_embedder")._value.get()  # type: ignore[attr-defined]
    assert sample == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_metrics.py -q`
Expected: `AttributeError` on the new metric names.

- [ ] **Step 3: Implement the metrics**

In `src/vector_service/core/metrics.py`, after the existing `EMBEDDING_REQUESTS_TOTAL` definition (around line 36) add:

```python
IMAGE_EMBEDDING_REQUESTS_TOTAL = Counter(
    "vs_image_embedding_requests_total",
    "Total image embedding requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

IMAGE_EMBEDDING_DURATION_SECONDS = Histogram(
    "vs_image_embedding_duration_seconds",
    "Image embedding request duration in seconds",
    labelnames=("model", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

IMAGE_EMBEDDING_INPUTS_TOTAL = Counter(
    "vs_image_embedding_inputs_total",
    "Total images processed by image embedder",
    labelnames=("model",),
    registry=REGISTRY,
)
```

In the same file, alongside the existing `MODEL_LOADED.labels(kind=...)` initializers (line 73-74), add:

```python
MODEL_LOADED.labels(kind="image_embedder").set(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_metrics.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vector_service/core/metrics.py tests/unit/test_image_metrics.py
git commit -m "feat(image): image embedding metrics + MODEL_LOADED kind"
```

---

### Task 8: Lifespan wiring + readyz

**Files:**
- Modify: `src/vector_service/core/lifespan.py` (add `build_image_embedder` + lifespan block)
- Modify: `src/vector_service/api/health.py` (add `_image_embedder_loaded`, `/readyz` body field)
- Create: `tests/unit/test_image_lifespan.py`

**Interfaces:**
- Consumes: `ImageEmbedderError`, `ModelNotLoadedForImages`, `ImageEmbeddingSettings`, `IMAGE_EMBEDDING_*` metrics, `get_image_embedder_class`, `OpenCLIPVitL14ImageEmbedder`
- Produces: `build_image_embedder(settings) -> ImageEmbedder`; `app.state.image_embedder` after lifespan; `/readyz` body contains `"image_embedder"` key

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_lifespan.py`:

```python
"""Image embedder lifespan wiring + /readyz reporting."""
from __future__ import annotations

import pytest

from vector_service.core import lifespan as lifespan_mod
from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder


class _RecordingImageEmbedder(ImageEmbedder):
    dim = 4
    model_name = "recording"

    def __init__(self, raise_on_load=None):
        self.load_called = 0
        self._raise = raise_on_load
        self._impl = None  # sentinel pattern

    def load(self):
        self.load_called += 1
        if self._raise is not None:
            raise self._raise
        self._impl = object()

    def embed_images(self, images):
        return [[0.0] * self.dim for _ in images]

    def embed_query_image(self, image):
        return [0.0] * self.dim


class _FakeTextEmbedder:
    model_name = "text"
    dim = 4

    def load(self):
        self.load_called = getattr(self, "load_called", 0) + 1


class _FakeStore:
    backend_name = "fake"
    uri = ""

    def _ensure_connected(self):
        pass

    def list_databases(self):
        return []

    def close(self):
        pass


@pytest.fixture
def patched_lifespan_deps(monkeypatch):
    img_embedder = _RecordingImageEmbedder()
    text_embedder = _FakeTextEmbedder()
    fake_store = _FakeStore()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: text_embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: img_embedder)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: type("R", (), {"model_name": "r", "load": lambda self: None})())

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    return img_embedder, app


def test_lifespan_calls_image_embedder_load(patched_lifespan_deps):
    img, app = patched_lifespan_deps
    from fastapi.testclient import TestClient

    with TestClient(app):
        assert img.load_called == 1


def test_lifespan_sets_image_embedder_on_state(patched_lifespan_deps):
    img, app = patched_lifespan_deps
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        # The embedder is reachable via the app.state we built
        assert app.state.image_embedder is img


def test_readyz_reports_image_embedder_loaded(patched_lifespan_deps):
    _, app = patched_lifespan_deps
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["image_embedder"] == "loaded"


def test_readyz_reports_image_embedder_not_loaded_on_failure(monkeypatch):
    img = _RecordingImageEmbedder(raise_on_load=ModelNotLoadedForImages("disk full"))
    text_embedder = _FakeTextEmbedder()
    fake_store = _FakeStore()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: text_embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: img)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: type("R", (), {"model_name": "r", "load": lambda self: None})())

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["image_embedder"] == "not_loaded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_lifespan.py -q`
Expected: `AttributeError` on `build_image_embedder`, or the `/readyz` body missing `image_embedder`.

- [ ] **Step 3: Implement `build_image_embedder` + lifespan block**

In `src/vector_service/core/lifespan.py`:

Add the import:

```python
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.embeddings.image_base import ImageEmbedder
```

Add the factory function (after `build_reranker`):

```python
def build_image_embedder(settings) -> ImageEmbedder:
    """Construct an image embedder instance for ``settings.image_embedding.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_image_embedder_class(settings.image_embedding.backend)
    return cls(settings=settings.image_embedding)
```

In the `lifespan` async context manager, **before** `yield`, add the load block (after the existing `embedder.load()` block, before the reranker block — or right after the reranker; placement does not matter as long as it's before `yield`):

```python
    # ---- image embedder (parallel to text embedder) ----
    # A failed image-embedder load is logged + surfaced via /readyz (503),
    # but does NOT kill the process — same fail-open policy as the text
    # embedder.
    try:
        image_embedder = build_image_embedder(settings)
        image_embedder.load()
        app.state.image_embedder = image_embedder
        MODEL_LOADED.labels(kind="image_embedder").set(1)
        log.info(
            "image_embedder_loaded",
            model=image_embedder.model_name,
            device=getattr(image_embedder, "_device", "unknown"),
            dim=image_embedder.dim,
        )
    except (ModelNotLoadedForImages, ImageEmbedderError) as exc:
        MODEL_LOADED.labels(kind="image_embedder").set(0)
        log.error(
            "image_embedder_load_failed",
            backend=settings.image_embedding.backend,
            error=str(exc),
        )
```

- [ ] **Step 4: Update `/readyz` to include `image_embedder`**

In `src/vector_service/api/health.py`:

Add a helper after `_reranker_loaded`:

```python
def _image_embedder_loaded(request: Request) -> bool:
    """True iff the image embedder has finished its eager load()."""
    img = getattr(request.app.state, "image_embedder", None)
    if img is None:
        return False
    # Concrete classes follow the `_impl` / `_model` sentinel convention.
    for attr in ("_impl", "_model"):
        if hasattr(img, attr):
            return getattr(img, attr) is not None
    return True
```

Update the `/readyz` handler so the body always contains an `image_embedder` key (and add a 503 path when the image embedder is not loaded, mirroring text):

```python
@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: text embedder loaded AND image embedder loaded AND store reachable."""
    embedder = getattr(request.app.state, "embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    store = getattr(request.app.state, "store", None)
    embedder_ok = _embedder_loaded(embedder)
    image_ok = _image_embedder_loaded(request)
    if store is None:
        body = (
            '{"status":"not_ready","store":"down","embedder":"'
            + ("loaded" if embedder_ok else "not_loaded")
            + '","image_embedder":"'
            + ("loaded" if image_ok else "not_loaded")
            + '","reranker":"'
            + ("ready" if _reranker_loaded(request) else "not_loaded")
            + '"}'
        )
        return Response(content=body, status_code=503, media_type="application/json")
    store_ok = True
    try:
        store.list_databases()
    except Exception:
        store_ok = False

    overall_ok = embedder_ok and image_ok and store_ok
    if overall_ok:
        status = "ready"
    elif (embedder_ok and image_ok) and not store_ok:
        status = "degraded"
    else:
        status = "not_ready"
    code = 200 if overall_ok else 503
    body = (
        '{"status":"' + status
        + '","store":"' + ("ok" if store_ok else "down")
        + '","embedder":"' + ("loaded" if embedder_ok else "not_loaded")
        + '","image_embedder":"' + ("loaded" if image_ok else "not_loaded")
        + '","reranker":"' + ("ready" if _reranker_loaded(request) else "not_loaded")
        + '"}'
    )
    return Response(content=body, status_code=code, media_type="application/json")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_lifespan.py -q`
Expected: all tests pass.

- [ ] **Step 6: Run full unit suite (regression check)**

Run: `pytest tests/unit -q`
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/vector_service/core/lifespan.py src/vector_service/api/health.py \
        tests/unit/test_image_lifespan.py
git commit -m "feat(image): lifespan wiring + /readyz reports image_embedder"
```

---

### Task 9: Image embedding schemas (request / response)

**Files:**
- Create: `src/vector_service/schemas/image_embeddings.py`
- Create: `tests/unit/test_image_embedding_schemas.py`

**Interfaces:**
- Consumes: existing `EmbeddingUsage` from `schemas/openai.py`
- Produces: `ImageInputItem`, `ImageEmbeddingRequest`, `ImageEmbeddingData`, `ImageEmbeddingResponse`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_embedding_schemas.py`:

```python
"""Image embedding request/response schemas."""
from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from vector_service.schemas.image_embeddings import (
    ImageEmbeddingData,
    ImageEmbeddingRequest,
    ImageEmbeddingResponse,
    ImageInputItem,
)


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


def test_image_input_item_accepts_base64_and_mime():
    item = ImageInputItem(data=VALID_PNG_B64, mime="image/png")
    assert item.data == VALID_PNG_B64
    assert item.mime == "image/png"


def test_image_input_item_requires_mime():
    with pytest.raises(ValidationError):
        ImageInputItem(data=VALID_PNG_B64, mime="")  # type: ignore[arg-type]


def test_request_single_image():
    req = ImageEmbeddingRequest(model="openclip-vit-l-14", input=ImageInputItem(data=VALID_PNG_B64, mime="image/png"))
    assert req.model == "openclip-vit-l-14"
    assert isinstance(req.input, ImageInputItem)


def test_request_list_of_images():
    items = [ImageInputItem(data=VALID_PNG_B64, mime="image/png") for _ in range(3)]
    req = ImageEmbeddingRequest(model="openclip-vit-l-14", input=items)
    assert isinstance(req.input, list)
    assert len(req.input) == 3


def test_request_requires_model():
    with pytest.raises(ValidationError):
        ImageEmbeddingRequest(input=ImageInputItem(data=VALID_PNG_B64, mime="image/png"))  # type: ignore[call-arg]


def test_data_object_literal():
    d = ImageEmbeddingData(index=0, embedding=[0.1] * 768)
    assert d.object == "image_embedding"
    assert d.index == 0
    assert len(d.embedding) == 768


def test_response_envelope_shape():
    resp = ImageEmbeddingResponse(
        data=[ImageEmbeddingData(index=0, embedding=[0.0] * 768)],
        model="openclip-vit-l-14",
        usage={"prompt_tokens": 1, "total_tokens": 1},  # type: ignore[arg-type]
    )
    assert resp.object == "list"
    assert resp.model == "openclip-vit-l-14"
    assert resp.usage.prompt_tokens == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_embedding_schemas.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the schemas**

Create `src/vector_service/schemas/image_embeddings.py`:

```python
"""Pydantic schemas for ``POST /v1/image_embeddings``."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vector_service.schemas.openai import EmbeddingUsage


class ImageInputItem(BaseModel):
    """A single base64-encoded image inside an ``input`` payload."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"data": "<base64>", "mime": "image/png"}}
    )

    data: str = Field(
        description="Base64-encoded image bytes (no data: URI prefix).",
    )
    mime: str = Field(
        min_length=1,
        description="Image MIME type; must be one of allowed_mime on the server.",
        examples=["image/png", "image/jpeg", "image/webp"],
    )


class ImageEmbeddingRequest(BaseModel):
    """Request body for ``POST /v1/image_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "openclip-vit-l-14",
                "input": {"data": "<base64>", "mime": "image/png"},
            }
        }
    )

    model: str = Field(
        description="Image embedder model id (see ``GET /v1/models`` filtered by type).",
        examples=["openclip-vit-l-14"],
    )
    input: ImageInputItem | list[ImageInputItem] = Field(
        description="One image or a list of images to embed.",
    )
    encoding_format: Literal["float"] = Field(
        default="float",
        description="Encoding format. Only ``float`` is supported today.",
    )
    user: str | None = None


class ImageEmbeddingData(BaseModel):
    """A single embedding result inside ``ImageEmbeddingResponse.data``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
        }
    )

    object: Literal["image_embedding"] = "image_embedding"
    index: int
    embedding: list[float]


class ImageEmbeddingResponse(BaseModel):
    """Response envelope for ``POST /v1/image_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
                ],
                "model": "openclip-vit-l-14",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        }
    )

    object: Literal["list"] = "list"
    data: list[ImageEmbeddingData]
    model: str
    usage: EmbeddingUsage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_embedding_schemas.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vector_service/schemas/image_embeddings.py tests/unit/test_image_embedding_schemas.py
git commit -m "feat(image): image embedding request/response schemas"
```

---

### Task 10: Extend `Model` schema + models listing route

**Files:**
- Modify: `src/vector_service/schemas/openai.py:165` (widen `Model.type` Literal)
- Modify: `src/vector_service/api/models.py` (include image embedders in the registry view)
- Create: `tests/unit/test_image_models_listing.py`

**Interfaces:**
- Consumes: `Model`, `ModelList`, `IMAGE_EMBEDDER_REGISTRY`, `list_image_embedder_names()`, `get_image_embedder_class()`
- Produces: `GET /v1/models` JSON contains image embedder entries with `type="image_embedder"`, `dimensions=768`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_models_listing.py`:

```python
"""GET /v1/models lists image embedders under type=image_embedder."""
from __future__ import annotations

import pytest

from vector_service.schemas.openai import Model


def test_model_type_accepts_image_embedder():
    m = Model(id="openclip-vit-l-14", type="image_embedder", dimensions=768)
    assert m.type == "image_embedder"
    assert m.dimensions == 768


def test_model_type_rejects_unknown():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Model(id="x", type="unknown_kind")  # type: ignore[arg-type]


def test_models_route_includes_image_embedder(monkeypatch):
    """The /v1/models route must surface the image embedder registry."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.models import router as models_router

    app = FastAPI()
    app.include_router(models_router)
    client = TestClient(app)

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    types = {entry["type"] for entry in body["data"]}
    assert "image_embedder" in types
    ids = {entry["id"]: entry for entry in body["data"]}
    assert "openclip-vit-l-14" in ids
    entry = ids["openclip-vit-l-14"]
    assert entry["type"] == "image_embedder"
    assert entry["dimensions"] == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_models_listing.py -q`
Expected: `ValidationError` on `type="image_embedder"` (Literal too narrow), and `/v1/models` body missing image entries.

- [ ] **Step 3: Widen `Model.type`**

In `src/vector_service/schemas/openai.py`, change line 165 from:

```python
type: Literal["embedder", "reranker"] = Field(
```

to:

```python
type: Literal["embedder", "reranker", "image_embedder"] = Field(
```

Also update the `dimensions` docstring/comment to clarify it applies to both `embedder` and `image_embedder`.

- [ ] **Step 4: Update `api/models.py`**

The current `list_models()` body iterates `list_embedder_names()` then `list_reranker_names()`. Extend it to also iterate `list_image_embedder_names()`. In `src/vector_service/api/models.py`:

Add the import at the top:

```python
from vector_service.embeddings.image_registry import (
    IMAGE_EMBEDDER_REGISTRY,
    list_image_embedder_names,
)
```

Add a third loop inside `list_models()` (after the reranker loop):

```python
    image_embedder = request.app.state.image_embedder
    for name in list_image_embedder_names():
        try:
            dim = image_embedder.dim if (image_embedder and name == image_embedder.model_name) else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="image_embedder", dimensions=dim))
```

Also extend `get_model()` so an image-embedder id is resolvable. Add inside the handler, after the reranker branch:

```python
    if model_id in IMAGE_EMBEDDER_REGISTRY:
        image_embedder = request.app.state.image_embedder
        dim = image_embedder.dim if (image_embedder and image_embedder.model_name == model_id) else None
        return Model(id=model_id, type="image_embedder", dimensions=dim)
```

Also update `registered = sorted(EMBEDDER_REGISTRY) + sorted(list_reranker_names())` to include the image registry names — extend the list:

```python
    registered = sorted(EMBEDDER_REGISTRY) + sorted(list_reranker_names()) + sorted(IMAGE_EMBEDDER_REGISTRY)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_models_listing.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vector_service/schemas/openai.py src/vector_service/api/models.py \
        tests/unit/test_image_models_listing.py
git commit -m "feat(image): Model.type=image_embedder + GET /v1/models lists them"
```

---

### Task 11: Extend management schemas (3-way XOR)

**Files:**
- Modify: `src/vector_service/schemas/management.py` (`UpsertVectorsRequest`, `SearchRequest`)
- Create: `tests/unit/test_management_image_schemas.py`

**Interfaces:**
- Produces: `UpsertVectorsRequest.images`, `.image_mimes`, `.model` fields + 3-way XOR validator; `SearchRequest.query_image`, `.query_image_mime`, `.model` fields + 3-way XOR validator

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_management_image_schemas.py`:

```python
"""Extended UpsertVectorsRequest + SearchRequest accept image inputs."""
from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from vector_service.schemas.management import SearchRequest, UpsertVectorsRequest


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


# ---- upsert ----


def test_upsert_accepts_images():
    req = UpsertVectorsRequest(
        primary_field="id",
        vector_field="vector",
        ids=["a", "b"],
        images=[VALID_PNG_B64, VALID_PNG_B64],
        image_mimes=["image/png", "image/png"],
        model="openclip-vit-l-14",
    )
    assert req.images is not None and len(req.images) == 2


def test_upsert_requires_image_mimes_when_images_present():
    with pytest.raises(ValidationError) as ei:
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            images=[VALID_PNG_B64],
            model="openclip-vit-l-14",
        )
    assert "image_mimes" in str(ei.value)


def test_upsert_requires_model_when_images_present():
    with pytest.raises(ValidationError) as ei:
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            images=[VALID_PNG_B64],
            image_mimes=["image/png"],
        )
    assert "model" in str(ei.value)


def test_upsert_rejects_texts_plus_images():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            texts=["hello"],
            images=[VALID_PNG_B64],
            image_mimes=["image/png"],
            model="openclip-vit-l-14",
        )


def test_upsert_image_mime_length_mismatch():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a", "b"],
            images=[VALID_PNG_B64, VALID_PNG_B64],
            image_mimes=["image/png"],
            model="openclip-vit-l-14",
        )


# ---- search ----


def test_search_accepts_query_image():
    req = SearchRequest(
        primary_field="id",
        vector_field="vector",
        query_image=VALID_PNG_B64,
        query_image_mime="image/png",
        model="openclip-vit-l-14",
        top_k=5,
    )
    assert req.query_image is not None


def test_search_requires_query_image_mime():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_image=VALID_PNG_B64,
            model="openclip-vit-l-14",
        )


def test_search_requires_model_when_query_image_present():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_image=VALID_PNG_B64,
            query_image_mime="image/png",
        )


def test_search_rejects_query_text_plus_query_image():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_text="hello",
            query_image=VALID_PNG_B64,
            query_image_mime="image/png",
            model="openclip-vit-l-14",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_management_image_schemas.py -q`
Expected: `ValidationError`s not raised (fields don't exist yet).

- [ ] **Step 3: Extend `UpsertVectorsRequest`**

In `src/vector_service/schemas/management.py`, add three fields after `vectors`:

```python
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
```

Replace the existing `_xor_texts_vectors` validator body with a 3-way XOR that also enforces `image_mimes` and `model`:

```python
@model_validator(mode="after")
def _xor_three_way(self):
    has_t = self.texts is not None
    has_v = self.vectors is not None
    has_i = self.images is not None
    if (has_t + has_v + has_i) != 1:
        raise ValueError("provide exactly one of texts, vectors, images")
    n = len(self.ids)
    if has_t and len(self.texts) != n:  # type: ignore[arg-type]
        raise ValueError("ids and texts must have same length")
    if has_v and len(self.vectors) != n:  # type: ignore[arg-type]
        raise ValueError("ids and vectors must have same length")
    if has_i:
        if self.image_mimes is None or len(self.image_mimes) != n:
            raise ValueError("image_mimes required, same length as ids")
        if self.model is None:
            raise ValueError("model is required when images is provided")
    if self.fields is not None and len(self.fields) != n:
        raise ValueError("ids and fields must have same length")
    return self
```

- [ ] **Step 4: Extend `SearchRequest`**

In the same file, add three fields after `query_vector`:

```python
query_image: str | None = Field(
    default=None,
    description="Base64-encoded query image. Mutually exclusive with query_text / query_vector.",
)
query_image_mime: str | None = Field(
    default=None,
    description="MIME for query_image. Required when query_image is provided.",
)
```

Replace the existing `_xor_text_vector` validator body with the 3-way version:

```python
@model_validator(mode="after")
def _xor_three_way(self):
    has_t = self.query_text is not None
    has_v = self.query_vector is not None
    has_i = self.query_image is not None
    if (has_t + has_v + has_i) != 1:
        raise ValueError("provide exactly one of query_text, query_vector, query_image")
    if has_i:
        if self.query_image_mime is None:
            raise ValueError("query_image_mime required when query_image is provided")
        if self.model is None:
            raise ValueError("model is required when query_image is provided")
    return self
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_management_image_schemas.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vector_service/schemas/management.py tests/unit/test_management_image_schemas.py
git commit -m "feat(image): 3-way XOR on upsert/search; images + model fields"
```

---

### Task 12: Image embeddings route

**Files:**
- Create: `src/vector_service/api/image_embeddings.py`
- Create: `tests/unit/test_image_embedding_route.py`

**Interfaces:**
- Consumes: `ImageEmbeddingRequest`/`Response` schemas, `ImageEmbedderError`, `ModelNotLoadedForImages`, `UnsupportedMime`, `ImageTooLarge`, `ImageDecodeError`, `decode_image()`, `get_image_embedder_class()`, `IMAGE_EMBEDDING_*` metrics
- Produces: `POST /v1/image_embeddings` route handler

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_image_embedding_route.py`:

```python
"""POST /v1/image_embeddings route behavior."""
from __future__ import annotations

import base64

import pytest

from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
VALID_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode("ascii")
BMP_B64 = base64.b64encode(b"fake-bmp").decode("ascii")
HUGE_B64 = base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode("ascii")


class _StubImageEmbedder(ImageEmbedder):
    """Records calls; returns deterministic vectors."""

    dim = 768
    model_name = "stub"

    def __init__(self):
        self.calls = []

    def load(self): self._impl = object()
    @property
    def _impl(self): return getattr(self, "_impl_obj", None)
    @_impl.setter
    def _impl(self, v): self._impl_obj = v

    def embed_images(self, images):
        self.calls.append(("batch", [i.data for i in images]))
        return [[float(i)] * 768 for i in range(len(images))]

    def embed_query_image(self, image):
        self.calls.append(("query", image.data))
        return [1.0] * 768


@pytest.fixture
def app_with_stub(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vector_service.api.image_embeddings import router as img_router

    stub = _StubImageEmbedder()
    stub.load()  # mark _impl so /readyz-style checks pass

    app = FastAPI()
    app.include_router(img_router)
    app.state.image_embedder = stub
    app.state.settings = type("S", (), {
        "image_embedding": type("IE", (), {
            "max_images_per_request": 2,
            "max_image_bytes": 1024,
            "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
        })(),
    })()

    return app, stub, TestClient(app)


def test_happy_path_single_image(app_with_stub):
    _, stub, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "openclip-vit-l-14"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "image_embedding"
    assert len(body["data"][0]["embedding"]) == 768
    assert body["usage"]["prompt_tokens"] == 1


def test_happy_path_list_of_images(app_with_stub):
    _, stub, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": [
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_JPEG_B64, "mime": "image/jpeg"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2
    assert [d["index"] for d in body["data"]] == [0, 1]


def test_unknown_model_returns_model_not_found(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "does-not-exist",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_unsupported_mime_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": BMP_B64, "mime": "image/bmp"},
    })
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unsupported_mime"


def test_image_too_large_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": HUGE_B64, "mime": "image/png"},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "image_too_large"


def test_too_many_images_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": [
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_PNG_B64, "mime": "image/png"},
        ],
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_many_images"


def test_image_decode_failed_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": "!!!not-base64!!!", "mime": "image/png"},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "image_decode_failed"


def test_embedder_inference_failure_returns_503(app_with_stub):
    app, stub, client = app_with_stub

    class _BrokenEmbedder(_StubImageEmbedder):
        def embed_images(self, images):
            from vector_service.core.errors import ImageEmbedderError
            raise ImageEmbedderError("inference failed")

    app.state.image_embedder = _BrokenEmbedder()
    app.state.image_embedder.load()

    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "image_embedder_unavailable"


def test_metrics_recorded(app_with_stub):
    _, stub, client = app_with_stub
    client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    from vector_service.core import metrics
    sample = metrics.IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(
        model="openclip-vit-l-14", status="ok"
    )._value.get()  # type: ignore[attr-defined]
    assert sample >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_image_embedding_route.py -q`
Expected: `ModuleNotFoundError: No module named 'vector_service.api.image_embeddings'`.

- [ ] **Step 3: Implement the route**

Create `src/vector_service/api/image_embeddings.py`:

```python
"""``POST /v1/image_embeddings`` — image vectorization endpoint."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    ImageDecodeError,
    ImageEmbedderError,
    ImageTooLarge,
    ModelNotLoadedForImages,
    UnsupportedMime,
)
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    IMAGE_EMBEDDING_DURATION_SECONDS,
    IMAGE_EMBEDDING_INPUTS_TOTAL,
    IMAGE_EMBEDDING_REQUESTS_TOTAL,
)
from vector_service.embeddings.image_decoding import decode_image
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.image_embeddings import (
    ImageEmbeddingData,
    ImageEmbeddingRequest,
    ImageEmbeddingResponse,
)
from vector_service.schemas.openai import EmbeddingUsage

router = APIRouter(prefix="/v1", tags=["image_embeddings"])
log = get_logger(__name__)


def _decode_all(items, *, max_bytes, allowed_mime):
    """Decode every input; raise a 422 with failed_indices on any failure."""
    decoded = []
    failed = []
    for i, item in enumerate(items):
        try:
            decoded.append(decode_image(
                item.data, item.mime,
                max_bytes=max_bytes, allowed_mime=allowed_mime,
            ))
        except UnsupportedMime as e:
            failed.append((i, "unsupported_mime", str(e), {"got": e.got, "allowed": e.allowed}))
        except ImageTooLarge as e:
            failed.append((i, "image_too_large", str(e), {"got": e.got, "max": e.max}))
        except ImageDecodeError as e:
            failed.append((i, "image_decode_failed", str(e), {}))
    if failed:
        # All errors in one batch share the same code? Pick the first and aggregate.
        code = failed[0][1]
        if not all(f[1] == code for f in failed):
            code = "image_decode_failed"  # mixed failures collapse to generic
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": code,
                "message": failed[0][2],
                "failed_indices": [f[0] for f in failed],
                "failures": [
                    {"index": f[0], "code": f[1], "message": f[2], **f[3]}
                    for f in failed
                ],
            }},
        )
    return decoded


@router.post(
    "/image_embeddings",
    response_model=ImageEmbeddingResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown image embedder id."},
        422: {"model": ErrorEnvelope, "description": "Image decoding / validation failure."},
        503: {"model": ErrorEnvelope, "description": "Image embedder unavailable."},
    },
    summary="Create image embeddings",
    description=(
        "Embed one or more base64-encoded images using a registered image "
        "embedder. Model id must be one of the image_embedder entries "
        "returned by `GET /v1/models`."
    ),
)
async def create_image_embeddings(body: ImageEmbeddingRequest, request: Request):
    settings = request.app.state.settings.image_embedding
    embedder = request.app.state.image_embedder

    # Validate model id.
    try:
        get_image_embedder_class(body.model)
    except ImageEmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "model_not_found",
            "message": str(e) or f"unknown model {body.model!r}",
            "model": body.model,
        }})

    items = body.input if isinstance(body.input, list) else [body.input]
    if len(items) > settings.max_images_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": "too_many_images",
                "message": (
                    f"got {len(items)} images but the per-request limit is "
                    f"{settings.max_images_per_request}"
                ),
                "max": settings.max_images_per_request,
                "got": len(items),
            }},
        )

    decoded = _decode_all(
        items,
        max_bytes=settings.max_image_bytes,
        allowed_mime=set(settings.allowed_mime),
    )

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"
    try:
        vectors = await loop.run_in_executor(None, embedder.embed_images, decoded)
    except (ImageEmbedderError, ModelNotLoadedForImages) as e:
        status = "error"
        raise HTTPException(status_code=503, detail={"error": {
            "code": "image_embedder_unavailable",
            "message": str(e) or f"image embedder {body.model!r} unavailable",
            "model": body.model,
            "image_count": len(decoded),
            "exception_type": type(e).__name__,
        }})
    finally:
        IMAGE_EMBEDDING_DURATION_SECONDS.labels(model=body.model, status=status).observe(time.perf_counter() - t0)
        IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(model=body.model, status=status).inc()
        IMAGE_EMBEDDING_INPUTS_TOTAL.labels(model=body.model).inc(len(decoded))

    data = [ImageEmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)]
    log.info(
        "image_embedding_request",
        model=body.model,
        image_count=len(decoded),
        duration_ms=int((time.perf_counter() - t0) * 1000),
        status=status,
    )
    return ImageEmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(
            prompt_tokens=len(decoded),
            total_tokens=len(decoded),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_image_embedding_route.py -q`
Expected: all tests pass.

- [ ] **Step 5: Register the router in `main.py`**

In `src/vector_service/main.py`, add the import alongside the other routers (around line 19):

```python
from vector_service.api.image_embeddings import router as image_embeddings_router
```

In `create_app()`, alongside the other `app.include_router(...)` calls (around line 279-285), add:

```python
    app.include_router(image_embeddings_router)
```

Also add a tag entry in `OPENAPI_TAGS` (after the `embeddings` entry, around line 95-101):

```python
    {
        "name": "image_embeddings",
        "description": (
            "Image vectorization under `/v1`. `POST /v1/image_embeddings` "
            "returns dense vectors for a registered image embedder model "
            "from base64-encoded image inputs."
        ),
    },
```

- [ ] **Step 6: Run full unit suite**

Run: `pytest tests/unit -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/vector_service/api/image_embeddings.py src/vector_service/main.py \
        tests/unit/test_image_embedding_route.py
git commit -m "feat(image): POST /v1/image_embeddings route + router registration"
```

---

### Task 13: Extend management routes (upsert + search with images)

**Files:**
- Modify: `src/vector_service/api/management.py` (`upsert_vectors`, `search`)
- Create: `tests/unit/test_management_image_routes.py`

**Interfaces:**
- Consumes: extended `UpsertVectorsRequest`/`SearchRequest`, `decode_image()`, `get_image_embedder_class()`, `ImageEmbedderError`, `ModelNotLoadedForImages`
- Produces: dispatch logic in `upsert_vectors` (images → `embed_images` + `store.upsert`) and `search` (query_image → `embed_query_image` + `store.search`); new `op="upsert_image"` / `op="search_image"` metric labels

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_management_image_routes.py`:

```python
"""upsert_vectors + search with image inputs dispatch to image_embedder."""
from __future__ import annotations

import base64
import types

import pytest

from vector_service.embeddings.image_base import ImageEmbedder, ImageInput
from vector_service.stores.base import Hit


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


class _StubImageEmbedder(ImageEmbedder):
    dim = 768
    model_name = "stub"

    def __init__(self):
        self.batch_calls: list[list[bytes]] = []
        self.query_calls: list[bytes] = []

    def load(self): pass

    def embed_images(self, images):
        self.batch_calls.append([i.data for i in images])
        return [[0.1] * 768 for _ in images]

    def embed_query_image(self, image):
        self.query_calls.append(image.data)
        return [0.5] * 768


class _StubStore:
    backend_name = "stub"

    def upsert(self, db, coll, primary_field, vector_field, ids, vectors, fields):
        self.upsert_args = (db, coll, primary_field, vector_field, ids, vectors, fields)
        return None

    def search(self, db, coll, vector_field, qvec, top_k, filter_expr, output_fields):
        self.search_args = (db, coll, vector_field, qvec, top_k, filter_expr, output_fields)
        return [Hit(id="a", score=0.9, fields={})]

    def list_databases(self): return []
    def close(self): pass


@pytest.fixture
def app_with_image_embedder(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.management import router as mgmt_router

    img = _StubImageEmbedder()
    store = _StubStore()

    app = FastAPI()
    app.include_router(mgmt_router)
    app.state.image_embedder = img
    app.state.embedder = type("E", (), {
        "model_name": "text-stub",
        "embed_documents": lambda self, texts: [[0.0] * 4 for _ in texts],
        "embed_query": lambda self, text: [0.0] * 4,
    })()
    app.state.store = store
    app.state.settings = types.SimpleNamespace(image_embedding=types.SimpleNamespace(
        max_image_bytes=1024 * 1024,
        allowed_mime=["image/jpeg", "image/png", "image/webp"],
    ))

    return app, img, store, TestClient(app)


def test_upsert_with_images_dispatches_to_image_embedder(app_with_image_embedder):
    app, img, store, client = app_with_image_embedder
    r = client.put(
        "/v1/databases/tenant/collections/products/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a", "b"],
            "images": [VALID_PNG_B64, VALID_PNG_B64],
            "image_mimes": ["image/png", "image/png"],
            "model": "openclip-vit-l-14",
            "fields": [{"category": "x"}, {"category": "y"}],
        },
    )
    assert r.status_code == 200, r.text
    assert len(img.batch_calls) == 1
    assert len(img.batch_calls[0]) == 2
    # Vectors reached the store.
    db, coll, pf, vf, ids, vectors, fields = store.upsert_args
    assert db == "tenant" and coll == "products"
    assert ids == ["a", "b"]
    assert len(vectors) == 2 and all(len(v) == 768 for v in vectors)
    assert fields == [{"category": "x"}, {"category": "y"}]


def test_upsert_image_decode_failure_returns_422(app_with_image_embedder):
    _, _, _, client = app_with_image_embedder
    r = client.put(
        "/v1/databases/tenant/collections/products/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "images": ["!!!not-base64!!!"],
            "image_mimes": ["image/png"],
            "model": "openclip-vit-l-14",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "image_decode_failed"


def test_search_with_query_image_dispatches_to_image_embedder(app_with_image_embedder):
    app, img, store, client = app_with_image_embedder
    r = client.post(
        "/v1/databases/tenant/collections/products/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_image": VALID_PNG_B64,
            "query_image_mime": "image/png",
            "model": "openclip-vit-l-14",
            "top_k": 3,
        },
    )
    assert r.status_code == 200, r.text
    assert len(img.query_calls) == 1
    db, coll, vf, qvec, top_k, fe, of = store.search_args
    assert db == "tenant" and coll == "products"
    assert len(qvec) == 768
    assert top_k == 3
    assert r.json()["hits"][0]["id"] == "a"


def test_search_unknown_image_model_returns_404(app_with_image_embedder):
    _, _, _, client = app_with_image_embedder
    r = client.post(
        "/v1/databases/tenant/collections/products/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_image": VALID_PNG_B64,
            "query_image_mime": "image/png",
            "model": "does-not-exist",
        },
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_management_image_routes.py -q`
Expected: requests fail (route ignores `images` / `query_image`) — or the embedder is never called.

- [ ] **Step 3: Extend `upsert_vectors`**

In `src/vector_service/api/management.py`, add the imports:

```python
from vector_service.embeddings.image_decoding import decode_image
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.core.errors import (
    ImageEmbedderError, ModelNotLoadedForImages, UnsupportedMime,
    ImageTooLarge, ImageDecodeError,
)
```

Replace the `if body.texts is not None:` block at the top of `upsert_vectors` with:

```python
    if body.images is not None:
        # Validate image embedder id; raises 404 via ImageEmbedderError mapping.
        try:
            get_image_embedder_class(body.model)  # type: ignore[arg-type]
        except ImageEmbedderError as e:
            raise HTTPException(status_code=404, detail={"error": {
                "code": "model_not_found",
                "message": str(e) or f"unknown model {body.model!r}",
                "model": body.model,
            }})
        image_embedder = request.app.state.image_embedder
        try:
            decoded = [
                decode_image(
                    b64, mime,
                    max_bytes=settings.image_embedding.max_image_bytes,
                    allowed_mime=set(settings.image_embedding.allowed_mime),
                )
                for b64, mime in zip(body.images, body.image_mimes)  # type: ignore[union-attr]
            ]
        except UnsupportedMime as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "unsupported_mime",
                "message": str(e), "got": e.got, "allowed": e.allowed,
            }})
        except ImageTooLarge as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "image_too_large",
                "message": str(e), "got": e.got, "max": e.max,
            }})
        except ImageDecodeError as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "image_decode_failed",
                "message": str(e),
            }})
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(None, image_embedder.embed_images, decoded)
        except (ImageEmbedderError, ModelNotLoadedForImages) as e:
            raise HTTPException(status_code=503, detail={"error": {
                "code": "image_embedder_unavailable",
                "message": str(e) or "image embedder unavailable",
                "model": body.model,
                "image_count": len(decoded),
                "exception_type": type(e).__name__,
            }})
    elif body.texts is not None:
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(None, embedder.embed_documents, body.texts)
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": str(e) or "embedder unavailable",
                "text_count": len(body.texts),
                "exception_type": type(e).__name__,
            }})
    else:
        vectors = body.vectors or []
```

Replace the `await _timed_async("upsert", ...)` call with one that uses `op="upsert_image"` when images were used:

```python
    op_label = "upsert_image" if body.images is not None else "upsert"
    try:
        await _timed_async(
            op_label, _backend(store), db, store.upsert,
            db, name, body.primary_field, body.vector_field,
            body.ids, vectors, body.fields,
        )
    except (DatabaseNotFound, CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(op=op_label, backend=_backend(store), database=db).inc(len(body.ids))
    return {"upserted": len(body.ids)}
```

- [ ] **Step 4: Extend `search`**

Replace the `if body.query_text is not None:` block with:

```python
    if body.query_image is not None:
        try:
            get_image_embedder_class(body.model)  # type: ignore[arg-type]
        except ImageEmbedderError as e:
            raise HTTPException(status_code=404, detail={"error": {
                "code": "model_not_found",
                "message": str(e) or f"unknown model {body.model!r}",
                "model": body.model,
            }})
        image_embedder = request.app.state.image_embedder
        try:
            decoded = decode_image(
                body.query_image, body.query_image_mime,  # type: ignore[arg-type]
                max_bytes=settings.image_embedding.max_image_bytes,
                allowed_mime=set(settings.image_embedding.allowed_mime),
            )
        except UnsupportedMime as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "unsupported_mime",
                "message": str(e), "got": e.got, "allowed": e.allowed,
            }})
        except ImageTooLarge as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "image_too_large",
                "message": str(e), "got": e.got, "max": e.max,
            }})
        except ImageDecodeError as e:
            raise HTTPException(status_code=422, detail={"error": {
                "code": "image_decode_failed",
                "message": str(e),
            }})
        loop = asyncio.get_running_loop()
        try:
            qvec = await loop.run_in_executor(None, image_embedder.embed_query_image, decoded)
        except (ImageEmbedderError, ModelNotLoadedForImages) as e:
            raise HTTPException(status_code=503, detail={"error": {
                "code": "image_embedder_unavailable",
                "message": str(e) or "image embedder unavailable",
                "model": body.model,
                "exception_type": type(e).__name__,
            }})
    elif body.query_text is not None:
        loop = asyncio.get_running_loop()
        try:
            qvec = await loop.run_in_executor(None, embedder.embed_query, body.query_text)
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": str(e) or "embedder unavailable",
                "exception_type": type(e).__name__,
            }})
    else:
        qvec = body.query_vector or []
```

Replace the `await _timed_async("search", ...)` call with:

```python
    op_label = "search_image" if body.query_image is not None else "search"
    try:
        hits = await _timed_async(
            op_label, _backend(store), db, store.search,
            db, name, body.vector_field, qvec, body.top_k,
            body.filter_expr, body.output_fields,
        )
    except (DatabaseNotFound, CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return SearchResponse(
        hits=[HitResponse(id=h.id, score=h.score, fields=h.fields) for h in hits]
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_management_image_routes.py -q`
Expected: all tests pass.

- [ ] **Step 6: Run full unit suite**

Run: `pytest tests/unit -q`
Expected: no regression.

- [ ] **Step 7: Commit**

```bash
git add src/vector_service/api/management.py tests/unit/test_management_image_routes.py
git commit -m "feat(image): upsert/search dispatch on images + query_image"
```

---

### Task 14: pyproject.toml + README + .env.example

**Files:**
- Modify: `pyproject.toml` (add `image-embed` extra + `[all]` includes it)
- Modify: `.env.example` (add `VS_IMAGE_EMBEDDING__*` block; create if missing)
- Modify: `README.md` (add 图像嵌入 section)

**Interfaces:**
- Produces: `pip install -e ".[image-embed]"` works; README documents the endpoint + config; .env.example shows all knobs

- [ ] **Step 1: Update `pyproject.toml`**

In `[project.optional-dependencies]`, after the `embed` block, add:

```toml
image-embed = [
    "open_clip_torch>=2.24",
]
```

In the `all` extra, add `image-embed` to the list:

```toml
all = [
    "vector-service[embed,store,image-embed]",
]
```

- [ ] **Step 2: Update `.env.example`**

If the file exists, append a new section. If it doesn't exist, create it with this content (placeholder format — match the existing layout of the project, omitting other unrelated blocks):

```bash
# ---------- Image embedding (env prefix VS_IMAGE_EMBEDDING__) ----------
VS_IMAGE_EMBEDDING__BACKEND=openclip-vit-l-14
VS_IMAGE_EMBEDDING__MODEL_DIR=./models/openclip-vit-l-14
VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=true
VS_IMAGE_EMBEDDING__DEVICE=auto
VS_IMAGE_EMBEDDING__BATCH_SIZE=16
VS_IMAGE_EMBEDDING__MAX_IMAGES_PER_REQUEST=64
VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES=10485760
VS_IMAGE_EMBEDDING__HF_REPO=openai/ViT-L-14
```

- [ ] **Step 3: Update `README.md`**

Insert a new section after the existing Rerank section (around line 332) and before the closing `## 添加新 reranker 后端` heading:

````markdown
## 图像嵌入（Image Embeddings）

图像嵌入子系统把 base64 编码的图片转成 `list[float]`。当前内置 `openclip-vit-l-14`（OpenCLIP ViT-L/14，openai 预训练权重，768 维）。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/image_embeddings` | 图像嵌入：接收 base64 图片，返回 `{index, embedding}` 列表 |

### 启动

首次启动会自动从 HuggingFace 下载 `openai/ViT-L-14` 权重到 `./models/openclip-vit-l-14/`；离线环境把 `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_IMAGE_EMBEDDING__MODEL_DIR` 指定的目录。

所有 `VS_IMAGE_EMBEDDING__*` 配置项见 `.env.example` 的 `Image embedding` 段（`backend` / `model_dir` / `auto_download` / `device` / `batch_size` / `max_images_per_request` / `max_image_bytes` / `hf_repo` / `allowed_mime`）。

### 示例

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

### 图搜图：upsert + search 接入

`PUT /v1/databases/{db}/collections/{coll}/vectors` 和 `POST .../search` 同时接受文本、向量和图像三种输入（三选一）。图像模式下：

- `PUT`：body 用 `images`（base64 列表）+ `image_mimes`（并行）+ `model`（图像嵌入器 id）代替 `texts`。
- `search`：body 用 `query_image`（base64 单张）+ `query_image_mime` + `model` 代替 `query_text`。

```bash
# 1) 建 collection（dim 必须等于 768）
curl -X POST localhost:8080/v1/databases/tenant-a/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "products",
    "primary_field": "id",
    "scalar_fields": [{"name": "id", "dtype": "varchar", "is_primary": true, "max_length": 64}],
    "vector_field": {"name": "vector", "dim": 768, "metric_type": "cosine"}
  }'

# 2) upsert 图片
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

# 3) 图搜图
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

### 错误码

| HTTP | code | 触发 |
|------|------|------|
| 404 | `model_not_found` | 图像嵌入器 id 未在 `IMAGE_EMBEDDER_REGISTRY` 注册 |
| 422 | `image_decode_failed` | base64 解码失败 |
| 422 | `image_too_large` | 解码后字节数超过 `VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES` |
| 422 | `too_many_images` | 输入图片数超过 `VS_IMAGE_EMBEDDING__MAX_IMAGES_PER_REQUEST` |
| 422 | `unsupported_mime` | MIME 不在 `VS_IMAGE_EMBEDDING__ALLOWED_MIME` |
| 503 | `image_embedder_unavailable` | 启动期权重加载失败 / 推理失败 |
| 500 | `internal` | 未捕获的兜底异常 |

## 添加新图像嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `ImageEmbedder` ABC
2. 在 `embeddings/image_registry.py` 注册
3. 如需新配置项，扩展 `core/config.py` 的 `ImageEmbeddingSettings`
````

Also update the README header table (around line 16-18) to mention image embedding as a supported component, and update the architecture bullet list (around line 12) to mention image embedding.

- [ ] **Step 4: Verify the install command works (smoke check)**

Run: `pip install -e ".[image-embed]"` (or `uv pip install -e ".[image-embed]"`)

Expected: install succeeds. If open_clip_torch isn't already in your environment, this will download it. If you can't run the install, run `python -c "import vector_service.embeddings.openclip_vit_l14 as m; assert hasattr(m, 'OpenCLIPVitL14ImageEmbedder')"` instead to confirm the module imports cleanly.

- [ ] **Step 5: Run the full test suite one last time**

Run: `pytest tests/unit -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example README.md
git commit -m "docs(image): pyproject extra, README, .env.example for image embeddings"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Covered by |
|--------------|------------|
| §3 Architecture (parallel ABC + registry) | Tasks 3, 5, 6 |
| §4.1 ImageEmbedder ABC + ImageInput | Task 3 |
| §4.2 Registry | Task 6 |
| §4.3 OpenCLIP concrete | Task 5 |
| §4.4 Image decoding helper | Task 2 |
| §4.5 Lifespan hook | Task 8 |
| §4.6 Config | Task 4 |
| §4.7 Schemas (request/response) | Task 9 |
| §4.8 Model schema widen | Task 10 |
| §4.9 Management schemas (3-way XOR) | Task 11 |
| §4.10 Image embedding route | Task 12 |
| §4.11 Management route extensions | Task 13 |
| §4.12 Metrics | Task 7 |
| §4.13 Errors (5 codes) | Tasks 1, 12, 13 |
| §5 Data flow (3 paths) | Tasks 12, 13 |
| §6 Testing (unit + contract) | Tasks 1–13 |
| §7 Dependencies (`open_clip_torch` + README + .env) | Task 14 |

All spec sections covered. ✅

**2. Placeholder scan:** None. Every test contains concrete code; every implementation step shows the exact lines. ✅

**3. Type consistency:**

| Symbol | Defined in | Used in |
|--------|-----------|---------|
| `ImageInput` | Task 3 (`embeddings/image_base.py`) | Tasks 5, 9, 12, 13 |
| `ImageEmbedder` | Task 3 | Tasks 5, 6, 8, 12, 13 |
| `ImageEmbedderError` / `ModelNotLoadedForImages` / `UnsupportedMime` / `ImageTooLarge` / `ImageDecodeError` | Task 1 (`core/errors.py`) | Tasks 2, 3, 8, 12, 13 |
| `ImageEmbeddingSettings` | Task 4 (`core/config.py`) | Tasks 5, 8, 12, 13 |
| `IMAGE_EMBEDDING_REQUESTS_TOTAL` / `..._DURATION_SECONDS` / `..._INPUTS_TOTAL` | Task 7 (`core/metrics.py`) | Task 12 |
| `decode_image` | Task 2 → updated Task 3 | Tasks 12, 13 |
| `OpenCLIPVitL14ImageEmbedder` | Task 5 (`embeddings/openclip_vit_l14.py`) | Tasks 6, 10 |
| `IMAGE_EMBEDDER_REGISTRY` / `get_image_embedder_class` / `list_image_embedder_names` | Task 6 | Tasks 10, 12, 13 |
| `build_image_embedder` / `app.state.image_embedder` | Task 8 | Tasks 12, 13 |
| `ImageEmbeddingRequest` / `Data` / `Response` | Task 9 | Task 12 |
| `UpsertVectorsRequest.images` / `image_mimes` / `model` | Task 11 | Tasks 13 |
| `SearchRequest.query_image` / `query_image_mime` / `model` | Task 11 | Tasks 13 |

All symbols traced consistently. ✅

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-12-image-embeddings.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?