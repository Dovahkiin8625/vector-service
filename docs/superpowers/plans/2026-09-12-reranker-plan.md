# Reranker Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-encoder reranker subsystem to vector-service, exposing `POST /v1/rerank` and `GET /v1/rerank/models`, mirroring the existing `Embedder` / `VectorStore` ABC+Registry pattern.

**Architecture:** New top-level package `rerankers/` parallel to `embeddings/` and `stores/`. `Reranker` ABC + `ScoredHit` dataclass + module-level `RERANKER_REGISTRY`. Single concrete implementation `CrossEncoderReranker` wrapping `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3` default. Settings nested under `Settings.reranker` (env `VS_RERANKER__*`). Eager load in `lifespan.py`, instance on `app.state.reranker`.

**Tech Stack:** Python ≥3.11, FastAPI, pydantic-settings, sentence-transformers ≥2.6 (transitive torch/transformers already in `[embed]` extra), structlog, Prometheus client.

**Spec:** `docs/superpowers/specs/2026-09-12-reranker-design.md`

## Global Constraints

- Test command: `pytest tests/unit -q` (Windows-default; contract tests skipped).
- No new top-level dependencies. Only `pyproject.toml` `[embed]` extra adds `sentence-transformers>=2.6` (already installed in venv).
- No shared `tests/conftest.py` — each test file builds its own `FastAPI()` + middleware + handlers + state. Use `_make_app(...)` helper inside each test file.
- All new files use `from __future__ import annotations` at the top (project convention).
- Error envelope is `ErrorEnvelope` from `vector_service.schemas.errors` — never invent new shapes.
- Default model name string: `"bge-reranker-v2-m3"` (the registry key, **not** `"BAAI/bge-reranker-v2-m3"`). The HF/MS repo id is stored separately in settings.
- Default settings: `device="auto"`, `batch_size=32`, `max_length=512`, `max_documents_per_request=256`, `max_chars_per_doc=8192`, `max_query_chars=2048`, `max_top_n=64`, `top_n_default=10`, `download_source="modelscope"`.
- `MODEL_LOADED` Prometheus metric switches from no-label to `labels=["kind"]`; initialise both `kind="embedder"` and `kind="reranker"` to 0 at module import so `/metrics` output is stable.
- `/readyz` `status` still gates only on `embedder` + `store`; `reranker` field is diagnostic only and does NOT block ready.
- Commit messages: conventional-commits style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), body explains why.

---

## Task 1: Errors + Metrics foundation

**Files:**
- Modify: `src/vector_service/core/errors.py`
- Modify: `src/vector_service/core/metrics.py`

**Interfaces:**
- Produces (errors): `class RerankerError(VectorServiceError)`, `class RerankerNotLoaded(RerankerError)` — both with `__init__(message: str = "")`.
- Produces (metrics):
  - `RERANK_DURATION_SECONDS: Histogram` with `labels=["model"]`, buckets `(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)`
  - `RERANK_REQUESTS_TOTAL: Counter` with `labels=["model", "status"]`
  - `MODEL_LOADED` redefined with `labels=["kind"]`; values initialised: `MODEL_LOADED.labels(kind="embedder").set(0)`, `MODEL_LOADED.labels(kind="reranker").set(0)`.

- [ ] **Step 1: Read existing errors.py and metrics.py**

  Read `src/vector_service/core/errors.py` and `src/vector_service/core/metrics.py` to confirm the existing exception base class and metric registration patterns.

- [ ] **Step 2: Add `RerankerError` and `RerankerNotLoaded`**

  In `src/vector_service/core/errors.py`, append after the existing classes (preserve ordering; new classes go after `BackendError`):

  ```python
  class RerankerError(VectorServiceError):
      """Base class for reranker failures."""


  class RerankerNotLoaded(RerankerError):
      """Reranker weights are not loaded (lifespan failed or skipped)."""
  ```

  Verify `VectorServiceError.__init__(self, message="")` accepts a default — if not, add `def __init__(self, message: str = "")` to the new subclasses explicitly.

- [ ] **Step 3: Switch `MODEL_LOADED` to labelled, init both kinds**

  In `src/vector_service/core/metrics.py`, locate the current `MODEL_LOADED = Gauge(...)` definition (currently no labels). Replace with:

  ```python
  MODEL_LOADED = Gauge(
      "vs_model_loaded",
      "Model load state, labelled by kind (1 = loaded, 0 = not loaded).",
      labels=["kind"],
  )
  MODEL_LOADED.labels(kind="embedder").set(0)
  MODEL_LOADED.labels(kind="reranker").set(0)
  ```

  Search the codebase for any other `MODEL_LOADED.set(` or `MODEL_LOADED.labels(...)` calls — there is at least one in `src/vector_service/api/health.py` and one in `core/lifespan.py`. Update those call sites to `MODEL_LOADED.labels(kind="embedder").set(...)`. Do NOT add a `kind="reranker"` call yet — that's Task 7.

- [ ] **Step 4: Add the two new rerank metrics**

  In `src/vector_service/core/metrics.py`, append after the existing metric definitions:

  ```python
  RERANK_DURATION_SECONDS = Histogram(
      "vs_rerank_duration_seconds",
      "Rerank latency in seconds (route handler end-to-end).",
      labels=["model"],
      buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
  )

  RERANK_REQUESTS_TOTAL = Counter(
      "vs_rerank_requests_total",
      "Total rerank requests, labelled by model and outcome.",
      labels=["model", "status"],  # status ∈ {"received", "ok", "error"}
  )
  ```

- [ ] **Step 5: Run existing tests**

  Run: `pytest tests/unit -q`
  Expected: PASS (no behaviour change yet; just metric label switch and new symbols).

- [ ] **Step 6: Commit**

  ```bash
  git add src/vector_service/core/errors.py src/vector_service/core/metrics.py
  # plus any file touched by the MODEL_LOADED.labels(...) update (likely
  # src/vector_service/api/health.py and src/vector_service/core/lifespan.py)
  git commit -m "feat(reranker): add RerankerError and rerank Prometheus metrics

  MODEL_LOADED switches from label-less Gauge to labels=['kind'] so
  embedder and reranker can coexist on the same metric. Both kinds
  initialised to 0 at import time so /metrics output is stable."
  ```

---

## Task 2: Settings — `RerankerSettings` + nested on `Settings`

**Files:**
- Modify: `src/vector_service/core/config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `class RerankerSettings(BaseSettings)` with `model_config = SettingsConfigDict(env_prefix="VS_RERANKER__", extra="ignore")` and all fields from spec §5.1.
- Produces: `Settings.reranker: RerankerSettings = Field(default_factory=RerankerSettings)`.

- [ ] **Step 1: Read `core/config.py` to confirm `BaseSettings` import + `Settings` shape**

- [ ] **Step 2: Add `RerankerSettings` class above `Settings`**

  In `src/vector_service/core/config.py`, add the imports (only add what's missing — `Literal`, `Field`, `field_validator`, `SettingsConfigDict`):

  ```python
  from typing import Literal
  from pydantic import Field, field_validator
  from pydantic_settings import SettingsConfigDict
  ```

  Then add the class above `Settings`:

  ```python
  class RerankerSettings(BaseSettings):
      """Reranker subsystem configuration.

      Env prefix: ``VS_RERANKER__`` (double underscore — pydantic-settings
      nested-field separator).
      """

      model_config = SettingsConfigDict(env_prefix="VS_RERANKER__", extra="ignore")

      backend: str = Field(
          ..., description="Reranker backend name; must exist in RERANKER_REGISTRY."
      )

      # Model identity
      model_name: str = "BAAI/bge-reranker-v2-m3"
      model_dir: str = "./models/bge-reranker-v2-m3"
      auto_download: bool = True
      download_source: Literal["huggingface", "modelscope"] = "modelscope"
      hf_repo: str = "BAAI/bge-reranker-v2-m3"
      ms_repo: str = "BAAI/bge-reranker-v2-m3"

      # Inference
      device: Literal["auto", "cpu", "cuda"] = "auto"
      batch_size: int = Field(32, ge=1, le=512)
      max_length: int = Field(512, ge=1, le=8192)

      # Input limits
      max_documents_per_request: int = Field(256, ge=1, le=4096)
      max_chars_per_doc: int = Field(8192, ge=1, le=32768)
      max_query_chars: int = Field(2048, ge=1, le=8192)
      max_top_n: int = Field(64, ge=1, le=1024)
      top_n_default: int = Field(10, ge=1, le=1024)

      @field_validator("top_n_default")
      @classmethod
      def _top_n_default_le_max(cls, v: int, info) -> int:
          max_top_n = info.data.get("max_top_n", 64)
          if v > max_top_n:
              raise ValueError(
                  f"top_n_default ({v}) must be <= max_top_n ({max_top_n})"
              )
          return v
  ```

- [ ] **Step 3: Nest `reranker` on `Settings`**

  In the same file, inside `class Settings(BaseSettings):`, add (place after the other top-level fields):

  ```python
      reranker: RerankerSettings = Field(default_factory=RerankerSettings)
  ```

- [ ] **Step 4: Update `.env.example`**

  Append to `.env.example` (keep existing content intact):

  ```bash
  # ---- Reranker ----
  # Required: backend name registered in RERANKER_REGISTRY.
  VS_RERANKER__BACKEND=bge-reranker-v2-m3
  VS_RERANKER__MODEL_DIR=./models/bge-reranker-v2-m3
  VS_RERANKER__DEVICE=auto
  VS_RERANKER__BATCH_SIZE=32
  VS_RERANKER__MAX_DOCUMENTS_PER_REQUEST=256
  VS_RERANKER__MAX_QUERY_CHARS=2048
  VS_RERANKER__MAX_CHARS_PER_DOC=8192
  VS_RERANKER__TOP_N_DEFAULT=10
  VS_RERANKER__MAX_TOP_N=64
  VS_RERANKER__DOWNLOAD_SOURCE=modelscope
  ```

- [ ] **Step 5: Sanity check by importing Settings**

  Run: `uv run python -c "from vector_service.core.config import Settings; s = Settings(); print(s.reranker.backend, s.reranker.top_n_default)"`
  Expected: prints `bge-reranker-v2-m3 10` (or whatever backend is in your `.env`; the defaults if no env).

  If `backend` is required and missing env var, this raises a validation error — that is correct.

- [ ] **Step 6: Commit**

  ```bash
  git add src/vector_service/core/config.py .env.example
  git commit -m "feat(reranker): add RerankerSettings nested under Settings

  Pure-env config (VS_RERANKER__* prefix). backend is required at
  startup; all other fields have safe defaults aligned with the
  embeddings input limits."
  ```

---

## Task 3: `Reranker` ABC + `ScoredHit`

**Files:**
- Create: `src/vector_service/rerankers/__init__.py`
- Create: `src/vector_service/rerankers/base.py`

**Interfaces:**
- Produces:
  - `class Reranker(ABC)` with class attr `model_name: str` and abstract methods `load(self) -> None` and `rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[ScoredHit]`.
  - `@dataclass(frozen=True) class ScoredHit` with fields `index: int` and `score: float`.

- [ ] **Step 1: Create the package marker**

  Create `src/vector_service/rerankers/__init__.py`:

  ```python
  """Reranker subsystem: ABC, registry, and concrete backends."""
  from __future__ import annotations
  ```

- [ ] **Step 2: Create `base.py`**

  Create `src/vector_service/rerankers/base.py`:

  ```python
  """Reranker abstract base class and result dataclass."""
  from __future__ import annotations

  from abc import ABC, abstractmethod
  from dataclasses import dataclass


  @dataclass(frozen=True)
  class ScoredHit:
      """A single document with a relevance score from the reranker.

      ``index`` is the position of the document in the input ``documents``
      list — never a database primary key. ``score`` is whatever the
      underlying reranker produces (typically a sigmoid-normalised logit
      in [0, 1] for sentence-transformers CrossEncoder, but the value
      is NOT guaranteed normalised; clients must not assume a range).
      """

      index: int
      score: float


  class Reranker(ABC):
      """Abstract base for reranker backends.

      The contract mirrors ``Embedder``: concrete subclasses expose a
      ``model_name`` class attribute and implement ``load()`` (eager,
      idempotent, called from ``lifespan``) plus ``rerank()``.

      Implementations are CPU/GPU-bound and are expected to be called from
      a thread executor by the route layer — they themselves do NOT need
      to be async.
      """

      model_name: str  # class attribute; e.g. "bge-reranker-v2-m3"

      @abstractmethod
      def load(self) -> None:
          """Load weights / connect to backend. Idempotent."""

      @abstractmethod
      def rerank(
          self,
          query: str,
          documents: list[str],
          top_n: int | None = None,
      ) -> list[ScoredHit]:
          """Return ``documents`` reordered by relevance.

          The returned list MUST be sorted by ``score`` descending.
          If ``top_n`` is given, the list is truncated to at most
          ``top_n`` entries; ``None`` means return everything.
          """
  ```

- [ ] **Step 3: Smoke-test the imports**

  Run: `uv run python -c "from vector_service.rerankers.base import Reranker, ScoredHit; print(Reranker, ScoredHit); Reranker()"`
  Expected: prints the two classes, then the last line raises `TypeError: Can't instantiate abstract class Reranker with abstract methods load, rerank`.

- [ ] **Step 4: Commit**

  ```bash
  git add src/vector_service/rerankers/__init__.py src/vector_service/rerankers/base.py
  git commit -m "feat(reranker): add Reranker ABC and ScoredHit dataclass"
  ```

---

## Task 4: Registry

**Files:**
- Create: `src/vector_service/rerankers/registry.py`

**Interfaces:**
- Produces:
  - `RERANKER_REGISTRY: dict[str, type[Reranker]]` (empty at this task — populated in Task 5)
  - `def get_reranker_class(name: str) -> type[Reranker]` — raises `KeyError` on unknown.
  - `def list_reranker_names() -> list[str]`

- [ ] **Step 1: Create `registry.py`**

  Create `src/vector_service/rerankers/registry.py`:

  ```python
  """Reranker registry: name → implementation class.

  Mirrors ``embeddings/registry.py``. Concrete classes are registered
  here to avoid forcing every backend's import at package import time.
  """
  from __future__ import annotations

  from vector_service.rerankers.base import Reranker

  RERANKER_REGISTRY: dict[str, type[Reranker]] = {}

  def get_reranker_class(name: str) -> type[Reranker]:
      """Return the registered reranker class for ``name``.

      Raises ``KeyError`` if no reranker is registered under that name.
      """
      try:
          return RERANKER_REGISTRY[name]
      except KeyError as exc:
          raise KeyError(
              f"unknown reranker backend: {name!r}; "
              f"registered: {sorted(RERANKER_REGISTRY)}"
          ) from exc

  def list_reranker_names() -> list[str]:
      """Return all registered reranker backend names (sorted by insertion)."""
      return list(RERANKER_REGISTRY)
  ```

  Note: `RERANKER_REGISTRY` is intentionally empty here. Task 5 will populate it after `CrossEncoderReranker` is defined.

- [ ] **Step 2: Verify imports + empty registry**

  Run: `uv run python -c "from vector_service.rerankers.registry import RERANKER_REGISTRY, list_reranker_names; print(RERANKER_REGISTRY, list_reranker_names())"`
  Expected: prints `{} []`.

- [ ] **Step 3: Commit**

  ```bash
  git add src/vector_service/rerankers/registry.py
  git commit -m "feat(reranker): add RERANKER_REGISTRY with empty entries

  Concrete implementations will register themselves after definition
  in follow-up tasks. Mirrors embeddings/registry.py shape."
  ```

---

## Task 5: `CrossEncoderReranker` implementation

**Files:**
- Create: `src/vector_service/rerankers/cross_encoder.py`
- Modify: `src/vector_service/rerankers/registry.py` (populate registry)
- Modify: `pyproject.toml` (declare `sentence-transformers>=2.6`)

**Interfaces:**
- Produces: `class CrossEncoderReranker(Reranker)` with `model_name = "bge-reranker-v2-m3"`.
- Modifies: `RERANKER_REGISTRY["bge-reranker-v2-m3"] = CrossEncoderReranker`.

- [ ] **Step 1: Add `sentence-transformers` to `[embed]` extra in `pyproject.toml`**

  In `pyproject.toml`, under `[project.optional-dependencies]`, find the `embed = [` block. Add (last entry):

  ```toml
      "sentence-transformers>=2.6",
  ```

  Run `uv lock` to refresh `uv.lock`:

  ```bash
  uv lock
  ```

  Expected: `uv.lock` updated; no new top-level deps since sentence-transformers is already installed transitively in the venv.

- [ ] **Step 2: Create `cross_encoder.py`**

  Create `src/vector_service/rerankers/cross_encoder.py`:

  ```python
  """Cross-encoder reranker backed by sentence-transformers ``CrossEncoder``."""
  from __future__ import annotations

  import threading
  from typing import TYPE_CHECKING

  from vector_service.core.config import Settings, get_settings
  from vector_service.core.errors import ModelNotLoaded, RerankerNotLoaded
  from vector_service.rerankers.base import Reranker, ScoredHit

  if TYPE_CHECKING:  # pragma: no cover
      from sentence_transformers import CrossEncoder


  def _dir_has_model(path: str) -> bool:
      """True if ``path`` already contains usable model files.

      Checks for any of ``config.json``, ``tokenizer_config.json``,
      ``model.safetensors``, ``pytorch_model.bin``, or ``model.onnx``.
      This matches the same convention used by ``BGEM3Embedder``.
      """
      import os

      if not os.path.isdir(path):
          return False
      markers = {
          "config.json",
          "tokenizer_config.json",
          "model.safetensors",
          "pytorch_model.bin",
          "model.onnx",
      }
      try:
          present = set(os.listdir(path))
      except OSError:
          return False
      return bool(markers & present)


  def _resolve_device(device: str) -> str:
      """Map ``auto|cpu|cuda`` to a concrete torch device string.

      Mirrors ``BGEM3Embedder._resolve_device``: on ``auto``, try CUDA
      first; if import or probing fails, fall back to CPU.
      """
      if device == "cpu":
          return "cpu"
      if device == "cuda":
          return "cuda"
      # auto
      try:
          import torch

          return "cuda" if torch.cuda.is_available() else "cpu"
      except Exception:
          return "cpu"


  class CrossEncoderReranker(Reranker):
      """sentence-transformers CrossEncoder reranker."""

      model_name = "bge-reranker-v2-m3"

      def __init__(self, settings: Settings | None = None) -> None:
          s = (settings or get_settings()).reranker
          self._device: str = _resolve_device(s.device)
          self._batch_size: int = s.batch_size
          self._max_length: int = s.max_length
          self._model_dir: str = s.model_dir
          self._auto_download: bool = s.auto_download
          self._download_source: str = s.download_source
          self._hf_repo: str = s.hf_repo
          self._ms_repo: str = s.ms_repo
          self._impl: CrossEncoder | None = None
          self._lock = threading.Lock()

      # ---- lifecycle --------------------------------------------------

      def load(self) -> None:
          """Eagerly load weights. Idempotent."""
          with self._lock:
              if self._impl is not None:
                  return
              self._ensure_model_dir()
              from sentence_transformers import CrossEncoder  # local import

              self._impl = CrossEncoder(
                  self._model_dir,
                  max_length=self._max_length,
                  device=self._device,
              )
              # warmup so first request isn't a cold start
              try:
                  self._impl.predict(
                      [("warmup", "")] * 5,
                      batch_size=self._batch_size,
                      show_progress_bar=False,
                  )
              except Exception as exc:  # pragma: no cover — defensive
                  self._impl = None
                  raise RerankerNotLoaded(
                      f"warmup failed for {self.model_name}: {exc}"
                  ) from exc

      def _ensure_model_dir(self) -> None:
          """Download weights if needed; otherwise raise ``RerankerNotLoaded``."""
          if _dir_has_model(self._model_dir):
              return
          if not self._auto_download:
              raise RerankerNotLoaded(
                  f"model dir {self._model_dir!r} missing required files "
                  f"and auto_download is disabled"
              )
          if self._download_source == "modelscope":
              try:
                  from modelscope import snapshot_download
              except ImportError as exc:  # pragma: no cover
                  raise RerankerNotLoaded(
                      "modelscope is not installed; install the [embed] extra "
                      "or switch download_source to 'huggingface'"
                  ) from exc
              snapshot_download(
                  self._ms_repo, local_dir=self._model_dir
              )
          else:  # huggingface
              try:
                  from huggingface_hub import snapshot_download
              except ImportError as exc:  # pragma: no cover
                  raise RerankerNotLoaded(
                      "huggingface-hub is not installed"
                  ) from exc
              snapshot_download(
                  self._hf_repo, local_dir=self._model_dir
              )
          if not _dir_has_model(self._model_dir):  # pragma: no cover
              raise RerankerNotLoaded(
                  f"download completed but {self._model_dir!r} still lacks "
                  f"required files"
              )

      def _ensure_loaded(self) -> None:
          if self._impl is None:
              raise RerankerNotLoaded(
                  f"reranker {self.model_name!r} is not loaded; "
                  f"call load() first"
              )

      # ---- inference --------------------------------------------------

      def rerank(
          self,
          query: str,
          documents: list[str],
          top_n: int | None = None,
      ) -> list[ScoredHit]:
          self._ensure_loaded()
          assert self._impl is not None  # for type checkers
          pairs = [(query, d) for d in documents]
          raw = self._impl.predict(
              pairs, batch_size=self._batch_size, show_progress_bar=False
          )
          # raw may be numpy array or list — coerce to plain floats
          scores = [float(s) for s in raw]
          hits = sorted(
              [ScoredHit(index=i, score=s) for i, s in enumerate(scores)],
              key=lambda h: h.score,
              reverse=True,
          )
          if top_n is not None:
              hits = hits[:top_n]
          return hits


  # Register self
  from vector_service.rerankers.registry import RERANKER_REGISTRY

  RERANKER_REGISTRY["bge-reranker-v2-m3"] = CrossEncoderReranker
  ```

  Note: `ModelNotLoaded` is imported but unused — drop it if your linter complains.

- [ ] **Step 3: Verify registry now has one entry**

  Run: `uv run python -c "from vector_service.rerankers.registry import RERANKER_REGISTRY, list_reranker_names; print(sorted(RERANKER_REGISTRY)); print(list_reranker_names())"`
  Expected: prints `['bge-reranker-v2-m3']` then `['bge-reranker-v2-m3']`.

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml uv.lock src/vector_service/rerankers/cross_encoder.py
  git commit -m "feat(reranker): CrossEncoderReranker with bge-reranker-v2-m3

  sentence-transformers CrossEncoder wrapper; auto-download via
  modelscope by default; warmup at load; rerank() returns hits sorted
  by score desc with optional top_n truncation."
  ```

---

## Task 6: Pydantic schemas

**Files:**
- Create: `src/vector_service/schemas/rerank.py`

**Interfaces:**
- Produces:
  - `RerankRequest(query: str, documents: list[str], top_n: int | None, model: str | None)` with `extra="forbid"`, `query min_length=1`, `documents min_length=1`, `top_n ge=1` (Optional).
  - `RerankResultItem(index: int, score: float)` with `index ge=0`.
  - `RerankResponse(model: str, results: list[RerankResultItem], request_id: str)`.
  - `RerankerInfo(name: str)`.
  - `RerankModelsResponse(data: list[RerankerInfo])`.

- [ ] **Step 1: Create `schemas/rerank.py`**

  Create `src/vector_service/schemas/rerank.py`:

  ```python
  """Pydantic schemas for the /v1/rerank endpoints."""
  from __future__ import annotations

  from pydantic import BaseModel, ConfigDict, Field


  class RerankRequest(BaseModel):
      """Request body for ``POST /v1/rerank``.

      Schema-level validation is intentionally minimal — length limits
      and ``top_n`` bounds are checked in the route against settings,
      so we can return the canonical error envelope with explicit
      ``code`` strings.
      """

      model_config = ConfigDict(extra="forbid")

      query: str = Field(..., min_length=1)
      documents: list[str] = Field(..., min_length=1)
      top_n: int | None = Field(None, ge=1)
      model: str | None = Field(
          None,
          description="Backend name; defaults to settings.reranker.backend.",
      )


  class RerankResultItem(BaseModel):
      """One ranked document, identified by its index in the request."""

      index: int = Field(..., ge=0)
      score: float


  class RerankResponse(BaseModel):
      """Response body for ``POST /v1/rerank``.

      ``results`` is sorted by ``score`` descending; length is
      ``min(top_n, len(documents))``.
      """

      model: str
      results: list[RerankResultItem]
      request_id: str


  class RerankerInfo(BaseModel):
      """One registered reranker backend."""

      name: str


  class RerankModelsResponse(BaseModel):
      """Response body for ``GET /v1/rerank/models``."""

      data: list[RerankerInfo]
  ```

- [ ] **Step 2: Verify schemas parse**

  Run:
  ```bash
  uv run python - <<'PY'
  from vector_service.schemas.rerank import (
      RerankRequest, RerankResponse, RerankResultItem,
      RerankModelsResponse, RerankerInfo,
  )
  r = RerankRequest(query="q", documents=["a", "b"], top_n=1)
  print(r.model_dump())
  # Should also reject extras:
  try:
      RerankRequest(query="q", documents=["a"], evil=True)
  except Exception as e:
      print("rejected extras:", type(e).__name__)
  PY
  ```
  Expected: prints dict, then `rejected extras: ValidationError`.

- [ ] **Step 3: Commit**

  ```bash
  git add src/vector_service/schemas/rerank.py
  git commit -m "feat(rerank): add Pydantic request/response schemas

  extra='forbid' so typos are caught at the edge; top_n ge=1 only;
  documents min_length=1. Length/character bounds live in the route
  layer for explicit error codes."
  ```

---

## Task 7: Lifespan + health integration

**Files:**
- Modify: `src/vector_service/core/lifespan.py`
- Modify: `src/vector_service/api/health.py`

**Interfaces:**
- Produces: `def build_reranker(settings: Settings) -> Reranker`
- Produces: `app.state.reranker` populated in `lifespan()`; `MODEL_LOADED.labels(kind="reranker").set(1|0)` updated accordingly.
- Produces: `/readyz` response gains `reranker: "ready" | "not_loaded"` field. `status` still computed from embedder + store only.

- [ ] **Step 1: Read `core/lifespan.py` to understand `build_embedder` / `build_store` and the lifespan body**

- [ ] **Step 2: Add `build_reranker`**

  In `src/vector_service/core/lifespan.py`, add the import:

  ```python
  from vector_service.core.errors import RerankerError, RerankerNotLoaded
  from vector_service.rerankers.registry import get_reranker_class
  ```

  Add the function (mirror `build_embedder` shape):

  ```python
  def build_reranker(settings: Settings) -> Reranker:
      """Construct a reranker instance for ``settings.reranker.backend``.

      Raises ``KeyError`` if the backend name is not registered.
      """
      cls = get_reranker_class(settings.reranker.backend)
      return cls(settings=settings)
  ```

- [ ] **Step 3: Wire reranker into `lifespan()`**

  Locate the existing `build_embedder` + `build_store` calls in `lifespan()`. After the store-open block, add:

  ```python
      # ---- reranker (loaded last; not on the /search hot path) ----
      try:
          reranker = build_reranker(settings)
          reranker.load()
          app.state.reranker = reranker
          MODEL_LOADED.labels(kind="reranker").set(1)
          log.info(
              "reranker_loaded",
              model=reranker.model_name,
              backend=settings.reranker.backend,
          )
      except (RerankerNotLoaded, RerankerError, KeyError) as exc:
          MODEL_LOADED.labels(kind="reranker").set(0)
          log.error(
              "reranker_load_failed",
              backend=settings.reranker.backend,
              error=str(exc),
          )
          raise
  ```

  The `raise` makes lifespan exit non-zero — this is the documented behaviour: missing/invalid `VS_RERANKER__BACKEND` is a startup error.

- [ ] **Step 4: Update `/readyz` to include `reranker` field**

  In `src/vector_service/api/health.py`, find the function that builds the `/readyz` response. There should be a dict like `{"status": ..., "store": ..., "embedder": ...}`. Add the reranker status:

  ```python
  reranker_state = "ready" if _reranker_loaded(request) else "not_loaded"

  body = {
      "status": ...,        # unchanged — embedder+store only
      "store": store_state,
      "embedder": embedder_state,
      "reranker": reranker_state,
  }
  ```

  Add a helper alongside `_embedder_loaded()`:

  ```python
  def _reranker_loaded(request: Request) -> bool:
      reranker = getattr(request.app.state, "reranker", None)
      return reranker is not None and getattr(reranker, "_impl", None) is not None
  ```

  Do NOT change how `status` is computed — embedder + store are still the gating pair.

- [ ] **Step 5: Run existing tests**

  Run: `pytest tests/unit -q`
  Expected: PASS. The lifespan tests should still pass because they monkeypatch `build_embedder` and `build_store` but don't touch the new `build_reranker`. If any test does an end-to-end lifespan with a real settings, it will fail because reranker tries to download weights — investigate and monkeypatch `build_reranker` in that test the same way the existing tests monkeypatch `build_embedder`.

- [ ] **Step 6: Commit**

  ```bash
  git add src/vector_service/core/lifespan.py src/vector_service/api/health.py
  # also any test file modified in Step 5
  git commit -m "feat(reranker): eager load in lifespan, expose reranker field on /readyz

  Lifespan builds and loads reranker after embedder+store; failure
  raises and aborts startup. /readyz reports reranker status for
  diagnostics but does not gate the overall ready signal."
  ```

---

## Task 8: Route + main.py wiring

**Files:**
- Create: `src/vector_service/api/rerank.py`
- Modify: `src/vector_service/main.py`

**Interfaces:**
- Produces:
  - `router = APIRouter(prefix="/v1", tags=["rerank"])`
  - `POST /v1/rerank` → `RerankResponse`
  - `GET /v1/rerank/models` → `RerankModelsResponse`

- [ ] **Step 1: Create `api/rerank.py`**

  Create `src/vector_service/api/rerank.py`:

  ```python
  """Rerank API: ``POST /v1/rerank`` and ``GET /v1/rerank/models``."""
  from __future__ import annotations

  import asyncio
  import time
  from typing import TYPE_CHECKING

  from fastapi import APIRouter, HTTPException, Request

  from vector_service.core.config import Settings
  from vector_service.core.errors import RerankerError, RerankerNotLoaded
  from vector_service.core.logging import get_logger, request_id_var
  from vector_service.core.metrics import RERANK_DURATION_SECONDS, RERANK_REQUESTS_TOTAL
  from vector_service.rerankers.base import Reranker
  from vector_service.rerankers.registry import list_reranker_names
  from vector_service.schemas.rerank import (
      RerankModelsResponse,
      RerankRequest,
      RerankResponse,
      RerankResultItem,
      RerankerInfo,
  )

  if TYPE_CHECKING:  # pragma: no cover
      pass

  log = get_logger(__name__)

  router = APIRouter(prefix="/v1", tags=["rerank"])


  def _bad_request(code: str, message: str, extra: dict | None = None) -> HTTPException:
      """Build a 422 HTTPException with the canonical error envelope."""
      detail: dict = {"error": {"code": code, "message": message}}
      if extra:
          detail["error"].update(extra)
      return HTTPException(status_code=422, detail=detail)


  @router.post("/rerank", response_model=RerankResponse)
  async def rerank(req: RerankRequest, request: Request) -> RerankResponse:
      settings: Settings = request.app.state.settings
      reranker: Reranker | None = getattr(request.app.state, "reranker", None)
      model = req.model or settings.reranker.backend

      # ---- registry check (404) --------------------------------------
      if model not in list_reranker_names():
          raise HTTPException(
              status_code=404,
              detail={"error": {"code": "model_not_found",
                                 "message": f"reranker {model!r} is not registered",
                                 "registered": list_reranker_names()}},
          )

      # ---- instance check (503) --------------------------------------
      if reranker is None or getattr(reranker, "_impl", None) is None:
          raise RerankerNotLoaded(f"reranker {model!r} is not loaded")

      # ---- input limits (422) ----------------------------------------
      r_settings = settings.reranker
      n_docs = len(req.documents)
      if n_docs > r_settings.max_documents_per_request:
          raise _bad_request(
              "too_many_documents",
              f"{n_docs} documents > max {r_settings.max_documents_per_request}",
              {"got": n_docs, "max": r_settings.max_documents_per_request},
          )
      for i, d in enumerate(req.documents):
          if len(d) > r_settings.max_chars_per_doc:
              raise _bad_request(
                  "document_too_long",
                  f"documents[{i}] length {len(d)} > max {r_settings.max_chars_per_doc}",
                  {"index": i, "got": len(d), "max": r_settings.max_chars_per_doc},
              )
      if len(req.query) > r_settings.max_query_chars:
          raise _bad_request(
              "query_too_long",
              f"query length {len(req.query)} > max {r_settings.max_query_chars}",
              {"got": len(req.query), "max": r_settings.max_query_chars},
          )
      top_n = req.top_n if req.top_n is not None else r_settings.top_n_default
      if top_n > r_settings.max_top_n:
          raise _bad_request(
              "invalid_top_n",
              f"top_n {top_n} > max {r_settings.max_top_n}",
              {"got": top_n, "max": r_settings.max_top_n},
          )

      # ---- inference -------------------------------------------------
      RERANK_REQUESTS_TOTAL.labels(model, "received").inc()
      loop = asyncio.get_event_loop()
      t0 = time.perf_counter()
      try:
          hits = await loop.run_in_executor(
              None, reranker.rerank, req.query, req.documents, top_n
          )
      except RerankerError:
          RERANK_REQUESTS_TOTAL.labels(model, "error").inc()
          raise
      except Exception as exc:  # defensive wrap
          RERANK_REQUESTS_TOTAL.labels(model, "error").inc()
          log.warning(
              "rerank_failed",
              model=model,
              n_docs=n_docs,
              top_n=top_n,
              error_type=type(exc).__name__,
              error=str(exc),
              request_id=request_id_var.get(),
          )
          raise RerankerError(str(exc) or "rerank failed") from exc

      dt = time.perf_counter() - t0
      RERANK_DURATION_SECONDS.labels(model).observe(dt)
      RERANK_REQUESTS_TOTAL.labels(model, "ok").inc()
      log.info(
          "rerank_completed",
          model=model,
          n_docs=n_docs,
          top_n=top_n,
          latency_ms=int(dt * 1000),
          request_id=request_id_var.get(),
      )

      return RerankResponse(
          model=model,
          results=[RerankResultItem(index=h.index, score=h.score) for h in hits],
          request_id=request_id_var.get(),
      )


  @router.get("/rerank/models", response_model=RerankModelsResponse)
  async def list_rerank_models() -> RerankModelsResponse:
      """List registered reranker backends (does not require a loaded instance)."""
      return RerankModelsResponse(
          data=[RerankerInfo(name=n) for n in list_reranker_names()]
      )
  ```

- [ ] **Step 2: Wire router + exception handlers + OpenAPI tag in `main.py`**

  In `src/vector_service/main.py`:

  a. Add the import (next to other router imports):

  ```python
  from vector_service.api.rerank import router as rerank_router
  ```

  b. Add the import for the new errors (extend the existing `core.errors` import):

  ```python
  from vector_service.core.errors import (
      BackendError,
      CollectionAlreadyExists,
      CollectionNotFound,
      DatabaseAlreadyExists,
      DatabaseNotFound,
      DimensionMismatch,
      RerankerError,
      RerankerNotLoaded,
      VectorServiceError,
  )
  ```

  c. Add a tag to `OPENAPI_TAGS`:

  ```python
  OPENAPI_TAGS += [
      {
          "name": "rerank",
          "description": (
              "Cross-encoder reranking. `POST /v1/rerank` takes a query "
              "and a list of documents and returns documents reordered "
              "by relevance. Use `GET /v1/rerank/models` to list available "
              "reranker backends."
          ),
      },
  ]
  ```

  d. Register exception handlers — put them next to `_backend_err`:

  ```python
  @app.exception_handler(RerankerNotLoaded)
  async def _rerank_not_loaded(request: Request, exc: RerankerNotLoaded):
      return _err(
          "reranker_not_loaded",
          str(exc) or "reranker not loaded",
          503,
          exc=exc,
      )


  @app.exception_handler(RerankerError)
  async def _rerank_error(request: Request, exc: RerankerError):
      log.warning("reranker_error", error=str(exc))
      return _err(
          "reranker_error",
          str(exc) or "reranker failed",
          503,
          exc=exc,
      )
  ```

  e. Register the router (next to other `include_router` calls):

  ```python
  app.include_router(rerank_router)
  ```

- [ ] **Step 3: Verify the routes appear in OpenAPI**

  Run: `uv run python -c "from vector_service.main import app; print([r.path for r in app.routes if 'rerank' in r.path])"`
  Expected: `['/v1/rerank', '/v1/rerank/models']`.

- [ ] **Step 4: Commit**

  ```bash
  git add src/vector_service/api/rerank.py src/vector_service/main.py
  git commit -m "feat(rerank): POST /v1/rerank and GET /v1/rerank/models routes

  Pure /v1/rerank handler dispatches to run_in_executor; validates
  inputs against RerankerSettings; records latency + received/ok/error
  counts; main.py registers router, error envelope handlers, and
  OpenAPI tag."
  ```

---

## Task 9: Tests (unit + contract)

**Files:**
- Create: `tests/unit/test_reranker_settings.py`
- Create: `tests/unit/test_rerank_schemas.py`
- Create: `tests/unit/test_reranker_load.py`
- Create: `tests/unit/test_rerank_route.py`
- Create: `tests/contract/test_cross_encoder_reranker.py`

**Interfaces:** None new (consumes everything built in Tasks 1–8).

- [ ] **Step 1: Write `test_reranker_settings.py`**

  Create `tests/unit/test_reranker_settings.py`:

  ```python
  """Unit tests for ``RerankerSettings`` validation."""
  from __future__ import annotations

  import pytest
  from pydantic import ValidationError

  from vector_service.core.config import RerankerSettings


  def test_defaults_when_only_backend_given():
      s = RerankerSettings(backend="bge-reranker-v2-m3")
      assert s.device == "auto"
      assert s.batch_size == 32
      assert s.max_documents_per_request == 256
      assert s.top_n_default == 10
      assert s.max_top_n == 64
      assert s.download_source == "modelscope"


  def test_backend_required():
      with pytest.raises(ValidationError):
          RerankerSettings()  # type: ignore[call-arg]


  def test_top_n_default_cannot_exceed_max():
      with pytest.raises(ValidationError):
          RerankerSettings(
              backend="bge-reranker-v2-m3", top_n_default=100, max_top_n=50
          )


  def test_batch_size_bounds():
      with pytest.raises(ValidationError):
          RerankerSettings(backend="bge-reranker-v2-m3", batch_size=0)
      with pytest.raises(ValidationError):
          RerankerSettings(backend="bge-reranker-v2-m3", batch_size=10_000)


  def test_device_literal():
      with pytest.raises(ValidationError):
          RerankerSettings(backend="bge-reranker-v2-m3", device="tpu")


  def test_download_source_literal():
      with pytest.raises(ValidationError):
          RerankerSettings(backend="bge-reranker-v2-m3", download_source="civitai")
  ```

- [ ] **Step 2: Write `test_rerank_schemas.py`**

  Create `tests/unit/test_rerank_schemas.py`:

  ```python
  """Unit tests for ``schemas.rerank``."""
  from __future__ import annotations

  import pytest
  from pydantic import ValidationError

  from vector_service.schemas.rerank import (
      RerankModelsResponse,
      RerankRequest,
      RerankResponse,
      RerankResultItem,
      RerankerInfo,
  )


  def test_rerank_request_minimal():
      r = RerankRequest(query="q", documents=["a", "b"])
      assert r.query == "q"
      assert r.documents == ["a", "b"]
      assert r.top_n is None
      assert r.model is None


  def test_rerank_request_rejects_empty_query():
      with pytest.raises(ValidationError):
          RerankRequest(query="", documents=["a"])


  def test_rerank_request_rejects_empty_documents():
      with pytest.raises(ValidationError):
          RerankRequest(query="q", documents=[])


  def test_rerank_request_top_n_must_be_positive():
      with pytest.raises(ValidationError):
          RerankRequest(query="q", documents=["a"], top_n=0)


  def test_rerank_request_rejects_extras():
      with pytest.raises(ValidationError):
          RerankRequest(query="q", documents=["a"], evil=True)


  def test_rerank_response_roundtrip():
      r = RerankResponse(
          model="bge-reranker-v2-m3",
          results=[RerankResultItem(index=0, score=0.5)],
          request_id="req_abc",
      )
      assert r.model_dump() == {
          "model": "bge-reranker-v2-m3",
          "results": [{"index": 0, "score": 0.5}],
          "request_id": "req_abc",
      }


  def test_reranker_info_and_models_response():
      m = RerankModelsResponse(data=[RerankerInfo(name="bge-reranker-v2-m3")])
      assert m.data[0].name == "bge-reranker-v2-m3"
  ```

- [ ] **Step 3: Write `test_reranker_load.py`**

  Create `tests/unit/test_reranker_load.py`:

  ```python
  """Unit tests for ``Reranker`` ABC and lifespan integration."""
  from __future__ import annotations

  import pytest
  from fastapi import FastAPI
  from fastapi.testclient import TestClient

  from vector_service.core.errors import RerankerNotLoaded
  from vector_service.core.metrics import MODEL_LOADED
  from vector_service.core.middleware import RequestIDMiddleware
  from vector_service.rerankers.base import Reranker, ScoredHit


  class _FakeReranker(Reranker):
      """Reranker double with predictable, monotonic scores."""

      model_name = "fake-reranker"

      def __init__(self, settings=None) -> None:
          self._impl: object | None = None
          self.load_calls = 0

      def load(self) -> None:
          self.load_calls += 1
          self._impl = "ready"

      def rerank(self, query, documents, top_n=None):
          n = len(documents) if top_n is None else min(top_n, len(documents))
          scored = sorted(
              [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))],
              key=lambda h: h.score,
              reverse=True,
          )
          return scored[:n]


  def test_reranker_is_abstract():
      with pytest.raises(TypeError):
          Reranker()  # type: ignore[abstract]


  def test_fake_reranker_load_is_idempotent():
      r = _FakeReranker()
      r.load()
      r.load()
      assert r.load_calls == 2
      assert r._impl == "ready"


  def test_scored_hit_is_frozen():
      h = ScoredHit(index=0, score=0.5)
      with pytest.raises(Exception):  # FrozenInstanceError
          h.index = 1  # type: ignore[misc]


  # ----- lifespan / /readyz integration ----------------------------------


  def _make_app(reranker):
      from vector_service.api import health
      from vector_service.main import create_app

      app = create_app()
      # Strip everything except the health router and force a custom reranker.
      app.state.reranker = reranker
      return app


  def test_readyz_reports_reranker_ready():
      r = _FakeReranker()
      r.load()
      app = _make_app(r)
      MODEL_LOADED.labels(kind="reranker").set(1)
      with TestClient(app) as client:
          resp = client.get("/readyz")
      assert resp.status_code == 200
      assert resp.json()["reranker"] == "ready"


  def test_readyz_reports_reranker_not_loaded():
      r = _FakeReranker()  # never .load()'d
      app = _make_app(r)
      MODEL_LOADED.labels(kind="reranker").set(0)
      with TestClient(app) as client:
          resp = client.get("/readyz")
      # status still gated on embedder+store — they may be missing in this
      # stripped app; assert at minimum the reranker field is "not_loaded"
      body = resp.json()
      assert body["reranker"] == "not_loaded"


  def test_reranker_not_loaded_exc_subclass():
      assert issubclass(RerankerNotLoaded, Exception)
      msg = RerankerNotLoaded("boom").args[0]
      assert msg == "boom"
  ```

  Note: `_make_app` is intentionally minimal — the full `create_app()` registers all routers and state via `lifespan`, which we don't want to exercise here. Adjust as needed to match how the existing `test_embedder_load.py` constructs its app (look at its `_make_app` helper for the canonical pattern).

- [ ] **Step 4: Write `test_rerank_route.py`**

  Create `tests/unit/test_rerank_route.py`:

  ```python
  """Unit tests for ``/v1/rerank`` and ``/v1/rerank/models``."""
  from __future__ import annotations

  import pytest
  from fastapi import FastAPI
  from fastapi.testclient import TestClient

  from vector_service.api.rerank import router as rerank_router
  from vector_service.core.config import RerankerSettings, Settings
  from vector_service.core.errors import RerankerNotLoaded
  from vector_service.core.middleware import RequestIDMiddleware
  from vector_service.main import _err
  from vector_service.rerankers.base import Reranker, ScoredHit


  class _FakeReranker(Reranker):
      """Stable scores: index 0 → 1.0, index 1 → 0.5, index 2 → 0.25, …"""

      model_name = "fake-reranker"

      def __init__(self, settings=None) -> None:
          self._impl: object | None = "ready"

      def load(self) -> None:
          self._impl = "ready"

      def rerank(self, query, documents, top_n=None):
          n = len(documents) if top_n is None else min(top_n, len(documents))
          scored = sorted(
              [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))],
              key=lambda h: h.score,
              reverse=True,
          )
          return scored[:n]


  def _make_app(reranker: Reranker | None, *, settings: Settings | None = None) -> FastAPI:
      app = FastAPI(title="rerank-test")
      app.add_middleware(RequestIDMiddleware)
      app.include_router(rerank_router)
      app.state.settings = settings or Settings(
          reranker=RerankerSettings(backend="bge-reranker-v2-m3")
      )
      app.state.reranker = reranker

      from fastapi.exceptions import RequestValidationError

      @app.exception_handler(RerankerNotLoaded)
      async def _h(req, exc):
          return _err("reranker_not_loaded", str(exc), 503, exc=exc)

      @app.exception_handler(RequestValidationError)
      async def _v(req, exc):
          return _err("invalid_request", str(exc), 422, exc=exc)

      from starlette.exceptions import HTTPException as StarletteHTTPException

      @app.exception_handler(StarletteHTTPException)
      async def _http(req, exc):
          d = exc.detail
          if isinstance(d, dict) and "error" in d:
              inner = d["error"]
              extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
              return _err(
                  inner.get("code", "error"),
                  inner.get("message", str(d)),
                  exc.status_code,
                  extras,
              )
          return _err("error", str(d), exc.status_code)

      return app


  # ---- happy path --------------------------------------------------------


  def test_rerank_returns_results_sorted_by_score_desc():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a", "b", "c"]},
          )
      assert resp.status_code == 200
      body = resp.json()
      assert body["model"] == "bge-reranker-v2-m3"
      scores = [r["score"] for r in body["results"]]
      assert scores == sorted(scores, reverse=True)
      assert [r["index"] for r in body["results"]] == [0, 1, 2]
      assert "request_id" in body and body["request_id"]


  def test_rerank_top_n_truncates():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a", "b", "c", "d"], "top_n": 2},
          )
      assert resp.status_code == 200
      assert len(resp.json()["results"]) == 2


  def test_rerank_uses_default_top_n():
      app = _make_app(_FakeReranker(),
                      settings=Settings(reranker=RerankerSettings(
                          backend="bge-reranker-v2-m3", top_n_default=2)))
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a", "b", "c"]},
          )
      assert resp.status_code == 200
      assert len(resp.json()["results"]) == 2


  def test_rerank_request_id_is_propagated():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a"]},
              headers={"X-Request-ID": "req_test_xyz"},
          )
      assert resp.json()["request_id"] == "req_test_xyz"


  # ---- error paths -------------------------------------------------------


  def test_rerank_404_when_model_unknown():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a"], "model": "nope"},
          )
      assert resp.status_code == 404
      assert resp.json()["error"]["code"] == "model_not_found"


  def test_rerank_422_too_many_documents():
      settings = Settings(reranker=RerankerSettings(
          backend="bge-reranker-v2-m3", max_documents_per_request=2))
      app = _make_app(_FakeReranker(), settings=settings)
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a", "b", "c"]},
          )
      assert resp.status_code == 422
      assert resp.json()["error"]["code"] == "too_many_documents"


  def test_rerank_422_document_too_long():
      settings = Settings(reranker=RerankerSettings(
          backend="bge-reranker-v2-m3", max_chars_per_doc=4))
      app = _make_app(_FakeReranker(), settings=settings)
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["ok", "thisistoolong"]},
          )
      assert resp.status_code == 422
      assert resp.json()["error"]["code"] == "document_too_long"


  def test_rerank_422_query_too_long():
      settings = Settings(reranker=RerankerSettings(
          backend="bge-reranker-v2-m3", max_query_chars=2))
      app = _make_app(_FakeReranker(), settings=settings)
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "toolong", "documents": ["a"]},
          )
      assert resp.status_code == 422
      assert resp.json()["error"]["code"] == "query_too_long"


  def test_rerank_422_invalid_top_n():
      settings = Settings(reranker=RerankerSettings(
          backend="bge-reranker-v2-m3", max_top_n=2))
      app = _make_app(_FakeReranker(), settings=settings)
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a"], "top_n": 5},
          )
      assert resp.status_code == 422
      assert resp.json()["error"]["code"] == "invalid_top_n"


  def test_rerank_503_when_reranker_not_loaded():
      class _Unloaded(Reranker):
          model_name = "unloaded"

          def __init__(self):
              self._impl = None

          def load(self):
              self._impl = "ready"

          def rerank(self, q, docs, top_n=None):
              return [ScoredHit(index=0, score=1.0)]

      app = _make_app(_Unloaded())  # never .load()'d
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": ["a"]},
          )
      assert resp.status_code == 503
      assert resp.json()["error"]["code"] == "reranker_not_loaded"


  def test_rerank_422_pydantic_rejects_empty_documents():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.post(
              "/v1/rerank",
              json={"query": "q", "documents": []},
          )
      assert resp.status_code == 422
      assert resp.json()["error"]["code"] == "invalid_request"


  # ---- list models -------------------------------------------------------


  def test_list_rerank_models_returns_registered_names():
      app = _make_app(_FakeReranker())
      with TestClient(app) as client:
          resp = client.get("/v1/rerank/models")
      assert resp.status_code == 200
      names = [m["name"] for m in resp.json()["data"]]
      assert "bge-reranker-v2-m3" in names
  ```

- [ ] **Step 5: Write `tests/contract/test_cross_encoder_reranker.py`**

  Create `tests/contract/test_cross_encoder_reranker.py`:

  ```python
  """Contract test for the real CrossEncoderReranker.

  Skipped by default on Windows (matches project convention from b86e077).
  Run explicitly with::

      pytest -m contract tests/contract/test_cross_encoder_reranker.py
  """
  from __future__ import annotations

  import sys

  import pytest

  from vector_service.core.config import RerankerSettings, Settings
  from vector_service.rerankers.cross_encoder import CrossEncoderReranker

  pytestmark = [pytest.mark.contract, pytest.mark.slow]

  # Match project convention: skip contract tests on Windows by default.
  _SKIP_ON_WINDOWS = pytest.mark.skipif(
      sys.platform.startswith("win"),
      reason="contract tests require model weights; skip on Windows by default",
  )


  @_SKIP_ON_WINDOWS
  def test_bge_reranker_v2_m3_orders_relevant_first():
      settings = Settings(
          reranker=RerankerSettings(
              backend="bge-reranker-v2-m3",
              model_dir="./models/bge-reranker-v2-m3",
          )
      )
      r = CrossEncoderReranker(settings=settings)
      r.load()
      docs = [
          "巴黎是法国首都。",
          "苹果是一种水果。",
          "北京是中华人民共和国的首都。",
      ]
      hits = r.rerank("中国首都", docs, top_n=3)
      assert hits[0].index == 2
      scores = [h.score for h in hits]
      assert scores == sorted(scores, reverse=True)
  ```

- [ ] **Step 6: Run the unit-test suite**

  Run: `pytest tests/unit -q`
  Expected: all unit tests pass; contract test is collected but skipped (it has `pytest.mark.contract`).

- [ ] **Step 7: Commit**

  ```bash
  git add tests/unit/test_reranker_settings.py \
          tests/unit/test_rerank_schemas.py \
          tests/unit/test_reranker_load.py \
          tests/unit/test_rerank_route.py \
          tests/contract/test_cross_encoder_reranker.py
  git commit -m "test(reranker): unit + contract tests for the reranker subsystem

  Covers: settings validation, schema constraints, ABC contract,
  lifespan /readyz reranker field, all 9 error code paths, happy-path
  routing + top_n truncation + request_id propagation, list models.
  Contract test gated by @pytest.mark.contract + Windows skip."
  ```

---

## Task 10: README + final verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example` if anything missing

**Interfaces:** Documentation only.

- [ ] **Step 1: Read the existing README to find where the embeddings section is**

- [ ] **Step 2: Add a Reranker section**

  In `README.md`, add a new top-level section after the embeddings docs (find the closest heading like `## Embeddings` or similar and insert after). Use this template (translate to match the existing README voice and language; the existing README is Chinese — adapt accordingly):

  ```markdown
  ## Rerank

  ### 端点

  | 方法 | 路径 | 说明 |
  |---|---|---|
  | `POST` | `/v1/rerank` | 重排序：接收 query + documents，返回按相关性降序的 `{index, score}` 列表 |
  | `GET`  | `/v1/rerank/models` | 列出已注册的 reranker 后端 |

  ### 启动

  ```bash
  VS_RERANKER__BACKEND=bge-reranker-v2-m3 uv run vector-service
  ```

  首次启动会自动从 ModelScope 下载 `BAAI/bge-reranker-v2-m3` 权重到 `./models/bge-reranker-v2-m3/`。

  ### 示例

  ```bash
  curl -X POST localhost:8080/v1/rerank \
    -H 'Content-Type: application/json' \
    -d '{
      "query": "中国首都",
      "documents": ["巴黎是法国首都", "苹果是一种水果", "北京是中华人民共和国的首都"],
      "top_n": 3
    }'
  ```

  ### 错误码

  | HTTP | code | 触发 |
  |---|---|---|
  | 404 | `model_not_found` | 后端名未注册 |
  | 422 | `too_many_documents` | documents 超过 `VS_RERANKER__MAX_DOCUMENTS_PER_REQUEST` |
  | 422 | `document_too_long` | 任一 document 超过 `VS_RERANKER__MAX_CHARS_PER_DOC` |
  | 422 | `query_too_long` | query 超过 `VS_RERANKER__MAX_QUERY_CHARS` |
  | 422 | `invalid_top_n` | top_n 超过 `VS_RERANKER__MAX_TOP_N` |
  | 503 | `reranker_not_loaded` / `reranker_error` | reranker 未加载或推理失败 |
  ```

  Adapt to match the existing README's exact style — Chinese headers, table format, and tone.

- [ ] **Step 3: Final end-to-end verification**

  Run all unit tests one more time:

  ```bash
  pytest tests/unit -q
  ```

  Expected: all green.

  Then verify the app boots in dev mode (without actually downloading model weights) by short-circuiting reranker load. The simplest way:

  ```bash
  VS_RERANKER__BACKEND=bge-reranker-v2-m3 \
  VS_RERANKER__AUTO_DOWNLOAD=false \
  VS_RERANKER__MODEL_DIR=/tmp/nonexistent \
  uv run python -c "
  import asyncio
  from fastapi.testclient import TestClient
  from vector_service.main import create_app
  app = create_app()
  with TestClient(app) as c:
      print('healthz:', c.get('/healthz').json())
      print('models:', c.get('/v1/rerank/models').json())
  "
  ```

  Expected:
  - First line: app startup raises (because `RerankerNotLoaded` — model dir missing + auto_download disabled). That's the **intended** behaviour per spec §1.1.
  - To see a successful boot path without weights, monkeypatch `build_reranker` in a script:

  ```bash
  uv run python - <<'PY'
  from vector_service.rerankers.base import Reranker, ScoredHit

  class _Stub(Reranker):
      model_name = "stub"
      def __init__(self): self._impl = None
      def load(self): self._impl = "ready"
      def rerank(self, q, docs, top_n=None):
          return [ScoredHit(index=i, score=1.0/(i+1)) for i in range(min(top_n or len(docs), len(docs)))]

  from vector_service.core import lifespan
  lifespan.build_reranker = lambda s: _Stub()
  from fastapi.testclient import TestClient
  from vector_service.main import create_app
  app = create_app()
  with TestClient(app) as c:
      print("healthz:", c.get("/healthz").json())
      print("readyz:", c.get("/readyz").json())
      print("models:", c.get("/v1/rerank/models").json())
      r = c.post("/v1/rerank", json={"query": "q", "documents": ["a","b","c"], "top_n": 2})
      print("rerank:", r.status_code, r.json())
  PY
  ```

  Expected: all calls return 200 (or for `/readyz`, may be 200 or 503 depending on embedder/store state — focus on `reranker: "ready"`).

- [ ] **Step 4: Commit**

  ```bash
  git add README.md .env.example
  git commit -m "docs(reranker): README section with endpoints, config, examples

  Covers POST /v1/rerank and GET /v1/rerank/models, env-var setup,
  curl smoke, and the canonical error code table."
  ```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| §1 概述 (范围) | Task 1, 2, 3, 4, 5 |
| §3.2 模块布局 | Task 3, 4, 5, 6, 8 |
| §3.3 `Reranker` ABC | Task 3 |
| §3.4 `CrossEncoderReranker` | Task 5 |
| §3.5 `RERANKER_REGISTRY` | Task 4, 5 |
| §4.0 GET /v1/rerank/models | Task 8 |
| §4.1 POST /v1/rerank | Task 8 |
| §4.2 响应形状 | Task 6, 8 |
| §4.3 错误码矩阵 (9 个) | Task 1 (errors), 8 (handler), 9 (tests) |
| §5 配置 (`RerankerSettings`) | Task 2, 9 (settings tests) |
| §6 Schemas | Task 6, 9 (schema tests) |
| §7 生命周期 | Task 7 |
| §8.1 指标 | Task 1 |
| §8.2 `MODEL_LOADED` 标签变更 | Task 1, 7 |
| §8.3 `/readyz` 增量 | Task 7, 9 (health test) |
| §8.4 日志格式 | Task 7, 8 |
| §9 测试 (4 unit + 1 contract) | Task 9 |
| §10 依赖 (`sentence-transformers>=2.6`) | Task 5 |
| §11.2 curl 验证 | Task 10 |
| §13 验收标准 1-8 | Tasks 1-10 collectively |

No spec gap.

**2. Placeholder scan:** No `TBD` / `TODO` / "implement later" / "similar to" / unspecified references.

**3. Type consistency:**
- `Reranker.model_name` (Task 3) ↔ `CrossEncoderReranker.model_name = "bge-reranker-v2-m3"` (Task 5) ↔ `RERANKER_REGISTRY["bge-reranker-v2-m3"]` (Task 5) ↔ `req.model or settings.reranker.backend` default string in route (Task 8) — all consistent.
- `ScoredHit(index: int, score: float)` defined Task 3, consumed Tasks 5, 9 — consistent.
- `RerankRequest.query min_length=1`, `documents min_length=1`, `top_n ge=1`, `extra="forbid"` defined Task 6, exercised Task 9 — consistent.
- `MODEL_LOADED.labels(kind="reranker").set(0|1)` defined Task 1, used Task 7 (lifespan) and Task 9 (test) — consistent.
- `RERANK_DURATION_SECONDS.labels(model).observe(...)` / `RERANK_REQUESTS_TOTAL.labels(model, status).inc()` defined Task 1, used Task 8 — consistent.
- `Settings.reranker: RerankerSettings` defined Task 2, consumed Tasks 5 (init), 7 (build), 8 (route), 9 (test factory), 10 (smoke) — consistent.
- All field names on `RerankerSettings` (Task 2) match the references in Tasks 5, 8, 9 — checked `backend`, `model_dir`, `device`, `batch_size`, `max_length`, `auto_download`, `download_source`, `hf_repo`, `ms_repo`, `max_documents_per_request`, `max_chars_per_doc`, `max_query_chars`, `max_top_n`, `top_n_default` — all consistent.
