"""Manual smoke: bypass BGE-M3, use FakeEmbedder + FakeStore."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vector_service.main import create_app
from vector_service.core import lifespan as lspan
from vector_service.testing.fake_embedder import FakeEmbedder
from vector_service.testing.fake_store import FakeStore

# monkey patch to skip heavy model load + milvus connect
lspan.build_embedder = lambda settings: FakeEmbedder(dim=4)
lspan.build_store = lambda settings: FakeStore()

from fastapi.testclient import TestClient

app = create_app()
with TestClient(app) as c:
    r = c.get("/healthz"); print("healthz:", r.status_code, r.json())
    r = c.get("/readyz"); print("readyz:", r.status_code, r.json())
    r = c.get("/v1/models"); print("models:", r.status_code, r.json())
    r = c.post("/v1/embeddings", json={"input": "hi", "model": "bge-m3"})
    print("embed:", r.status_code, len(r.json()["data"][0]["embedding"]))
    r = c.post("/collections", json={"name": "demo"}); print("create coll:", r.status_code, r.json())
    r = c.put("/collections/demo/vectors", json={"ids": ["a"], "texts": ["hi"]})
    print("upsert:", r.status_code, r.json())
    r = c.post("/collections/demo/search", json={"query_text": "hi", "top_k": 1})
    print("search:", r.status_code, r.json())
    r = c.get("/metrics"); print("metrics lines:", len(r.text.splitlines()))