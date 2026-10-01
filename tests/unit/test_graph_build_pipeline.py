"""End-to-end ``run_graph_build_pipeline`` with a real corpus + fake LLM.

The pipeline's network hop (``get_chat_client``) is replaced with a
deterministic local chat function; the vector store is an in-memory
collection-lifecycle fake. Pins the derived-layer contract:

- leaves → LLM extraction → canonical merge → one community;
- two physical collections created/replaced with the right row counts;
- graph rows land in the corpus only after counts verify;
- claims toggle with ``include_claims``;
- an empty corpus is a 400 ``graph_empty``.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from vector_service.api import graph as graph_api
from vector_service.api.graph import run_graph_build_pipeline
from vector_service.core.errors import CollectionNotFound
from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.graph.names import graph_collection_names

# ---- fakes -------------------------------------------------------------


class FakeChatClient:
    """Extraction JSON per leaf; paragraphs for community summaries."""

    def __init__(self, *, with_claim=False):
        self._with_claim = with_claim
        self.extraction_calls = 0
        self.summary_calls = 0

    def as_chat_fn(self):
        def fn(messages):
            system = messages[0]["content"]
            if system.startswith("You are an expert knowledge-graph"):
                self.extraction_calls += 1
                return self._extraction(messages[1]["content"])
            self.summary_calls += 1
            return "Alice and Bob collaborate closely on the project."

        return fn

    def _extraction(self, user_content):
        if "第一段" in user_content:
            edge_desc = "works with"
            alice_desc = "an analyst"
        else:
            edge_desc = "collaborates"
            alice_desc = "team lead"
        payload = {
            "entities": [
                {"name": "Alice", "type": "person", "description": alice_desc},
                {"name": "Bob", "type": "person", "description": "a colleague"},
            ],
            "relationships": [
                {"source": "Alice", "target": "Bob",
                 "description": edge_desc},
            ],
            "claims": [],
        }
        if self._with_claim:
            payload["claims"] = [{
                "subject": "Alice", "object": "Bob", "type": "collaboration",
                "status": "true",
                "statement": "Alice collaborates with Bob",
            }]
        return json.dumps(payload)


class FakeStore:
    """Minimal collection lifecycle with per-collection upsert rows."""

    def __init__(self):
        self.databases = ["default"]
        self.rows: dict[str, list] = {}
        self.created: list[str] = []
        self.dropped: list[str] = []

    def list_databases(self):
        return list(self.databases)

    def create_database(self, name):
        self.databases.append(name)

    def list_collections(self, database):
        return list(self.rows)

    def create_collection(self, **kwargs):
        name = kwargs["name"]
        self.rows.setdefault(name, [])
        self.created.append(name)

    def drop_collection(self, database, name):
        if name not in self.rows:
            raise CollectionNotFound(name)
        del self.rows[name]
        self.dropped.append(name)

    def upsert(self, database, collection, primary_field, vector_field,
               ids, vectors, scalar_rows):
        bucket = self.rows.setdefault(collection, [])
        assert len(ids) == len(vectors) == len(scalar_rows)
        bucket.extend(ids)

    def count_rows(self, database, collection):
        return len(self.rows.get(collection, []))


class FakeEmbedder:
    model_name = "bge-m3"
    dim = 4

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _Settings:
    inference_timeout_seconds = 30.0

    class llm:
        max_concurrency = 4


# ---- harness -----------------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    store = FakeStore()
    chat_client = FakeChatClient()
    monkeypatch.setattr(
        graph_api, "get_chat_client", lambda settings: chat_client
    )

    def seed(with_claim=False):
        doc = DocumentRecord(
            doc_id="d1", database="default", collection="ingest",
            filename="年报.pdf", mime="application/pdf",
            content_hash="a" * 64,
        )
        chunks = [
            ChunkRecord(chunk_id="d1_0", doc_id="d1", database="default",
                        collection="ingest", chunk_index=0, text="第一段内容",
                        token_count=5),
            ChunkRecord(chunk_id="d1_1", doc_id="d1", database="default",
                        collection="ingest", chunk_index=1, text="第二段内容",
                        token_count=5),
        ]
        repo.store_document(doc, chunks)

    async def _never_cancel():
        return False

    async def _ignore_event(event):
        return None

    async def run(*, include_claims=True, with_claim=False, job_id="j1"):
        repo.create_job(
            job_id, database="default", collection="ingest",
            filename=None, mime=None, job_type="graph_build",
        )
        entity_coll, community_coll = graph_collection_names("ingest")
        await run_graph_build_pipeline(
            settings=_Settings(), store=store, repo=repo,
            embedder=FakeEmbedder(),
            database="default", logical_collection="ingest",
            entity_coll=entity_coll, community_coll=community_coll,
            entity_types=(),
            community_iterations=20, min_community_size=2,
            include_claims=include_claims, batch_size=16,
            inference_timeout_seconds=30.0,
            emit=_ignore_event,
            job_id=job_id,
            should_cancel=_never_cancel,
        )
        return entity_coll, community_coll

    return SimpleNamespace(repo=repo, store=store, chat=chat_client,
                           seed=seed, run=run)


def _run(coro):
    return asyncio.run(coro)


# ---- tests -------------------------------------------------------------


def test_graph_build_full_pipeline(harness):
    harness.seed()
    entity_coll, community_coll = _run(harness.run())

    # Two physical collections replaced with exact row counts.
    assert set(harness.store.created) == {entity_coll, community_coll}
    assert len(harness.store.rows[entity_coll]) == 2
    assert len(harness.store.rows[community_coll]) == 1
    stats = harness.repo.graph_stats("default", "ingest")
    assert stats["entities_count"] == 2
    assert stats["edges_count"] == 1
    assert stats["communities_count"] == 1
    community = stats["communities"][0]
    assert community["size"] == 2
    assert community["summary"] == (
        "Alice and Bob collaborate closely on the project."
    )
    # Edge weight accumulated over both chunks.
    with harness.repo._txn() as conn:
        weight = conn.execute(
            "SELECT weight FROM edges WHERE database = 'default' "
            "AND collection = 'ingest'"
        ).fetchone()["weight"]
    assert weight == 2

    job = harness.repo.get_job("j1")
    assert job["status"] == "done"
    assert job["chunk_count"] == 2
    assert harness.chat.extraction_calls == 2
    assert harness.chat.summary_calls == 1


def test_graph_build_persists_claims_when_enabled(harness):
    # Swap in a claim-emitting chat client.
    claim_chat = FakeChatClient(with_claim=True)
    import vector_service.api.graph as g

    g.get_chat_client = lambda settings: claim_chat  # local patch
    harness.seed()
    try:
        _run(harness.run(include_claims=True, job_id="jc"))
    finally:
        g.get_chat_client = lambda s: harness.chat
    assert harness.repo.graph_stats("default", "ingest")["claims_count"] == 1


def test_graph_build_drops_claims_when_disabled(harness):
    claim_chat = FakeChatClient(with_claim=True)
    import vector_service.api.graph as g

    original = g.get_chat_client
    g.get_chat_client = lambda settings: claim_chat
    harness.seed()
    try:
        _run(harness.run(include_claims=False, job_id="jn"))
    finally:
        g.get_chat_client = original
    assert harness.repo.graph_stats("default", "ingest")["claims_count"] == 0


def test_graph_build_replaces_previous_graph(harness):
    harness.seed()
    _run(harness.run(job_id="j1"))
    # Rebuild on a second job must not duplicate graph rows.
    _run(harness.run(job_id="j2"))
    stats = harness.repo.graph_stats("default", "ingest")
    assert stats["entities_count"] == 2
    assert stats["edges_count"] == 1
    assert stats["communities_count"] == 1


def test_graph_build_empty_corpus_is_400(harness):
    with pytest.raises(Exception) as exc:
        _run(harness.run())
    assert exc.value.status_code == 400
    assert exc.value.detail["error"]["code"] == "graph_empty"
    # No physical collection survived the failure cleanup.
    assert harness.store.rows == {}
