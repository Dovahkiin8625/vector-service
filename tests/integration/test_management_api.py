def test_create_and_list_collection(client):
    r = client.post("/collections", json={"name": "c1"})
    assert r.status_code == 201
    r = client.get("/collections")
    assert r.status_code == 200
    assert "c1" in r.json()["collections"]


def test_create_twice_409(client):
    client.post("/collections", json={"name": "c"})
    r = client.post("/collections", json={"name": "c"})
    assert r.status_code == 409


def test_get_collection_info(client):
    client.post("/collections", json={"name": "c"})
    r = client.get("/collections/c")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "c"
    assert body["dim"] == 4  # FakeEmbedder


def test_upsert_with_texts_and_search(client):
    client.post("/collections", json={"name": "docs"})
    r = client.put("/collections/docs/vectors", json={
        "ids": ["a", "b"],
        "texts": ["alpha", "beta"],
        "metadatas": [{"src": "x"}, {"src": "y"}],
    })
    assert r.status_code == 200
    assert r.json()["upserted"] == 2

    r = client.post("/collections/docs/search", json={
        "query_text": "alpha", "top_k": 1,
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["id"] == "a"


def test_upsert_with_embeddings(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"],
        "embeddings": [[1.0, 0.0, 0.0, 0.0]],
    })
    assert r.status_code == 200


def test_upsert_xor_422(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"], "texts": ["t"], "embeddings": [[1, 0, 0, 0]],
    })
    assert r.status_code == 422


def test_dim_mismatch_422(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"],
        "embeddings": [[1.0, 0.0]],  # dim=2 != collection dim=4
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "dimension_mismatch"


def test_search_missing_collection_404(client):
    r = client.post("/collections/nope/search", json={"query_text": "x"})
    assert r.status_code == 404


def test_filter_search(client):
    client.post("/collections", json={"name": "f"})
    client.put("/collections/f/vectors", json={
        "ids": ["a", "b"],
        "texts": ["alpha", "beta"],
        "metadatas": [{"src": "x"}, {"src": "y"}],
    })
    r = client.post("/collections/f/search", json={
        "query_text": "alpha", "top_k": 10, "filter": {"src": "y"},
    })
    assert r.status_code == 200
    assert [h["id"] for h in r.json()["hits"]] == ["b"]


def test_delete_and_drop(client):
    client.post("/collections", json={"name": "d"})
    client.put("/collections/d/vectors", json={"ids": ["a"], "texts": ["x"]})
    r = client.post("/collections/d/vectors/delete", json={"ids": ["a"]})
    assert r.status_code == 200
    r = client.delete("/collections/d")
    assert r.status_code == 200


def test_get_vectors(client):
    client.post("/collections", json={"name": "g"})
    client.put("/collections/g/vectors", json={"ids": ["a"], "texts": ["x"], "metadatas": [{"k": "v"}]})
    r = client.post("/collections/g/vectors/get", json={"ids": ["a"]})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == "a"
    assert items[0]["metadata"] == {"k": "v"}


def test_request_id_in_response_header(client):
    r = client.get("/healthz", headers={"X-Request-ID": "req_test123"})
    assert r.headers.get("X-Request-ID") == "req_test123"