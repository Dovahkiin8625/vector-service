def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "bge-m3" for m in data)


def test_get_model(client):
    r = client.get("/v1/models/bge-m3")
    assert r.status_code == 200
    assert r.json()["id"] == "bge-m3"


def test_get_unknown_model_404(client):
    r = client.get("/v1/models/bogus")
    assert r.status_code == 404


def test_embed_string_input(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "model": "bge-m3"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "bge-m3"
    assert len(body["data"]) == 1
    assert isinstance(body["data"][0]["embedding"], list)
    # FakeEmbedder dim=4（见 conftest）
    assert len(body["data"][0]["embedding"]) == 4
    assert body["usage"]["prompt_tokens"] >= 1


def test_embed_list_input(client):
    r = client.post("/v1/embeddings", json={"input": ["a", "b", "c"], "model": "bge-m3"})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    assert [d["index"] for d in r.json()["data"]] == [0, 1, 2]


def test_embed_unknown_model_404(client):
    r = client.post("/v1/embeddings", json={"input": "x", "model": "bogus"})
    assert r.status_code == 404


def test_embed_empty_input_422(client):
    r = client.post("/v1/embeddings", json={"input": "", "model": "bge-m3"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_embed_invalid_encoding_422(client):
    r = client.post("/v1/embeddings", json={"input": "x", "model": "bge-m3", "encoding_format": "base64"})
    assert r.status_code == 422