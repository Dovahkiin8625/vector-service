from prometheus_client import parser


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ok_with_components(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_metrics_exposition(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "vs_info" in text
    parsed = list(parser.text_string_to_metric_families(text))
    names = {f.name for f in parsed}
    assert "vs_info" in names