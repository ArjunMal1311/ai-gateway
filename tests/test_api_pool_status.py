def test_pool_status_all_available_initially(api_client):
    resp = api_client.get("/v1/pool/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_keys"] == 2
    assert body["available"] == 2
    assert body["permanently_blocked"] == 0


def test_keys_endpoint_lists_each_configured_key(api_client):
    resp = api_client.get("/v1/keys")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    for entry in body:
        assert entry["status"] == "available"


def test_health_and_ready(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp_ready = api_client.get("/health/ready")
    assert resp_ready.status_code == 200
    body = resp_ready.json()
    assert body["redis_ok"] is True
    assert body["keys_configured"] == 2


def test_usage_keys_counts_calls_and_failures_per_key(api_client, monkeypatch):
    from app.providers.base import ProviderResult
    from app.providers.gemini.provider import GeminiProvider

    outcomes = iter([Exception("429 Too Many Requests"), None, None])

    async def fake_generate(self, ctx):
        o = next(outcomes)
        if o:
            raise o
        return ProviderResult(text="ok", input_tokens=1, output_tokens=1, total_tokens=2)

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)
    api_client.post("/v1/generate", json={"prompt": "hi"})
    api_client.post("/v1/generate", json={"prompt": "hi"})

    body = api_client.get("/v1/usage/keys").json()
    assert len(body["keys"]) == 2
    assert sum(k["calls"] for k in body["keys"]) == 3
    assert sum(k["failures"] for k in body["keys"]) == 1
    hit = [k for k in body["keys"] if k["failures"]][0]
    assert hit["status"] == "rate_limited"
    assert all(set(k) >= {"suffix", "status", "calls", "failures", "tokens", "models"} for k in body["keys"])
