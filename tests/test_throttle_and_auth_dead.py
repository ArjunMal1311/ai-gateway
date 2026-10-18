from app.models.enums import FailureReason
from app.providers.base import ProviderResult
from app.providers.gemini.errors import classify_gemini_error
from app.providers.gemini.provider import GeminiProvider

SERVICE_ACCOUNT_401 = (
    "401 UNAUTHENTICATED. {'error': {'code': 401, 'message': 'The bound service account is "
    "deleted or disabled. The service account bound to the API key must be active.', "
    "'status': 'UNAUTHENTICATED'}}"
)


def test_deleted_service_account_401_is_dead_auth():
    c = classify_gemini_error(SERVICE_ACCOUNT_401)
    assert c.reason == FailureReason.AUTH_DEAD
    assert c.scope == "key"


def test_throttle_stops_after_cap_and_returns_429(api_client, monkeypatch):
    calls = []

    async def fake_generate(self, ctx):
        calls.append(ctx.model)
        raise Exception("429 RESOURCE_EXHAUSTED. Too many requests")

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)

    resp = api_client.post("/v1/generate", json={"prompt": "hi"})
    assert resp.status_code == 429
    assert len(calls) <= 3
    assert "retry_after_seconds" in resp.json() or resp.headers.get("Retry-After")


def test_dead_auth_key_is_not_retried_forever(api_client, monkeypatch):
    calls = []

    async def fake_generate(self, ctx):
        calls.append(ctx.api_key if hasattr(ctx, "api_key") else None)
        if len(calls) == 1:
            raise Exception(SERVICE_ACCOUNT_401)
        return ProviderResult(text="ok", input_tokens=1, output_tokens=1, total_tokens=2)

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)

    resp = api_client.post("/v1/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["attempts"] == 2


def test_unknown_error_gives_up_after_three_keys(api_client, monkeypatch):
    calls = []

    async def fake_generate(self, ctx):
        calls.append(1)
        raise Exception("something nobody has classified yet")

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)

    try:
        resp = api_client.post("/v1/generate", json={"prompt": "hi"})
        assert resp.status_code >= 500
    except Exception as e:
        assert "nobody has classified" in str(e)
    assert len(calls) == 3


def test_ready_reports_degraded_when_keys_are_dead(api_client, monkeypatch):
    async def fake_generate(self, ctx):
        raise Exception(SERVICE_ACCOUNT_401)

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)
    api_client.post("/v1/generate", json={"prompt": "hi"})

    body = api_client.get("/health/ready").json()
    assert body["status"] == "degraded"
    assert body["usable_keys"] == 0
    assert body["detail"]
