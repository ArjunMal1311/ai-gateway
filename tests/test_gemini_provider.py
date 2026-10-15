from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models.enums import FailureReason
from app.providers.base import GenerateContext
from app.providers.gemini.errors import classify_gemini_error
from app.providers.gemini.provider import GeminiProvider


@pytest.mark.parametrize(
    "message,expected_reason",
    [
        ("400 API key not valid. Please pass a valid API key.", FailureReason.AUTH_DEAD),
        ("PERMISSION_DENIED: caller does not have permission", FailureReason.AUTH_DEAD),
        # Cross-key/expired File API ref: 403 but must NOT be treated as a dead key,
        # else uploading with key A + generating with key B cascades the whole pool.
        (
            "403 PERMISSION_DENIED. You do not have permission to access the File abc123 or it may not exist.",
            FailureReason.STALE_MEDIA,
        ),
        ("daily quota exceeded for this project", FailureReason.QUOTA_EXHAUSTED),
        ("404 models/foo is not found for API version v1beta", FailureReason.NOT_FOUND),
        ("429 Resource has been exhausted (e.g. check quota).", FailureReason.RATE_LIMIT),
        ("Too Many Requests", FailureReason.RATE_LIMIT),
        ("503 The model is overloaded. Please try again later.", FailureReason.HIGH_DEMAND),
        ("some totally unrelated error", FailureReason.UNKNOWN),
    ],
)
def test_classify_gemini_error(message, expected_reason):
    result = classify_gemini_error(message)
    assert result.reason == expected_reason


@pytest.fixture
def provider() -> GeminiProvider:
    return GeminiProvider(
        model_priority=["gemini-2.5-flash"],
        model_aliases={"gemini-3.1": "gemini-3.1-flash-preview"},
        quota_table={"gemini-2.5-flash": {"rpm": 15, "tpm": 1000000, "rpd": 500}},
    )


def test_resolve_model_uses_alias(provider):
    assert provider.resolve_model("gemini-3.1") == "gemini-3.1-flash-preview"
    assert provider.resolve_model("gemini-2.5-flash") == "gemini-2.5-flash"
    assert provider.resolve_model(None) == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_generate_success_parses_usage_metadata(provider, monkeypatch):
    mock_response = SimpleNamespace(
        text="Hello there  ",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8, total_token_count=20),
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr("app.providers.gemini.provider.genai.Client", lambda **kwargs: mock_client)

    ctx = GenerateContext(prompt_text="hi", model="gemini-2.5-flash", api_key="key-aaaa1111")
    result = await provider.generate(ctx)

    assert result.text == "Hello there"
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.total_tokens == 20


@pytest.mark.asyncio
async def test_generate_propagates_sdk_error(provider, monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("429 Resource has been exhausted")
    monkeypatch.setattr("app.providers.gemini.provider.genai.Client", lambda **kwargs: mock_client)

    ctx = GenerateContext(prompt_text="hi", model="gemini-2.5-flash", api_key="key-aaaa1111")
    with pytest.raises(RuntimeError):
        await provider.generate(ctx)


def _upload_client(get_side_effect):
    client = MagicMock()
    client.files.upload.return_value = SimpleNamespace(name="files/abc", display_name="clip.mp4")
    client.files.get.side_effect = get_side_effect
    return client


def test_upload_poll_rides_out_a_429_instead_of_failing(provider, monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    client = _upload_client([
        Exception("429 Too Many Requests"),
        SimpleNamespace(name="files/abc", state="PROCESSING"),
        SimpleNamespace(name="files/abc", state="ACTIVE"),
    ])
    monkeypatch.setattr("app.providers.gemini.provider.genai.Client", lambda **kwargs: client)
    monkeypatch.setattr("app.providers.gemini.provider.time.sleep", lambda s: None)

    ref = provider._upload_sync(str(media), "key-1")

    assert ref.name == "files/abc"
    assert client.files.upload.call_count == 1
    client.files.delete.assert_not_called()


def test_upload_deletes_file_it_gives_up_on(provider, monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    client = _upload_client([SimpleNamespace(name="files/abc", state="FAILED")])
    monkeypatch.setattr("app.providers.gemini.provider.genai.Client", lambda **kwargs: client)
    monkeypatch.setattr("app.providers.gemini.provider.time.sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        provider._upload_sync(str(media), "key-1")

    client.files.delete.assert_called_once_with(name="files/abc")


def test_upload_poll_non_transient_error_still_raises_and_cleans_up(provider, monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    client = _upload_client([Exception("400 API key not valid")])
    monkeypatch.setattr("app.providers.gemini.provider.genai.Client", lambda **kwargs: client)
    monkeypatch.setattr("app.providers.gemini.provider.time.sleep", lambda s: None)

    with pytest.raises(Exception, match="API key not valid"):
        provider._upload_sync(str(media), "key-1")

    client.files.delete.assert_called_once_with(name="files/abc")
