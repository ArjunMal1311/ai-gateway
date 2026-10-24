import base64
import json

import httpx
import pytest

from app.models.enums import FailureReason
from app.providers.base import GenerateContext
from app.providers.openai_compat.errors import UnsupportedMediaError, classify_openai_compat_error
from app.providers.openai_compat.vendors import GroqProvider
from app.providers.registry import ProviderRegistry


def _provider(handler, **kwargs) -> GroqProvider:
    return GroqProvider(
        model_priority=["text-model", "vision-model"],
        model_aliases={"vis": "vision-model"},
        quota_table={"text-model": {"rpm": 30, "tpm": 1000, "rpd": 100}},
        vision_models=["vision-model"],
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _ok(text="Hello\nthere  ", usage=None):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
    )


@pytest.mark.parametrize(
    "message,expected_reason",
    [
        ("HTTP 401: invalid api key", FailureReason.AUTH_DEAD),
        ("HTTP 403: forbidden", FailureReason.AUTH_DEAD),
        ("HTTP 429: Rate limit reached for model x on tokens per minute (TPM). Try again in 7s.", FailureReason.RATE_LIMIT),
        ("HTTP 429: Rate limit reached on tokens per day (TPD): Limit 200000", FailureReason.QUOTA_EXHAUSTED),
        ("HTTP 429: token_quota_exceeded", FailureReason.QUOTA_EXHAUSTED),
        ("HTTP 404: The model `x` does not exist", FailureReason.NOT_FOUND),
        ("HTTP 503: Service unavailable", FailureReason.HIGH_DEMAND),
        ("HTTP 498: Flex tier capacity exceeded, over capacity", FailureReason.HIGH_DEMAND),
        ("HTTP 400: context length exceeded", FailureReason.UNKNOWN),
        ("groq accepts text and images only, got video/mp4.", FailureReason.UNKNOWN),
    ],
)
def test_classify_openai_compat_error(message, expected_reason):
    assert classify_openai_compat_error(message).reason == expected_reason


def test_rate_limit_honours_retry_delay():
    result = classify_openai_compat_error("HTTP 429: slow down (retry in 12s)")
    assert result.reason == FailureReason.RATE_LIMIT
    assert result.cooldown_seconds == 12.0


@pytest.mark.asyncio
async def test_generate_text_success_sends_bearer_key_and_parses_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok()

    provider = _provider(handler)
    result = await provider.generate(
        GenerateContext(prompt_text="hi", prompt_parts=["part one"], model="text-model", api_key="gsk_test")
    )

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    # Text-only is sent as a plain string, not a list of parts.
    assert seen["body"]["messages"] == [{"role": "user", "content": "part one\nhi"}]
    assert seen["body"]["model"] == "text-model"
    assert result.text == "Hello there"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (12, 8, 20)


@pytest.mark.asyncio
async def test_generate_http_error_carries_status_for_classifier():
    provider = _provider(lambda r: httpx.Response(429, text="Rate limit reached", headers={"retry-after": "9"}))
    with pytest.raises(Exception) as exc_info:
        await provider.generate(GenerateContext(prompt_text="hi", model="text-model", api_key="k"))

    classification = provider.classify_error(str(exc_info.value))
    assert classification.reason == FailureReason.RATE_LIMIT
    assert classification.cooldown_seconds == 9.0


@pytest.mark.asyncio
async def test_generate_empty_text_raises():
    provider = _provider(lambda r: _ok(text="   "))
    with pytest.raises(RuntimeError, match="no text"):
        await provider.generate(GenerateContext(prompt_text="hi", model="text-model", api_key="k"))


@pytest.mark.asyncio
async def test_video_is_refused_before_any_http_call(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00")

    def handler(request):
        raise AssertionError("no HTTP call expected")

    provider = _provider(handler)
    with pytest.raises(UnsupportedMediaError):
        await provider.generate(
            GenerateContext(prompt_text="describe", media_paths=[str(video)], model="vision-model", api_key="k")
        )


@pytest.mark.asyncio
async def test_image_on_text_only_model_is_refused(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG")
    provider = _provider(lambda r: _ok())
    with pytest.raises(UnsupportedMediaError, match="does not accept images"):
        await provider.generate(
            GenerateContext(prompt_text="describe", media_paths=[str(image)], model="text-model", api_key="k")
        )


@pytest.mark.asyncio
async def test_image_on_vision_model_goes_inline_as_data_url(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNGdata")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _ok()

    provider = _provider(handler)
    await provider.generate(
        GenerateContext(prompt_text="describe", media_paths=[str(image)], model="vision-model", api_key="k")
    )

    content = seen["body"]["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNGdata").decode()},
    }
    assert content[1] == {"type": "text", "text": "describe"}


def test_registry_loads_all_configured_providers_with_vision_models():
    registry = ProviderRegistry("config/models.yaml")

    assert {"gemini", "groq", "cerebras", "mistral"} <= set(registry.names())
    groq = registry.get("groq")
    assert groq.resolve_model("llama-4-scout") == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert groq.resolve_model(None) == groq.model_priority()[0]
    assert "meta-llama/llama-4-scout-17b-16e-instruct" in groq._vision_models
    assert registry.get("mistral")._vision_models
