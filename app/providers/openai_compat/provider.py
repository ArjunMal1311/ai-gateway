from __future__ import annotations

import asyncio
import base64
import mimetypes
from typing import Optional

import httpx

from app.providers.base import FailureClassification, GenerateContext, Provider, ProviderResult
from app.providers.openai_compat.errors import (
    ProviderHTTPError,
    UnsupportedMediaError,
    classify_openai_compat_error,
)

# Images go inline as base64 data URLs; there is no File API on these providers, so
# requires_file_upload() stays False and anything bigger than this is refused up front.
_INLINE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 120.0


class OpenAICompatProvider(Provider):
    """Base for any vendor that speaks POST {base_url}/chat/completions with a Bearer key.
    A concrete vendor only sets `name` and `base_url`; model lists, quotas and which models
    accept images come from its models.yaml section.

    Text and images only. Video/audio raise UnsupportedMediaError before any HTTP call.
    """

    name: str
    base_url: str

    def __init__(
        self,
        model_priority: list[str],
        model_aliases: dict[str, str],
        quota_table: dict[str, dict],
        vision_models: Optional[list[str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._model_priority = model_priority
        self._model_aliases = model_aliases
        self._quota_table = quota_table
        self._vision_models = set(vision_models or [])
        # One pooled client per provider; the key is a per-request header, so keys share
        # connections to the same host instead of paying a TLS handshake each.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=_DEFAULT_TIMEOUT_SECONDS,
            transport=transport,
        )

    def model_priority(self) -> list[str]:
        return self._model_priority

    def model_aliases(self) -> dict[str, str]:
        return self._model_aliases

    def quota_table(self) -> dict[str, dict]:
        return self._quota_table

    def classify_error(self, error: str) -> FailureClassification:
        return classify_openai_compat_error(error)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- message building ----------

    @staticmethod
    def _image_part(mime_type: str, data: bytes) -> dict:
        encoded = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}

    def _build_content(self, ctx: GenerateContext) -> list[dict]:
        content: list[dict] = []

        for part in ctx.prompt_parts or []:
            if isinstance(part, dict) and "inline_data" in part:
                mime_type = part["inline_data"]["mime_type"]
                if not mime_type.startswith("image/"):
                    raise UnsupportedMediaError(f"{self.name} accepts text and images only, got {mime_type}.")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{part['inline_data']['data']}"},
                    }
                )
            else:
                content.append({"type": "text", "text": str(part)})

        for media_path in ctx.media_paths:
            mime_type = mimetypes.guess_type(media_path)[0] or "application/octet-stream"
            if not mime_type.startswith("image/"):
                raise UnsupportedMediaError(f"{self.name} accepts text and images only, got {mime_type}.")
            with open(media_path, "rb") as f:
                data = f.read(_INLINE_IMAGE_MAX_BYTES + 1)
            if len(data) > _INLINE_IMAGE_MAX_BYTES:
                raise UnsupportedMediaError(f"Image exceeds {_INLINE_IMAGE_MAX_BYTES} bytes: {media_path}")
            content.append(self._image_part(mime_type, data))

        if ctx.prompt_text:
            content.append({"type": "text", "text": ctx.prompt_text})

        if any(c["type"] == "image_url" for c in content) and ctx.model not in self._vision_models:
            raise UnsupportedMediaError(f"{self.name} model {ctx.model} does not accept images.")
        return content

    # ---------- generate ----------

    async def _post(self, ctx: GenerateContext, content: list[dict]) -> dict:
        # Text-only goes as a plain string: some text models reject the list-of-parts form.
        has_image = any(c["type"] == "image_url" for c in content)
        message_content = content if has_image else "\n".join(c["text"] for c in content)

        response = await self._client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {ctx.api_key}"},
            json={"model": ctx.model, "messages": [{"role": "user", "content": message_content}]},
        )
        if response.status_code >= 400:
            retry_after = response.headers.get("retry-after")
            try:
                retry_seconds = float(retry_after) if retry_after else None
            except ValueError:
                retry_seconds = None
            raise ProviderHTTPError(response.status_code, response.text, retry_seconds)
        return response.json()

    async def generate(self, ctx: GenerateContext) -> ProviderResult:
        content = self._build_content(ctx)
        if not content:
            raise UnsupportedMediaError(f"{self.name} request has no prompt content.")

        try:
            if ctx.timeout_seconds and ctx.timeout_seconds > 0:
                body = await asyncio.wait_for(self._post(ctx, content), timeout=ctx.timeout_seconds)
            else:
                body = await self._post(ctx, content)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"{self.name} request timed out after {ctx.timeout_seconds:.1f}s.") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"{self.name} request timed out: {exc!r}") from exc

        choices = body.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") if choices else None) or ""
        if not text.strip():
            finish_reason = choices[0].get("finish_reason") if choices else None
            raise RuntimeError(f"{self.name} returned no text (finish_reason={finish_reason}).")

        usage = body.get("usage") or {}
        return ProviderResult(
            # Same flattening as GeminiProvider so callers see one output shape per gateway.
            text=text.strip().replace("\n", " "),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
