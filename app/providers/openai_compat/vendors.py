from __future__ import annotations

import os

from app.providers.openai_compat.provider import OpenAICompatProvider


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class CerebrasProvider(OpenAICompatProvider):
    name = "cerebras"
    base_url = "https://api.cerebras.ai/v1"


class MistralProvider(OpenAICompatProvider):
    name = "mistral"
    base_url = "https://api.mistral.ai/v1"


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"


class NvidiaProvider(OpenAICompatProvider):
    name = "nvidia"
    base_url = "https://integrate.api.nvidia.com/v1"


class ZaiProvider(OpenAICompatProvider):
    name = "zai"
    base_url = "https://api.z.ai/api/paas/v4"


class CloudflareProvider(OpenAICompatProvider):
    name = "cloudflare"

    def __init__(self, *args, account_id: str = "", **kwargs):
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        account_path = self.account_id or "missing-account-id"
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_path}/ai/v1"
        super().__init__(*args, **kwargs)
