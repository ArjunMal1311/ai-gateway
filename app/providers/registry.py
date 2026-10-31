from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from app.providers.base import Provider
from app.providers.gemini.provider import GeminiProvider
from app.providers.openai_compat.provider import OpenAICompatProvider
from app.providers.openai_compat.vendors import (
    CerebrasProvider,
    CloudflareProvider,
    GroqProvider,
    MistralProvider,
    OpenRouterProvider,
)

_BUILDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "cerebras": CerebrasProvider,
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
}


class ProviderRegistry:
    """Loads config/models.yaml once at startup and instantiates one Provider per
    configured section. Adding a new provider means: implement the
    Provider ABC, register its builder here, and add a section to models.yaml — no
    other code changes required. No hot-reload in v1; restart to pick up config edits.
    """

    def __init__(self, config_path: str, cloudflare_account_id: str = ""):
        self._providers: dict[str, Provider] = {}
        self._load(config_path, cloudflare_account_id)

    def _load(self, config_path: str, cloudflare_account_id: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Models config not found: {config_path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        for provider_name, section in data.items():
            builder = _BUILDERS.get(provider_name)
            if builder is None:
                continue
            kwargs = {}
            if issubclass(builder, OpenAICompatProvider):
                kwargs["vision_models"] = section.get("vision_models", [])
            if provider_name == "cloudflare":
                kwargs["account_id"] = cloudflare_account_id
            self._providers[provider_name] = builder(
                model_priority=section.get("model_priority", []),
                model_aliases=section.get("model_aliases", {}),
                quota_table=section.get("quota_table", {}),
                **kwargs,
            )

    def get(self, name: str) -> Optional[Provider]:
        return self._providers.get(name)

    def names(self) -> list[str]:
        return list(self._providers.keys())
