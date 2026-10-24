from __future__ import annotations

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
