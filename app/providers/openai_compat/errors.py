from __future__ import annotations

import re

from app.models.enums import FailureReason
from app.providers.base import FailureClassification


class ProviderHTTPError(RuntimeError):
    """Non-2xx reply from an OpenAI-compatible endpoint. The message always starts with
    "HTTP <status>" so classify_openai_compat_error() can key off the status code, the same
    string-in/classification-out seam the Gemini provider uses.
    """

    def __init__(self, status_code: int, body: str, retry_after_seconds: float | None = None):
        suffix = f" (retry in {retry_after_seconds:g}s)" if retry_after_seconds else ""
        super().__init__(f"HTTP {status_code}: {body[:500]}{suffix}")
        self.status_code = status_code


class UnsupportedMediaError(ValueError):
    """The request carries media this provider/model can't take (video, audio, or an image on
    a text-only model). Deliberately matches no classifier pattern -> FailureReason.UNKNOWN,
    which cools nothing: the cause is the request, not the key, so cooling would only hide it.
    """


_AUTH_RE = re.compile(r"\bhttp (401|403)\b|invalid api key|invalid_api_key|unauthorized")
_NOT_FOUND_RE = re.compile(r"\bhttp 404\b|model_not_found|does not exist|model .* not found|unknown model")
_IS_429_RE = re.compile(r"\bhttp 429\b|rate.?limit|too_many_requests|too many requests")
# Daily budgets (Groq "tokens per day (TPD)" / "requests per day (RPD)", Cerebras
# "token_quota_exceeded") don't recover in minutes: park the key for the hour cap instead of
# re-trying every backoff step. Z.ai sends "paid model, no balance" as a 429 too (code 1113,
# "Insufficient balance or no resource package. Please recharge.") — that never recovers.
_DAILY_RE = re.compile(
    r"per day|\btpd\b|\brpd\b|daily|token_quota_exceeded|quota_exceeded|insufficient balance|recharge"
)
_HIGH_DEMAND_RE = re.compile(
    r"\bhttp (500|502|503|504)\b|overloaded|over capacity|unavailable|queue_exceeded|try again later"
)
_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s|try again in (\d+(?:\.\d+)?)s")


def _retry_delay_seconds(message: str):
    match = _RETRY_DELAY_RE.search(message)
    if not match:
        return None
    return float(match.group(1) or match.group(2))


def classify_openai_compat_error(error: str) -> FailureClassification:
    message = str(error).lower()

    if _AUTH_RE.search(message):
        return FailureClassification(reason=FailureReason.AUTH_DEAD, scope="key", retryable=True)

    if _IS_429_RE.search(message):
        if _DAILY_RE.search(message):
            return FailureClassification(reason=FailureReason.QUOTA_EXHAUSTED, scope="key_model", retryable=True)
        return FailureClassification(
            reason=FailureReason.RATE_LIMIT,
            scope="key_model",
            cooldown_seconds=_retry_delay_seconds(message),
            retryable=True,
        )

    if _NOT_FOUND_RE.search(message):
        # Same per-key scoping as Gemini's 404: one key's account lacking a model isn't
        # proof the whole pool lacks it.
        return FailureClassification(reason=FailureReason.NOT_FOUND, scope="key_model", retryable=True)

    if _HIGH_DEMAND_RE.search(message):
        return FailureClassification(reason=FailureReason.HIGH_DEMAND, scope="key_model", retryable=True)

    return FailureClassification(reason=FailureReason.UNKNOWN, scope="key_model", retryable=True)
