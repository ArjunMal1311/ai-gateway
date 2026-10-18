from __future__ import annotations

import re

from app.models.enums import FailureReason
from app.providers.base import FailureClassification

_AUTH_MARKERS = (
    "permission_denied",
    "unrestricted key",
    "unrestricted keys",
    "api key not valid",
    "api_key_invalid",
    "unauthenticated",
    "service account",
)
# A 403 on an uploaded File API ref ("You do not have permission to access the
# File ... or it may not exist") is NOT a dead key — it means the file was
# uploaded under a different key/project or has expired. Must be caught before
# _AUTH_MARKERS' "permission_denied", otherwise it dead-cools a healthy key for
# an hour and cascades the whole pool. Retryable via re-upload, no cooldown.
_STALE_MEDIA_RE = re.compile(r"permission to access the file|file .*may not exist")
_QUOTA_MARKERS = ("daily", "project quota", "daily quota")
_NOT_FOUND_RE = re.compile(r"\b404\b|not_found|not found")
_RATE_LIMIT_RE = re.compile(
    r"\b429\b|rate limit|quota|resource has been exhausted|too many requests|exhausted"
)
_HIGH_DEMAND_RE = re.compile(r"\b503\b|unavailable|overloaded|high demand|temporary")

# A 429 says which limit was hit (quotaId / quota_limit / quota_limit_value), so it
# is read instead of treated as one generic "rate limit":
#   limit 0          -> this key can never call it (zero-quota project, or a paid-only
#                       model on a free key): QUOTA_EXHAUSTED, 1-hour cooldown.
#   per day          -> waiting minutes won't help until the daily reset: QUOTA_EXHAUSTED.
#   per region       -> the limit is the whole Google project, not one model, so the
#                       key is cooled on every model (scope "key").
#   anything else    -> per-minute: RATE_LIMIT, honouring Google's retryDelay.
# 2026-10-17: 3,766 429s this month were ApiRequestsPerMinutePerProjectPerRegion with
# quota_limit_value '0' on 11 keys, each re-tried every ~30s on every model.
_IS_429_RE = re.compile(r"\b429\b|resource_exhausted|resource has been exhausted|too many requests")
_ZERO_LIMIT_RE = re.compile(r"quota_limit_value['\"]?\s*:\s*['\"]0['\"]|\blimit: 0\b")
_PER_DAY_RE = re.compile(r"perday|per day|per_day")
_PROJECT_WIDE_RE = re.compile(r"perregion|for a region|quota metric 'api requests'")
_RETRY_DELAY_RE = re.compile(r"retrydelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s|retry in (\d+(?:\.\d+)?)s")


def _retry_delay_seconds(message: str):
    match = _RETRY_DELAY_RE.search(message)
    if not match:
        return None
    return float(match.group(1) or match.group(2))


def _classify_429(message: str) -> FailureClassification:
    scope = "key" if _PROJECT_WIDE_RE.search(message) else "key_model"
    if _ZERO_LIMIT_RE.search(message) or _PER_DAY_RE.search(message):
        return FailureClassification(reason=FailureReason.QUOTA_EXHAUSTED, scope=scope, retryable=True)
    return FailureClassification(
        reason=FailureReason.RATE_LIMIT,
        scope=scope,
        cooldown_seconds=_retry_delay_seconds(message),
        retryable=True,
    )


def classify_gemini_error(error: str) -> FailureClassification:
    """Ports the error-string classification that lived in both
    api_key_pool.report_failure() and gemini_util.py's except block in the source repo,
    merged into the single seam a Provider is responsible for.
    """
    message = str(error).lower()

    if _STALE_MEDIA_RE.search(message):
        return FailureClassification(reason=FailureReason.STALE_MEDIA, scope="key_model", retryable=True)

    if any(marker in message for marker in _AUTH_MARKERS):
        return FailureClassification(reason=FailureReason.AUTH_DEAD, scope="key", retryable=True)

    if _IS_429_RE.search(message):
        return _classify_429(message)

    if any(marker in message for marker in _QUOTA_MARKERS):
        return FailureClassification(reason=FailureReason.QUOTA_EXHAUSTED, scope="key_model", retryable=True)

    if _NOT_FOUND_RE.search(message):
        # A 404 here ("model not found" / "no longer available to new users") is a
        # property of the calling key's own Google Cloud project, not the model
        # itself — sibling projects created at other times routinely differ on
        # which models they can see. Scope key_model (not model): report_failure()
        # only escalates to a pool-wide model blacklist once every key has
        # independently confirmed the same 404, same as QUOTA_EXHAUSTED.
        return FailureClassification(reason=FailureReason.NOT_FOUND, scope="key_model", retryable=True)

    if _RATE_LIMIT_RE.search(message):
        return FailureClassification(reason=FailureReason.RATE_LIMIT, scope="key_model", retryable=True)

    if _HIGH_DEMAND_RE.search(message):
        return FailureClassification(reason=FailureReason.HIGH_DEMAND, scope="key_model", retryable=True)

    return FailureClassification(reason=FailureReason.UNKNOWN, scope="key_model", retryable=True)
