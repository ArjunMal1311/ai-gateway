from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request

from app.deps import get_call_tracker, get_key_pool
from app.pool.redis_keys import key_suffix

router = APIRouter()


@router.get("/usage/summary")
async def usage_summary(request: Request, provider: str = "gemini"):
    pool = get_key_pool(request, provider)
    tracker = get_call_tracker(request, provider)
    suffixes = [key_suffix(k) for k in pool.api_keys]
    return await tracker.get_quota_summary(suffixes)


@router.get("/usage/keys")
async def usage_keys(request: Request, provider: str = "gemini", day: Optional[str] = None):
    """Per-key calls, failures and tokens for one UTC day (YYYYMMDD, default today), by
    model, plus each key's current status. For the tray and `socials ai keys`."""
    pool = get_key_pool(request, provider)
    tracker = get_call_tracker(request, provider)
    day = day or datetime.now(timezone.utc).strftime("%Y%m%d")
    models = list(tracker.quota_table)
    suffixes = [key_suffix(k) for k in pool.api_keys]

    pipe = tracker.redis.pipeline(transaction=False)
    for suffix in suffixes:
        for model in models:
            pipe.get(tracker.rk.tracker_rpd(model, suffix, day))
            pipe.get(tracker.rk.tracker_failures_day(model, suffix, day))
            pipe.get(tracker.rk.tracker_tokens_day(model, suffix, day))
    values = iter(int(v) if v else 0 for v in await pipe.execute())

    status = await pool.get_pool_status()
    state = {
        entry["suffix"]: (bucket, entry.get("retry_in_seconds", 0.0), entry.get("last_reason"))
        for bucket, entries in status["keys"].items()
        for entry in entries
    }

    keys = []
    for suffix in suffixes:
        per_model = {}
        for model in models:
            calls, failures, tokens = next(values), next(values), next(values)
            if calls or failures:
                per_model[model] = {"calls": calls, "failures": failures, "tokens": tokens}
        bucket, retry_in, reason = state.get(suffix, ("available", 0.0, None))
        keys.append({
            "suffix": suffix,
            "status": bucket,
            "retry_in_seconds": retry_in,
            "last_reason": reason,
            "calls": sum(m["calls"] for m in per_model.values()),
            "failures": sum(m["failures"] for m in per_model.values()),
            "tokens": sum(m["tokens"] for m in per_model.values()),
            "models": per_model,
        })
    return {"day": day, "provider": provider, "keys": keys}
