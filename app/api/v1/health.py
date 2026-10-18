from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.models.responses import HealthResponse
from app.redis_client import ping

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(request: Request, response: Response) -> HealthResponse:
    redis_ok = await ping()
    pools = getattr(request.app.state, "pools", {})
    keys_configured = sum(pool.size() for pool in pools.values())

    if not redis_ok or keys_configured == 0:
        response.status_code = 503
        return HealthResponse(status="not_ready", redis_ok=redis_ok, keys_configured=keys_configured)

    # "Up" is not "usable": on 2026-10-17 this said ok all day while 10 of 34 keys were
    # dead and the success rate was 0.3%. Count keys that aren't dead, and whether any
    # model is left to call at all; still 200, so callers can tell degraded from down.
    usable_keys = 0
    no_model = []
    for name, pool in pools.items():
        if await pool.get_available_model() is None:
            no_model.append(name)
            continue
        status = await pool.get_pool_status()
        usable_keys += status["total_keys"] - status["permanently_blocked"]

    min_usable = request.app.state.settings.min_usable_keys
    detail = None
    if no_model:
        detail = f"every model is cooled for: {', '.join(no_model)}"
    elif usable_keys < min_usable:
        detail = f"only {usable_keys} usable keys (minimum {min_usable})"

    return HealthResponse(
        status="degraded" if detail else "ok",
        redis_ok=redis_ok,
        keys_configured=keys_configured,
        usable_keys=usable_keys,
        detail=detail,
    )
