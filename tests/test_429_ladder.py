import time

import pytest

from app.models.enums import FailureReason, KeyStatus
from app.pool.backoff import compute_backoff_seconds
from app.pool.key_pool import AsyncAPIKeyPool, _utc_day
from app.pool.redis_keys import key_id
from app.providers.gemini.errors import classify_gemini_error
from tests.conftest import GEMINI_MODEL_PRIORITY

REGION_ZERO_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': \"Quota exceeded for quota metric "
    "'API requests' and limit 'Request limit per minute for a region' of service "
    "'generativelanguage.googleapis.com' for consumer 'project_number:1'.\", 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'RATE_LIMIT_EXCEEDED', "
    "'metadata': {'quota_limit_value': '0', 'quota_limit': 'ApiRequestsPerMinutePerProjectPerRegion'}}]}}"
)
REGION_BUSY_429 = REGION_ZERO_429.replace("'quota_limit_value': '0'", "'quota_limit_value': '60'")
MODEL_LIMIT_ZERO_429 = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota. * Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro"
)
PER_DAY_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '31306s'}]}}"
)
PER_MINUTE_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '41s'}]}}"
)


def test_region_limit_zero_is_dead_quota_for_whole_key():
    c = classify_gemini_error(REGION_ZERO_429)
    assert c.reason == FailureReason.QUOTA_EXHAUSTED
    assert c.scope == "key"


def test_region_per_minute_cools_whole_key():
    c = classify_gemini_error(REGION_BUSY_429)
    assert c.reason == FailureReason.RATE_LIMIT
    assert c.scope == "key"


def test_model_limit_zero_is_dead_quota_for_that_model():
    c = classify_gemini_error(MODEL_LIMIT_ZERO_429)
    assert c.reason == FailureReason.QUOTA_EXHAUSTED
    assert c.scope == "key_model"


def test_per_day_is_quota_exhausted():
    assert classify_gemini_error(PER_DAY_429).reason == FailureReason.QUOTA_EXHAUSTED


def test_per_minute_carries_googles_retry_delay():
    c = classify_gemini_error(PER_MINUTE_429)
    assert c.reason == FailureReason.RATE_LIMIT
    assert c.scope == "key_model"
    assert c.cooldown_seconds == 41.0


def test_ladder_is_30_60_120_240_480_960_then_1000():
    got = [compute_backoff_seconds("rate_limit", s, jitter_fraction=0.0) for s in range(1, 9)]
    assert got == [30, 60, 120, 240, 480, 960, 1000, 1000]
    assert all(compute_backoff_seconds("rate_limit", s) <= 1000 for s in range(1, 20))


async def _cooldown_left(pool, redis_key):
    raw = await pool.redis.get(redis_key)
    return float(raw) - time.time()


@pytest.mark.asyncio
async def test_repeated_429_climbs_the_ladder_even_after_cooldown_expires(key_pool: AsyncAPIKeyPool):
    key, model = key_pool.api_keys[0], GEMINI_MODEL_PRIORITY[0]
    kid = key_id(key)
    c = classify_gemini_error("429 Too Many Requests")
    for _ in range(3):
        await key_pool.report_failure(key, model, c)
        await key_pool.redis.delete(key_pool.rk.cooldown_keymodel(kid, model))
    await key_pool.report_failure(key, model, c)
    left = await _cooldown_left(key_pool, key_pool.rk.cooldown_keymodel(kid, model))
    assert 230 <= left <= 300
    meta = await key_pool._read_failure_meta(kid, model)
    assert int(meta["streak"]) == 4


@pytest.mark.asyncio
async def test_success_resets_the_ladder(key_pool: AsyncAPIKeyPool):
    key, model = key_pool.api_keys[0], GEMINI_MODEL_PRIORITY[0]
    kid = key_id(key)
    c = classify_gemini_error("429 Too Many Requests")
    await key_pool.report_failure(key, model, c)
    await key_pool.report_failure(key, model, c)
    await key_pool.record_success(key, model)
    await key_pool.report_failure(key, model, c)
    meta = await key_pool._read_failure_meta(kid, model)
    assert int(meta["streak"]) == 1


@pytest.mark.asyncio
async def test_retry_delay_longer_than_ladder_is_honoured(key_pool: AsyncAPIKeyPool):
    key, model = key_pool.api_keys[0], GEMINI_MODEL_PRIORITY[0]
    await key_pool.report_failure(key, model, classify_gemini_error(PER_MINUTE_429))
    left = await _cooldown_left(key_pool, key_pool.rk.cooldown_keymodel(key_id(key), model))
    assert 39 <= left <= 52


@pytest.mark.asyncio
async def test_project_wide_429_cools_key_on_every_model_without_tripping_breaker(fake_redis, settings):
    keys = ",".join(f"key-{i:04d}xxxx" for i in range(6))
    pool = AsyncAPIKeyPool(fake_redis, keys, GEMINI_MODEL_PRIORITY, settings)
    model, other = GEMINI_MODEL_PRIORITY[0], GEMINI_MODEL_PRIORITY[1]
    c = classify_gemini_error(REGION_BUSY_429)
    for key in pool.api_keys:
        await pool.report_failure(key, model, c)
    status, _ = await pool.classify_key_status(pool.api_keys[0], other, time.time())
    assert status == KeyStatus.RATE_LIMITED.value
    assert not await fake_redis.exists(pool.rk.cooldown_model(model))


@pytest.mark.asyncio
async def test_zero_quota_project_is_dead_on_every_model(key_pool: AsyncAPIKeyPool):
    key = key_pool.api_keys[0]
    await key_pool.report_failure(key, GEMINI_MODEL_PRIORITY[0], classify_gemini_error(REGION_ZERO_429))
    for model in GEMINI_MODEL_PRIORITY:
        status, _ = await key_pool.classify_key_status(key, model, time.time())
        assert status == KeyStatus.DEAD_QUOTA.value


@pytest.mark.asyncio
async def test_model_limit_zero_does_not_trip_breaker(fake_redis, settings):
    keys = ",".join(f"key-{i:04d}yyyy" for i in range(6))
    pool = AsyncAPIKeyPool(fake_redis, keys, GEMINI_MODEL_PRIORITY, settings)
    model = GEMINI_MODEL_PRIORITY[0]
    for key in pool.api_keys[:5]:
        await pool.report_failure(key, model, classify_gemini_error(MODEL_LIMIT_ZERO_429))
    assert not await fake_redis.exists(pool.rk.cooldown_model(model))
    status, _ = await pool.classify_key_status(pool.api_keys[5], model, time.time())
    assert status == KeyStatus.AVAILABLE.value


@pytest.mark.asyncio
async def test_acquire_prefers_key_least_used_on_this_model_today(fake_redis, settings):
    pool = AsyncAPIKeyPool(fake_redis, "key-busy1111,key-idle2222", GEMINI_MODEL_PRIORITY, settings)
    model = GEMINI_MODEL_PRIORITY[0]
    await fake_redis.set(pool.rk.usage_rpd(key_id("key-busy1111"), model, _utc_day()), 50)
    await fake_redis.set(pool.rk.usage_key(key_id("key-idle2222")), 100000)
    key, got_model = await pool.acquire_key()
    assert (key, got_model) == ("key-idle2222", model)
    assert int(await fake_redis.get(pool.rk.usage_rpd(key_id("key-idle2222"), model, _utc_day()))) == 1


@pytest.mark.asyncio
async def test_busy_healthy_keys_never_blacklist_the_model(fake_redis, settings):
    import asyncio

    pool = AsyncAPIKeyPool(fake_redis, "key-good1111,key-good2222,key-dead3333,key-dead4444", GEMINI_MODEL_PRIORITY, settings)
    for dead in ("key-dead3333", "key-dead4444"):
        await pool.report_failure(dead, GEMINI_MODEL_PRIORITY[0], classify_gemini_error(REGION_ZERO_429))
    first, _ = await pool.acquire_key()
    second, _ = await pool.acquire_key()
    assert {first, second} == {"key-good1111", "key-good2222"}

    async def release_soon():
        await asyncio.sleep(0.5)
        await pool.release_key(first)

    releaser = asyncio.create_task(release_soon())
    third, model = await pool.acquire_key(max_wait_seconds=5)
    await releaser
    assert (third, model) == (first, GEMINI_MODEL_PRIORITY[0])
    for m in GEMINI_MODEL_PRIORITY:
        assert not await fake_redis.exists(pool.rk.cooldown_model(m))
