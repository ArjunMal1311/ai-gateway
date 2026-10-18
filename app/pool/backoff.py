from __future__ import annotations

import random

DEFAULT_BASES = {
    "rate_limit": 30.0,
    "high_demand": 8.0,
}
MAX_BACKOFF_SECONDS = 1000.0
MAX_STREAK = 8
JITTER_FRACTION = 0.25
# How long a key+model remembers its last failure after the cooldown ends. The ladder
# (30 -> 60 -> 120 -> ... -> 1000s) only climbs if the streak outlives the cooldown;
# when both expired together every 429 restarted at 30s (bug until 2026-10-17).
STREAK_MEMORY_SECONDS = 3600.0


def compute_backoff_seconds(
    failure_type: str,
    streak: int,
    bases: dict[str, float] | None = None,
    max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
    max_streak: int = MAX_STREAK,
    jitter_fraction: float = JITTER_FRACTION,
) -> float:
    """Exponential backoff with jitter, ported verbatim from the original
    APIKeyPool._compute_backoff_seconds. streak=1 -> base delay; each further
    consecutive failure of the same type doubles it, capped at max_streak
    doublings and max_backoff_seconds overall, plus up to jitter_fraction of
    extra random jitter so many workers don't retry in lockstep.
    """
    bases = bases or DEFAULT_BASES
    base = bases.get(failure_type, DEFAULT_BASES["rate_limit"])
    capped_streak = min(max(streak, 1), max_streak)
    exponential = base * (2 ** (capped_streak - 1))
    delay = min(exponential, max_backoff_seconds)
    jitter = random.uniform(0.0, delay * jitter_fraction)
    return min(delay + jitter, max_backoff_seconds)
