"""
services/news_sentinel.py
-------------------------
Mock "Live Sentiment Circuit Breaker" service.
In a real system this would call a news API or a market-data feed.
Here we use a small in-memory list of randomly-selected crisis events
that fire probabilistically to simulate real-world disruptions.

The LangGraph `sentiment_circuit_breaker` node calls `check_sentiment()`
and, if a crisis is detected, writes `rule:sentiment_lock_active=true`
to Redis so Layer 1 also rejects essential-goods changes immediately.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock crisis event library
# ---------------------------------------------------------------------------
_CRISIS_EVENTS: list[dict] = [
    {
        "headline": "Supply chain disruption: Port workers strike hits food imports",
        "category_lock": "food",
        "severity": 0.92,
    },
    {
        "headline": "Earthquake emergency: Government freezes essential goods pricing",
        "category_lock": "all",
        "severity": 0.98,
    },
    {
        "headline": "Fuel shortage alert: OPEC emergency session causes oil spike",
        "category_lock": "fuel",
        "severity": 0.85,
    },
    {
        "headline": "Pandemic resurgence: Panic buying of medicine and baby goods reported",
        "category_lock": "medicine",
        "severity": 0.90,
    },
    {
        "headline": "Flood warning: Water distribution severely impacted in 3 regions",
        "category_lock": "water",
        "severity": 0.88,
    },
]

# Base probability of a crisis event firing per request
_CRISIS_PROBABILITY = 0.00


@dataclass
class SentimentResult:
    crisis_detected: bool
    headline: str = ""
    severity: float = 0.0
    category_lock: str = ""


async def check_sentiment(
    redis: aioredis.Redis,
    metadata: dict | None = None,
) -> SentimentResult:
    """
    Probabilistically fires a mock crisis event.
    If triggered:
      - Writes `rule:sentiment_lock_active=true` to Redis with 5-min TTL
      - Returns SentimentResult with event details

    Redis key TTL means the lock auto-expires, simulating a crisis resolution.
    """
    # Check if a lock is already active (from a previous request this session)
    existing_lock = await redis.get("rule:sentiment_lock_active")
    if existing_lock == "true":
        # Fetch the last event headline for context
        headline = await redis.get("rule:sentiment_headline") or "Ongoing crisis lock active."
        return SentimentResult(
            crisis_detected=True,
            headline=headline,
            severity=1.0,
            category_lock="all",
        )

    # Roll the dice — 10% chance of a new crisis
    if random.random() < _CRISIS_PROBABILITY:
        event = random.choice(_CRISIS_EVENTS)

        # Write the lock to Redis with 5-minute TTL
        await redis.set("rule:sentiment_lock_active", "true", ex=300)
        await redis.set("rule:sentiment_headline", event["headline"], ex=300)

        logger.warning(
            f"[SentimentBreaker] CRISIS DETECTED: '{event['headline']}' "
            f"| severity={event['severity']} | lock={event['category_lock']}"
        )

        return SentimentResult(
            crisis_detected=True,
            headline=event["headline"],
            severity=event["severity"],
            category_lock=event["category_lock"],
        )

    # Clear any stale lock
    await redis.delete("rule:sentiment_lock_active")
    return SentimentResult(crisis_detected=False)
