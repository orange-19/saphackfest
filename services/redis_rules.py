"""
services/redis_rules.py
-----------------------
Layer 1: O(1) static rule validation using Redis.
Rules are seeded at startup by main.py and re-read here on every request.
All checks are fast in-memory lookups — no LLM, no DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class Layer1Result:
    passed: bool
    rule_name: str        # "PASS" or the name of the violated rule
    message: str          # human-readable verdict
    delta_pct: float


async def evaluate_static_rules(
    redis: aioredis.Redis,
    product_id: str,
    current_price: float,
    proposed_price: float,
    actor_type: str,
    metadata: dict,
) -> Layer1Result:
    """
    Checks all static governance rules against the pricing update.
    Returns a Layer1Result indicating pass/fail and which rule triggered.
    """
    keys = [
        "rule:max_increase_pct",
        "rule:sentiment_lock_active",
        "rule:essential_categories",
    ]
    values = await redis.mget(*keys)
    rules = dict(zip(keys, values))

    max_increase   = float(rules.get("rule:max_increase_pct")   or 50.0)
    sentiment_lock = (rules.get("rule:sentiment_lock_active") or "false").lower() == "true"
    essential_cats = [
        c.strip().lower()
        for c in (rules.get("rule:essential_categories") or "food,medicine,water,fuel,baby").split(",")
    ]

    delta_pct = ((proposed_price - current_price) / current_price) * 100

    # --- Rule 1: Absolute floor ($1.00 limit) ---
    if proposed_price < 1.0:
        msg = f"Critical Threshold Violation: proposed ${proposed_price:.2f} is below the $1.00 minimum."
        logger.warning(f"[Layer1] {msg} | product={product_id}")
        return Layer1Result(passed=False, rule_name="floor_price_rule", message=msg, delta_pct=delta_pct)

    # --- Rule 2: AI vs Human Limit Logic ---
    if actor_type.lower() == "human":
        if delta_pct < -80.0:
            msg = "Policy 2 Violation: Unauthorized Override. Exceeds 80% human limit."
            logger.warning(f"[Layer1] {msg} | product={product_id}")
            return Layer1Result(passed=False, rule_name="human_discount_rule", message=msg, delta_pct=delta_pct)
    else:
        if delta_pct < -65.0:
            msg = "Policy 1 Violation: Human session logged out. AI Hallucination detected."
            logger.warning(f"[Layer1] {msg} | product={product_id}")
            return Layer1Result(passed=False, rule_name="ai_discount_rule", message=msg, delta_pct=delta_pct)

    # --- Rule 3: Max price increase cap ---
    if delta_pct > max_increase:
        msg = f"REJECTED by max_increase_rule: +{delta_pct:.2f}% increase exceeds the +{max_increase:.0f}% hard cap."
        logger.warning(f"[Layer1] {msg} | product={product_id}")
        return Layer1Result(passed=False, rule_name="max_increase_rule", message=msg, delta_pct=delta_pct)

    # --- Rule 4: Sentiment / crisis lock on essential goods ---
    product_category = metadata.get("category", "").lower()
    if sentiment_lock and any(cat in product_category for cat in essential_cats):
        msg = (
            f"REJECTED by sentiment_lock_rule: A market crisis lock is active. "
            f"Essential goods in category '{product_category}' cannot be repriced."
        )
        logger.warning(f"[Layer1] {msg} | product={product_id}")
        return Layer1Result(
            passed=False, rule_name="sentiment_lock_rule", message=msg, delta_pct=delta_pct
        )

    # --- All rules passed ---
    return Layer1Result(
        passed=True,
        rule_name="PASS",
        message=f"Passed all static rules. Δ={delta_pct:+.2f}%",
        delta_pct=delta_pct,
    )
