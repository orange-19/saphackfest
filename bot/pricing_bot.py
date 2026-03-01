"""
bot/pricing_bot.py
------------------
Mock pricing bot that loads real product data from data/products.csv
(Amazon electronics review dataset) and continuously sends pricing
governance requests to the FastAPI endpoint.

CSV column mapping:
  name            → product name / SKU base
  brand           → bot identity / product category
  actual_price    → current live price
  discount_price  → proposed price (bot's target)
  no_of_ratings   → demand signal (high ratings = high demand)
  rating          → quality/trust signal
  cleaned_review_text → bot reasoning text
  sentiment       → Positive / Negative (market context)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("GOVERNANCE_API_URL", "http://localhost:8000")
SUBMIT_ENDPOINT = f"{API_URL}/api/v1/governance/submit"
DATA_FILE = Path(__file__).parent.parent / "data" / "products.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  BOT  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot pool — different "rogue" bots with distinct risk personalities
# ---------------------------------------------------------------------------
BOTS = [
    {"id": "bot-alpha-v2",          "style": "aggressive"},
    {"id": "bot-beta-pricer",       "style": "conservative"},
    {"id": "bot-clearance-engine",  "style": "clearance"},
    {"id": "bot-competitive-scan",  "style": "competitive"},
    {"id": "bot-demand-surge",      "style": "surge"},
]

# ---------------------------------------------------------------------------
# Load products from CSV
# ---------------------------------------------------------------------------

def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse a float, returning default on failure."""
    try:
        return float(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return default


def load_products(path: Path) -> list[dict]:
    """
    Read the CSV and return a list of product dicts ready for the bot.
    Filters out rows with invalid or zero prices.
    """
    products = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            actual  = _safe_float(row.get("actual_price", ""))
            discount = _safe_float(row.get("discount_price", ""))

            # Skip rows with missing/zero prices
            if actual <= 0 or discount <= 0:
                continue

            # Build a clean SKU from name (first 40 chars, alphanumeric)
            raw_name = row.get("name", f"PRODUCT-{idx}").strip()
            sku = "SKU-" + re.sub(r"[^A-Z0-9]", "-", raw_name[:40].upper()).strip("-")

            # Category from brand column
            brand = row.get("brand", "Generic").strip() or "Generic"

            # Rating (1-5 scale)
            rating = _safe_float(row.get("rating", "3"), 3.0)
            rating = max(1.0, min(5.0, rating))

            # Demand proxy from no_of_ratings
            no_ratings = int(_safe_float(row.get("no_of_ratings", "0")))

            # Bot reasoning from cleaned review
            review = row.get("cleaned_review_text", "").strip()
            sentiment = row.get("sentiment", "Neutral").strip()

            products.append({
                "sku":          sku,
                "name":         raw_name[:80],
                "brand":        brand,
                "actual_price": round(actual, 2),
                "proposed_price": round(discount, 2),
                "rating":       rating,
                "no_ratings":   no_ratings,
                "review":       review[:300] if review else "",
                "sentiment":    sentiment,
            })

    logger.info(f"Loaded {len(products)} valid products from {path.name}")
    return products


# ---------------------------------------------------------------------------
# Reasoning generator
# ---------------------------------------------------------------------------

def build_reasoning(product: dict, bot: dict, delta_pct: float) -> str:
    """
    Build a realistic bot reasoning string from the product data.
    """
    sentiment = product["sentiment"]
    review    = product["review"]
    brand     = product["brand"]
    rating    = product["rating"]
    demand    = product["no_ratings"]
    style     = bot["style"]

    demand_label = (
        "high" if demand > 50000 else
        "moderate" if demand > 10000 else
        "low"
    )
    review_snippet = review[:120] + "..." if len(review) > 120 else review

    if style == "aggressive":
        return (
            f"Aggressive repricing: {brand} product with {demand_label} demand "
            f"({demand:,} ratings). Rating {rating}/5. "
            f"Market sentiment: {sentiment}. Targeting {delta_pct:+.1f}% delta. "
            f"Customer insight: '{review_snippet}'"
        )
    elif style == "clearance":
        return (
            f"Clearance sale: inventory above threshold for {brand}. "
            f"Rating {rating}/5, {demand:,} customer ratings. "
            f"Pricing down {abs(delta_pct):.1f}% to clear stock. "
            f"Sentiment: {sentiment}. Review: '{review_snippet}'"
        )
    elif style == "competitive":
        return (
            f"Competitor analysis shows {delta_pct:+.1f}% price elasticity for {brand}. "
            f"Demand forecast: {demand_label} ({demand:,} ratings, {rating}/5 rating). "
            f"Market context: {sentiment}. '{review_snippet}'"
        )
    elif style == "surge":
        return (
            f"Demand surge detected for {brand}: {demand:,} ratings, {rating}/5. "
            f"Applying {delta_pct:+.1f}% surge pricing. "
            f"Customer sentiment: {sentiment}. '{review_snippet}'"
        )
    else:  # conservative
        return (
            f"Conservative pricing model: {brand}, rating {rating}/5, "
            f"{demand_label} demand. Proposed delta {delta_pct:+.1f}%. "
            f"Sentiment: {sentiment}. '{review_snippet}'"
        )


# ---------------------------------------------------------------------------
# Anomaly injection (7% of requests)
# ---------------------------------------------------------------------------

def maybe_inject_anomaly(product: dict, bot: dict) -> tuple[float, str]:
    """
    Occasionally inject rogue anomalies to test the governance system.
    Returns (proposed_price, reasoning).
    """
    anomaly_type = random.choice([
        "penny_drop",          # crash price to $0.01
        "10x_spike",           # spike price 10x
        "extreme_discount",    # >80% off
        "overflow_error",      # simulate a bot calculation bug
    ])

    current = product["actual_price"]

    if anomaly_type == "penny_drop":
        return 0.01, (
            f"[ANOMALY] Penny drop injected for {product['brand']} "
            f"(original ₹{current:.2f}). Possible overflow error in pricing module."
        )
    elif anomaly_type == "10x_spike":
        return round(current * 10, 2), (
            f"[ANOMALY] 10× price spike for {product['brand']}: "
            f"₹{current:.2f} → ₹{current*10:.2f}. Emergency repricing triggered."
        )
    elif anomaly_type == "extreme_discount":
        return round(current * 0.05, 2), (
            f"[ANOMALY] Extreme 95% clearance for {product['brand']}: "
            f"₹{current:.2f} → ₹{current*0.05:.2f}. Blind clearance engine misfired."
        )
    else:  # overflow_error
        return round(current * -1, 2) if current > 0 else 0.01, (
            f"[ANOMALY] Overflow error: negative price computed for {product['brand']}. "
            f"Bot calculation produced invalid result."
        )


# ---------------------------------------------------------------------------
# Single request fire
# ---------------------------------------------------------------------------

async def fire_once(client: httpx.AsyncClient, products: list[dict]) -> None:
    product = random.choice(products)
    bot     = random.choice(BOTS)

    current_price  = product["actual_price"]
    proposed_price = product["proposed_price"]

    # 7% chance of anomaly injection
    if random.random() < 0.07:
        proposed_price, reasoning = maybe_inject_anomaly(product, bot)
    else:
        # For non-anomalies, we want to ensure the bot doesn't accidentally
        # trigger the 65% AI Hallucination cap, so we'll artificial cap
        # the discount at 60% max.
        max_allowed_discount_price = current_price * 0.40  # 60% off
        if proposed_price < max_allowed_discount_price:
            proposed_price = max_allowed_discount_price

        # Small random jitter on the proposed price (±5%) for realism
        jitter = random.uniform(-0.05, 0.05)
        proposed_price = round(proposed_price * (1 + jitter), 2)
        proposed_price = max(0.01, proposed_price)

        delta_pct = ((proposed_price - current_price) / current_price) * 100
        reasoning = build_reasoning(product, bot, delta_pct)

    delta_pct = ((proposed_price - current_price) / current_price) * 100

    payload = {
        "product_id":     product["sku"],
        "current_price":  current_price,
        "proposed_price": proposed_price,
        "reasoning":      reasoning,
        "bot_id":         bot["id"],
        "metadata": {
            "brand":      product["brand"],
            "name":       product["name"],
            "rating":     product["rating"],
            "no_ratings": product["no_ratings"],
            "sentiment":  product["sentiment"],
        },
    }

    try:
        resp = await client.post(SUBMIT_ENDPOINT, json=payload, timeout=30)
        data = resp.json()
        status = data.get("status", "?")
        req_id = data.get("request_id", "?")[:8]
        logger.info(
            f"[{bot['id']}] {product['sku'][:30]:<30} "
            f"₹{current_price:>8.2f} → ₹{proposed_price:>8.2f} "
            f"({delta_pct:+.1f}%) → {status.upper():<14} id={req_id}"
        )
    except Exception as e:
        logger.warning(f"Request failed: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(interval: float) -> None:
    if not DATA_FILE.exists():
        logger.error(f"Dataset not found at {DATA_FILE}. Please place products.csv in data/")
        sys.exit(1)

    products = load_products(DATA_FILE)
    if not products:
        logger.error("No valid products loaded from CSV. Check price columns.")
        sys.exit(1)

    logger.info(f"Starting pricing bot — {len(products)} products, {interval}s interval")
    logger.info(f"Submitting to: {SUBMIT_ENDPOINT}")

    async with httpx.AsyncClient() as client:
        # Wait for API to be ready
        for attempt in range(10):
            try:
                r = await client.get(f"{API_URL}/health", timeout=5)
                if r.status_code == 200:
                    logger.info("API is ready. Starting fire loop.")
                    break
            except Exception:
                pass
            logger.info(f"Waiting for API... attempt {attempt + 1}/10")
            await asyncio.sleep(3)

        while True:
            await fire_once(client, products)
            await asyncio.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Governance Pricing Bot")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between requests (default: 3)")
    args = parser.parse_args()
    asyncio.run(run(args.interval))
