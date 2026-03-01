"""
services/governance_graph.py
-----------------------------
LangGraph state machine — the heart of the Governance Engine.

Graph topology:
                        ┌─────────────────────┐
                        │   check_static_rules │  (Layer 1 – Redis)
                        └──────────┬──────────┘
                   FAIL ◄──────────┤──────────► PASS
                    │              │                │
                    ▼              │                ▼
            [auto_reject]          │    [sentiment_circuit_breaker]
                                   │       (mock news events)
                                   │                │
                                   │      CRISIS ◄──┤──► CLEAR
                                   │         │      │         │
                                   │         ▼      │         ▼
                                   │   [auto_reject]│  [evaluate_context_llm]
                                   │               (Gemini / Groq)
                                   │                      │
                                   │              ┌───────┴────────┐
                                   │          HIGH│              LOW│confidence
                                   │              ▼                ▼
                                   │    [manual_review]     [route_decision]
                                   │    (HITL pause)        approve/reject
                                   └─────────────────────────────────────────►
                                                                     [persist_decision]
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from config import settings
from models.schemas import DecisionStatus, GovernanceState
from services.llm_factory import build_llm
from services.news_sentinel import check_sentiment
from services.redis_rules import evaluate_static_rules

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for Layer 2 LLM evaluation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict AI Pricing Governance Auditor for an e-commerce marketplace.
Your role: evaluate whether a proposed price change is contextually safe and commercially reasonable.

You must respond ONLY with a valid JSON object in this exact format:
{
  "confidence_score": <float 0.0–1.0>,
  "decision": "<APPROVE|REJECT|MANUAL_REVIEW>",
  "reasoning": "<one concise paragraph, max 120 words>"
}

Scoring guide:
  1.0 – 0.85 : Clearly safe and reasonable → APPROVE
  0.85 – 0.60 : Moderate uncertainty        → APPROVE (with note) or MANUAL_REVIEW
  0.60 – 0.00 : Risky, unusual, or suspect  → MANUAL_REVIEW or REJECT

Consider: competitor context, seasonal patterns, potential predatory / panic pricing,
and whether the bot's reasoning is coherent and trustworthy."""


def _make_user_prompt(state: GovernanceState) -> str:
    return f"""
PRICING UPDATE REQUEST
======================
Product ID    : {state['product_id']}
Bot ID        : {state['bot_id']}
Current Price : ${state['original_price']:.2f}
Proposed Price: ${state['proposed_price']:.2f}
Delta         : {state['price_delta_pct']:+.2f}%
Bot Reasoning : {state['bot_reasoning']}
Extra Context : {json.dumps(state.get('metadata', {}), indent=2)}

Layer 1 Result (static rules): {state.get('layer1_result', 'N/A')}

Evaluate this pricing change and return ONLY the JSON object as specified.
""".strip()


# ---------------------------------------------------------------------------
# Node: check_static_rules  (Layer 1)
# ---------------------------------------------------------------------------
async def check_static_rules(state: GovernanceState, redis: aioredis.Redis) -> dict:
    """Evaluates hardcoded static rules via Redis. Fast — no LLM."""
    result = await evaluate_static_rules(
        redis=redis,
        product_id=state["product_id"],
        current_price=state["original_price"],
        proposed_price=state["proposed_price"],
        actor_type=state.get("actor_type", "ai"),
        metadata=state.get("metadata", {}),
    )

    update: dict[str, Any] = {
        "layer1_passed": result.passed,
        "layer1_result": result.message,
        "price_delta_pct": result.delta_pct,
    }

    if not result.passed:
        update["status"] = DecisionStatus.REJECTED.value
        update["ai_reasoning"] = f"Auto-rejected by Layer 1 rule: {result.rule_name}"
        update["ai_confidence_score"] = 0.0

    return update


# ---------------------------------------------------------------------------
# Node: sentiment_circuit_breaker
# ---------------------------------------------------------------------------
async def sentiment_circuit_breaker(state: GovernanceState, redis: aioredis.Redis) -> dict:
    """
    Checks external mock 'news events' for market crises.
    If a crisis is detected → locks essential goods and sets sentiment_lock=True.
    """
    sentiment = await check_sentiment(redis, metadata=state.get("metadata", {}))

    update: dict[str, Any] = {
        "sentiment_lock": sentiment.crisis_detected,
        "sentiment_event": sentiment.headline if sentiment.crisis_detected else "",
    }

    if sentiment.crisis_detected:
        # Check if this product is in an essential category
        product_category = state.get("metadata", {}).get("category", "").lower()
        essential_cats = settings.ESSENTIAL_GOODS_CATEGORIES

        is_essential = any(c in product_category for c in essential_cats)

        if is_essential or sentiment.category_lock == "all":
            update["status"] = DecisionStatus.REJECTED.value
            update["ai_reasoning"] = (
                f"CRISIS LOCK: {sentiment.headline}. "
                f"Essential goods repricing blocked during market emergency."
            )
            update["ai_confidence_score"] = 0.0
            logger.warning(
                f"[SentimentBreaker] Blocked {state['product_id']} — crisis lock active."
            )

    return update


# ---------------------------------------------------------------------------
# Node: evaluate_context_llm  (Layer 2)
# ---------------------------------------------------------------------------
async def evaluate_context_llm(state: GovernanceState) -> dict:
    """
    Calls the lightweight LLM (Gemini Flash Lite or Groq Llama3-8B)
    to assess contextual risk and return a confidence score + decision.
    """
    llm = build_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_make_user_prompt(state)),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if the model wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        confidence  = float(parsed.get("confidence_score", 0.5))
        llm_decision = str(parsed.get("decision", "MANUAL_REVIEW")).upper()
        reasoning   = str(parsed.get("reasoning", "No reasoning provided."))

        logger.info(
            f"[LLM] product={state['product_id']} "
            f"decision={llm_decision} confidence={confidence:.3f}"
        )

        return {
            "ai_confidence_score": confidence,
            "ai_reasoning": reasoning,
            # Store raw LLM decision for route_decision to use
            "_llm_decision": llm_decision,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"[LLM] Failed to parse response: {e}. Defaulting to MANUAL_REVIEW.")
        return {
            "ai_confidence_score": 0.5,
            "ai_reasoning": f"LLM response parsing error: {e}. Escalated to human review.",
            "_llm_decision": "MANUAL_REVIEW",
        }


# ---------------------------------------------------------------------------
# Node: route_decision  (final router)
# ---------------------------------------------------------------------------
async def route_decision(state: GovernanceState) -> dict:
    """
    Maps confidence score + LLM decision to a final DecisionStatus.
    Thresholds are pulled from settings (configurable via .env).
    """
    confidence  = state.get("ai_confidence_score", 0.5)
    llm_decision = state.get("_llm_decision", "MANUAL_REVIEW")

    if llm_decision == "APPROVE" and confidence >= settings.AI_AUTO_APPROVE_THRESHOLD:
        final_status = DecisionStatus.APPROVED.value
    elif llm_decision == "REJECT" or confidence < settings.AI_HIGH_RISK_THRESHOLD:
        final_status = DecisionStatus.MANUAL_REVIEW.value   # HITL for LLM rejects
    elif confidence >= settings.AI_AUTO_APPROVE_THRESHOLD:
        final_status = DecisionStatus.APPROVED.value
    elif confidence >= settings.AI_HIGH_RISK_THRESHOLD:
        final_status = DecisionStatus.MANUAL_REVIEW.value
    else:
        final_status = DecisionStatus.MANUAL_REVIEW.value

    logger.info(
        f"[Router] product={state['product_id']} "
        f"confidence={confidence:.3f} → {final_status}"
    )
    return {"status": final_status}


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------
def _should_run_llm_after_layer1(state: GovernanceState) -> str:
    """After Layer 1: if rejected, skip to END; else run sentiment check."""
    if not state.get("layer1_passed", True):
        return "rejected"
    return "continue"


def _should_run_llm_after_sentiment(state: GovernanceState) -> str:
    """After sentiment check: if crisis blocked this product, skip to END."""
    if state.get("status") == DecisionStatus.REJECTED.value:
        return "crisis_blocked"
    return "continue"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_governance_graph(redis: aioredis.Redis) -> Any:
    """
    Constructs and compiles the LangGraph StateGraph.
    `redis` is captured in node closures so each node can access it.

    Returns a compiled LangGraph app (supports .ainvoke()).
    """
    # Wrap async node functions to inject the shared Redis client
    async def _check_static_rules(state: GovernanceState) -> dict:
        return await check_static_rules(state, redis)

    async def _sentiment_circuit_breaker(state: GovernanceState) -> dict:
        return await sentiment_circuit_breaker(state, redis)

    # Build graph
    graph = StateGraph(GovernanceState)

    # Register nodes
    graph.add_node("check_static_rules",       _check_static_rules)
    graph.add_node("sentiment_circuit_breaker", _sentiment_circuit_breaker)
    graph.add_node("evaluate_context_llm",      evaluate_context_llm)
    graph.add_node("route_decision",            route_decision)

    # Entry point
    graph.set_entry_point("check_static_rules")

    # Conditional edge: Layer 1 result
    graph.add_conditional_edges(
        "check_static_rules",
        _should_run_layer1_routing,
        {
            "rejected": END,           # Auto-reject — no LLM needed
            "continue": "sentiment_circuit_breaker",
        },
    )

    # Conditional edge: Sentiment result
    graph.add_conditional_edges(
        "sentiment_circuit_breaker",
        _should_run_sentiment_routing,
        {
            "crisis_blocked": END,     # Essential goods blocked — no LLM needed
            "continue": "evaluate_context_llm",
        },
    )

    # Linear: LLM → Router → END
    graph.add_edge("evaluate_context_llm", "route_decision")
    graph.add_edge("route_decision", END)

    return graph.compile()


def _should_run_layer1_routing(state: GovernanceState) -> str:
    return "rejected" if not state.get("layer1_passed", True) else "continue"


def _should_run_sentiment_routing(state: GovernanceState) -> str:
    return (
        "crisis_blocked"
        if state.get("status") == DecisionStatus.REJECTED.value
        else "continue"
    )


# ---------------------------------------------------------------------------
# Convenience: run the full graph for a pricing update
# ---------------------------------------------------------------------------
async def run_governance_check(
    request_id: str,
    product_id: str,
    bot_id: str,
    actor_type: str,
    original_price: float,
    proposed_price: float,
    price_delta_pct: float,
    bot_reasoning: str,
    metadata: dict,
    redis: aioredis.Redis,
) -> GovernanceState:
    """
    Entry point called by the FastAPI route.
    Builds and invokes the compiled LangGraph, returns the final state.
    """
    graph = build_governance_graph(redis)

    initial_state: GovernanceState = {
        "request_id":      request_id,
        "product_id":      product_id,
        "bot_id":          bot_id,
        "actor_type":      actor_type,
        "original_price":  original_price,
        "proposed_price":  proposed_price,
        "price_delta_pct": price_delta_pct,
        "bot_reasoning":   bot_reasoning,
        "metadata":        metadata,
    }

    final_state: GovernanceState = await graph.ainvoke(initial_state)
    return final_state
