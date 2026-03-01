"""
services/llm_factory.py
-----------------------
LLM provider factory supporting:
  - mock    → No API key, no rate limits. Rule-based heuristics. Perfect for demos.
  - gemini  → Google Gemini 2.0 Flash Lite (free-tier quota)
  - groq    → Groq Llama 3 8B (free-tier, very fast)
  - mistral → Mistral Codestral via Mistral AI (codestral.mistral.ai)
"""
from __future__ import annotations

import json
import logging
import random
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock LLM  — rule-based, zero API key, zero rate limits
# ---------------------------------------------------------------------------

class MockLLMMessage:
    """Mimics a LangChain AIMessage so governance_graph.py can call .content."""
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """
    Drop-in replacement for a real chat model.
    Uses heuristics on the prompt text to produce realistic governance verdicts.
    Supports .ainvoke() so it works identically to Gemini/Groq in async code.
    """

    async def ainvoke(self, messages) -> MockLLMMessage:
        # Extract the user prompt (last message)
        prompt = messages[-1].content if messages else ""
        return MockLLMMessage(content=self._evaluate(prompt))

    def _evaluate(self, prompt: str) -> str:
        # Parse delta from prompt
        delta_pct = 0.0
        try:
            for line in prompt.splitlines():
                if "Delta" in line and "%" in line:
                    val = line.split(":")[-1].strip().replace("%", "").replace("+", "")
                    delta_pct = float(val)
                    break
        except (ValueError, IndexError):
            pass

        prompt_lower = prompt.lower()

        # Heuristic confidence scoring
        confidence = 0.80  # neutral baseline

        # Large discounts are borderline
        if delta_pct < -15:
            confidence -= 0.25
        elif delta_pct < -10:
            confidence -= 0.15
        elif delta_pct < -5:
            confidence -= 0.05

        # Large increases are risky
        if delta_pct > 30:
            confidence -= 0.30
        elif delta_pct > 15:
            confidence -= 0.15

        # Trust signals that raise confidence
        if any(k in prompt_lower for k in ["seasonal", "competitor", "demand forecast", "clearance"]):
            confidence += 0.08
        if any(k in prompt_lower for k in ["bundle", "promotion", "cost-of-goods"]):
            confidence += 0.05

        # Suspicion signals that lower confidence
        if any(k in prompt_lower for k in ["error", "overflow", "emergency", "misfir"]):
            confidence -= 0.35
        if any(k in prompt_lower for k in ["anomaly", "extreme", "blind"]):
            confidence -= 0.25
        if "penny" in prompt_lower or "0.01" in prompt_lower:
            confidence -= 0.40

        # Clamp + small random jitter for realism
        confidence = max(0.05, min(0.99, confidence + random.uniform(-0.04, 0.04)))

        # Map confidence to decision
        if confidence >= settings.AI_AUTO_APPROVE_THRESHOLD:
            decision = "APPROVE"
            reasoning = (
                f"Price change of {delta_pct:+.2f}% is within normal competitive range. "
                "Bot reasoning is coherent and aligns with observed market patterns. "
                f"Governance risk is low (confidence {confidence:.2f}). Approved."
            )
        elif confidence < settings.AI_HIGH_RISK_THRESHOLD:
            decision = "MANUAL_REVIEW"
            reasoning = (
                f"Price change of {delta_pct:+.2f}% carries elevated risk. "
                "Bot reasoning contains ambiguous or suspicious signals. "
                f"Confidence score {confidence:.2f} is below the auto-approve threshold. "
                "Escalating to human reviewer for final decision."
            )
        else:
            decision = "APPROVE"
            reasoning = (
                f"Price change of {delta_pct:+.2f}% is acceptable but warrants monitoring. "
                f"Confidence {confidence:.2f} — approved with low-risk flag."
            )

        return json.dumps({
            "confidence_score": round(confidence, 4),
            "decision": decision,
            "reasoning": reasoning,
        })


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_llm():
    """Returns a cached model instance (real or mock)."""
    provider = settings.LLM_PROVIDER.lower()

    if provider == "mock":
        logger.info("Using LLM: MockLLM (no API key — rule-based heuristics)")
        return MockLLM()

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info(f"Using LLM: Google {settings.GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=512,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        logger.info(f"Using LLM: Groq {settings.GROQ_MODEL}")
        return ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=512,
        )

    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        logger.info(f"Using LLM: Mistral {settings.MISTRAL_MODEL} (Codestral endpoint)")
        return ChatMistralAI(
            model=settings.MISTRAL_MODEL,
            api_key=settings.MISTRAL_API_KEY,
            endpoint="https://codestral.mistral.ai/v1",
            temperature=0.1,
            max_tokens=512,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Choose: mock, gemini, groq, mistral"
        )
