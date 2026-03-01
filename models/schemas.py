"""
models/schemas.py
-----------------
All Pydantic v2 data models and the LangGraph TypedDict state for the
Supervising AI Governance Agent.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DecisionStatus(str, Enum):
    PENDING       = "pending"
    APPROVED      = "approved"
    REJECTED      = "rejected"
    MANUAL_REVIEW = "manual_review"


class HITLDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ   = "groq"


# ---------------------------------------------------------------------------
# Request / Input Models
# ---------------------------------------------------------------------------

class PricingUpdateRequest(BaseModel):
    """
    Payload posted by a pricing bot to the governance ingestion endpoint.
    """
    product_id:     str   = Field(..., description="Unique product identifier (e.g. SKU-001)")
    current_price:  float = Field(..., gt=0, description="Current live price (must be > 0)")
    proposed_price: float = Field(..., gt=0, description="Proposed new price (must be > 0)")
    bot_id:         str   = Field(..., description="Identifier of the submitting bot")
    reasoning:      str   = Field(..., min_length=5, description="Bot's natural-language justification")
    actor_type:     str   = Field("ai", description="Actor type: 'ai' or 'human'")
    metadata:       dict[str, Any] = Field(default_factory=dict, description="Optional extra context")

    @model_validator(mode="after")
    def prices_must_differ(self) -> "PricingUpdateRequest":
        if self.current_price == self.proposed_price:
            raise ValueError("proposed_price must differ from current_price")
        return self

    @property
    def price_delta_pct(self) -> float:
        """Signed percentage change: negative = discount, positive = increase."""
        return round(
            ((self.proposed_price - self.current_price) / self.current_price) * 100, 4
        )


class HITLReviewAction(BaseModel):
    """
    Payload posted by a human reviewer on the operations dashboard.
    """
    decision:     HITLDecision = Field(..., description="'approved' or 'rejected'")
    reviewer_id:  str          = Field(..., description="Human reviewer identifier / SSO ID")
    notes:        Optional[str] = Field(None, description="Optional reviewer notes")


# ---------------------------------------------------------------------------
# Response / Read Models
# ---------------------------------------------------------------------------

class SubmitResponse(BaseModel):
    """Returned immediately after a bot submits a pricing update."""
    request_id: uuid.UUID
    status:     DecisionStatus
    message:    str


class GovernanceLogEntry(BaseModel):
    """
    Full shape of a governance_logs row — used for API read responses
    and the Streamlit dashboard.
    """
    request_id:          uuid.UUID
    product_id:          str
    bot_id:              str
    original_price:      float
    proposed_price:      float
    price_delta_pct:     float
    status:              DecisionStatus
    layer1_result:       Optional[str]  = None
    ai_reasoning:        Optional[str]  = None
    ai_confidence_score: Optional[float] = None
    human_reviewer_id:   Optional[str]  = None
    human_decision:      Optional[str]  = None
    bot_reasoning:       Optional[str]  = None
    created_at:          datetime
    updated_at:          datetime

    model_config = {"from_attributes": True}


class PaginatedLogs(BaseModel):
    total:   int
    page:    int
    size:    int
    items:   list[GovernanceLogEntry]


# ---------------------------------------------------------------------------
# Internal Decision Model
# ---------------------------------------------------------------------------

class GovernanceDecision(BaseModel):
    """
    Internal model capturing a fully-resolved governance decision
    before it is persisted to the database.
    """
    request_id:          uuid.UUID         = Field(default_factory=uuid.uuid4)
    product_id:          str
    bot_id:              str
    original_price:      float
    proposed_price:      float
    price_delta_pct:     float
    status:              DecisionStatus    = DecisionStatus.PENDING
    layer1_result:       Optional[str]     = None
    ai_reasoning:        Optional[str]     = None
    ai_confidence_score: Optional[float]   = None
    human_reviewer_id:   Optional[str]     = None
    human_decision:      Optional[str]     = None
    bot_reasoning:       str               = ""


# ---------------------------------------------------------------------------
# LangGraph Shared State (TypedDict)
# ---------------------------------------------------------------------------

class GovernanceState(TypedDict, total=False):
    """
    Shared state dict threaded through every node in the LangGraph
    state machine. Using TypedDict (not Pydantic) as LangGraph requires it.

    Fields are populated progressively as the graph executes:
      check_static_rules  -> layer1_result, status
      sentiment_circuit_breaker -> sentiment_lock (optional)
      evaluate_context_llm -> ai_reasoning, ai_confidence_score
      route_decision -> final status persisted to DB
    """
    # --- Inputs (set at graph entry) ---
    request_id:          str          # UUID as string
    product_id:          str
    bot_id:              str
    original_price:      float
    proposed_price:      float
    price_delta_pct:     float
    actor_type:          str
    bot_reasoning:       str
    metadata:            dict[str, Any]

    # --- Layer 1 outputs ---
    layer1_passed:       bool
    layer1_result:       str          # human-readable rule verdict

    # --- Sentiment circuit breaker ---
    sentiment_lock:      bool         # True = essential-goods lock triggered
    sentiment_event:     str          # description of triggering news event

    # --- LLM outputs ---
    ai_reasoning:        str
    ai_confidence_score: float        # 0.0 – 1.0

    # --- Final decision ---
    status:              str          # DecisionStatus value
    error:               str          # set on unexpected exceptions
