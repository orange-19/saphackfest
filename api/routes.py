"""
api/routes.py
-------------
FastAPI routers:
  1. /api/v1/governance  — bot ingestion endpoints + log queries
  2. /api/v1/hitl        — human-in-the-loop review queue + action endpoints
"""
from __future__ import annotations

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from db.database import get_db
from models.schemas import (
    DecisionStatus,
    GovernanceLogEntry,
    HITLDecision,
    HITLReviewAction,
    PaginatedLogs,
    PricingUpdateRequest,
    SubmitResponse,
)
from services.governance_graph import run_governance_check

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router 1: Governance Ingestion
# ---------------------------------------------------------------------------
governance_router = APIRouter(prefix="/api/v1/governance", tags=["Governance"])


@governance_router.post(
    "/submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitResponse,
    summary="Submit a pricing update for governance review",
    description=(
        "A pricing bot POSTs a proposed price change here. "
        "The request is immediately persisted as 'pending', then the "
        "governance engine (LangGraph — Phase 2) is invoked asynchronously."
    ),
)
async def submit_pricing_update(
    payload: PricingUpdateRequest,
    request: Request,                     # to access app.state.redis
    db: AsyncSession = Depends(get_db),
) -> SubmitResponse:
    """
    Full Phase 2 implementation:
      1. Persist initial 'pending' row
      2. Run LangGraph governance pipeline (Layer 1 → Sentiment → LLM → Router)
      3. Upsert final decision back to DB
    """
    request_id = uuid.uuid4()
    delta_pct  = payload.price_delta_pct

    # 1. Insert placeholder row (status=pending)
    insert_sql = text("""
        INSERT INTO governance_logs
            (request_id, product_id, bot_id, bot_reasoning,
             original_price, proposed_price, price_delta_pct, status)
        VALUES
            (:request_id, :product_id, :bot_id, :bot_reasoning,
             :original_price, :proposed_price, :price_delta_pct, 'pending')
    """)
    await db.execute(insert_sql, {
        "request_id":      str(request_id),
        "product_id":      payload.product_id,
        "bot_id":          payload.bot_id,
        "bot_reasoning":   payload.reasoning,
        "original_price":  payload.current_price,
        "proposed_price":  payload.proposed_price,
        "price_delta_pct": delta_pct,
    })
    await db.commit()

    # 2. Run governance graph (async — returns final GovernanceState)
    redis = getattr(request.app.state, "redis", None)
    try:
        final_state = await run_governance_check(
            request_id=str(request_id),
            product_id=payload.product_id,
            bot_id=payload.bot_id,
            actor_type=payload.actor_type,
            original_price=payload.current_price,
            proposed_price=payload.proposed_price,
            price_delta_pct=delta_pct,
            bot_reasoning=payload.reasoning,
            metadata=payload.metadata,
            redis=redis,
        )
    except Exception as exc:
        logger.error(f"[SUBMIT] Governance graph error for {request_id}: {exc}")
        final_state = {
            "status": DecisionStatus.MANUAL_REVIEW.value,
            "ai_reasoning": f"Graph execution error: {exc}",
            "ai_confidence_score": 0.0,
            "layer1_result": "ERROR",
        }

    # 3. Upsert final decision back to the DB row
    final_status = final_state.get("status", DecisionStatus.MANUAL_REVIEW.value)
    update_sql = text("""
        UPDATE governance_logs
           SET status              = :status,
               layer1_result       = :layer1_result,
               ai_reasoning        = :ai_reasoning,
               ai_confidence_score = :confidence
         WHERE request_id = :rid
    """)
    await db.execute(update_sql, {
        "status":        final_status,
        "layer1_result": final_state.get("layer1_result", ""),
        "ai_reasoning":  final_state.get("ai_reasoning", ""),
        "confidence":    final_state.get("ai_confidence_score"),
        "rid":           str(request_id),
    })
    await db.commit()

    logger.info(
        f"[SUBMIT] request_id={request_id} product={payload.product_id} "
        f"delta={delta_pct:.2f}% → {final_status}"
    )

    status_map = {
        DecisionStatus.APPROVED.value:      "Pricing change approved and logged.",
        DecisionStatus.REJECTED.value:      "Pricing change rejected by governance engine.",
        DecisionStatus.MANUAL_REVIEW.value: "Escalated to human review. Awaiting HITL decision.",
    }
    return SubmitResponse(
        request_id=request_id,
        status=DecisionStatus(final_status),
        message=status_map.get(final_status, "Governance check complete."),
    )

@governance_router.delete("/reset")
async def reset_db(db: AsyncSession = Depends(get_db)):
    await db.execute(text("TRUNCATE TABLE governance_logs"))
    await db.commit()
    return {"status": "cleared"}


@governance_router.get(
    "/logs",
    response_model=PaginatedLogs,
    summary="List governance log entries",
)
async def list_logs(
    status_filter: Optional[DecisionStatus] = Query(None, alias="status"),
    product_id:    Optional[str]            = Query(None),
    page:          int                      = Query(1, ge=1),
    size:          int                      = Query(20, ge=1, le=100),
    db:            AsyncSession             = Depends(get_db),
) -> PaginatedLogs:
    where_clauses = []
    params: dict = {}

    if status_filter:
        where_clauses.append("status = :status")
        params["status"] = status_filter.value
    if product_id:
        where_clauses.append("product_id = :product_id")
        params["product_id"] = product_id

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    offset = (page - 1) * size

    count_sql = text(f"SELECT COUNT(*) FROM governance_logs {where_sql}")
    total_result = await db.execute(count_sql, params)
    total = total_result.scalar_one()

    rows_sql = text(f"""
        SELECT * FROM governance_logs
        {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)
    params.update({"limit": size, "offset": offset})
    rows_result = await db.execute(rows_sql, params)
    rows = rows_result.mappings().all()

    return PaginatedLogs(
        total=total,
        page=page,
        size=size,
        items=[GovernanceLogEntry.model_validate(dict(r)) for r in rows],
    )


@governance_router.get(
    "/logs/{request_id}",
    response_model=GovernanceLogEntry,
    summary="Get a single governance log entry by request_id",
)
async def get_log(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> GovernanceLogEntry:
    sql = text("SELECT * FROM governance_logs WHERE request_id = :rid")
    result = await db.execute(sql, {"rid": str(request_id)})
    row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No governance log found for request_id={request_id}",
        )
    return GovernanceLogEntry.model_validate(dict(row))


# ---------------------------------------------------------------------------
# Router 2: Human-in-the-Loop (HITL)
# ---------------------------------------------------------------------------
hitl_router = APIRouter(prefix="/api/v1/hitl", tags=["Human-in-the-Loop"])


@hitl_router.get(
    "/queue",
    response_model=list[GovernanceLogEntry],
    summary="Fetch all requests awaiting human review",
    description="Returns rows with status='manual_review', oldest-first. Powers the Streamlit dashboard.",
)
async def get_hitl_queue(
    db: AsyncSession = Depends(get_db),
) -> list[GovernanceLogEntry]:
    sql = text("""
        SELECT * FROM governance_logs
        WHERE status = 'manual_review'
        ORDER BY created_at ASC
    """)
    result = await db.execute(sql)
    rows = result.mappings().all()
    return [GovernanceLogEntry.model_validate(dict(r)) for r in rows]


@hitl_router.post(
    "/review/{request_id}",
    response_model=GovernanceLogEntry,
    summary="Submit a human review decision (approve / reject)",
    description=(
        "Called by the Streamlit dashboard when a human reviewer clicks "
        "Approve or Reject. Updates the row status and records reviewer ID."
    ),
)
async def submit_review(
    request_id: uuid.UUID,
    action: HITLReviewAction,
    db: AsyncSession = Depends(get_db),
) -> GovernanceLogEntry:
    # Verify the row exists and is in manual_review state
    fetch_sql = text("SELECT * FROM governance_logs WHERE request_id = :rid")
    result = await db.execute(fetch_sql, {"rid": str(request_id)})
    row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No governance log found for request_id={request_id}",
        )

    if row["status"] != DecisionStatus.MANUAL_REVIEW.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Request is not in 'manual_review' state (current: {row['status']})",
        )

    new_status = (
        DecisionStatus.APPROVED.value
        if action.decision == HITLDecision.APPROVED
        else DecisionStatus.REJECTED.value
    )

    update_sql = text("""
        UPDATE governance_logs
           SET status            = :status,
               human_reviewer_id = :reviewer_id,
               human_decision    = :decision
         WHERE request_id = :rid
    """)
    await db.execute(update_sql, {
        "status":      new_status,
        "reviewer_id": action.reviewer_id,
        "decision":    action.decision.value,
        "rid":         str(request_id),
    })
    await db.commit()

    logger.info(
        f"[HITL] request_id={request_id} → {new_status} by {action.reviewer_id}"
    )

    # Re-fetch and return the updated row
    updated = await db.execute(fetch_sql, {"rid": str(request_id)})
    updated_row = updated.mappings().first()
    return GovernanceLogEntry.model_validate(dict(updated_row))


@hitl_router.get(
    "/stats",
    summary="Quick stats for the ops dashboard header",
)
async def hitl_stats(db: AsyncSession = Depends(get_db)) -> dict:
    sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'manual_review') AS pending_review,
            COUNT(*) FILTER (WHERE status = 'approved')      AS total_approved,
            COUNT(*) FILTER (WHERE status = 'rejected')      AS total_rejected,
            COUNT(*)                                          AS total_requests
        FROM governance_logs
    """)
    result = await db.execute(sql)
    row = result.mappings().first()
    return dict(row)
