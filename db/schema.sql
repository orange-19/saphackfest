-- =============================================================================
-- Supervising AI Governance Agent — Database Schema
-- PostgreSQL 14+
-- =============================================================================

-- Enable pgcrypto for gen_random_uuid() if not already available
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Enum: decision_status
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'decision_status') THEN
        CREATE TYPE decision_status AS ENUM (
            'pending',
            'approved',
            'rejected',
            'manual_review'
        );
    END IF;
END$$;

-- ---------------------------------------------------------------------------
-- Table: governance_logs
-- Immutable audit ledger. Rows are INSERT-once; status updates are the only
-- allowed mutations (enforced by application logic). The trigger below
-- keeps updated_at fresh automatically.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS governance_logs (
    -- Primary key
    request_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source context
    product_id          TEXT            NOT NULL,
    bot_id              TEXT            NOT NULL,
    bot_reasoning       TEXT,

    -- Price data
    original_price      NUMERIC(12, 4)  NOT NULL,
    proposed_price      NUMERIC(12, 4)  NOT NULL,
    price_delta_pct     NUMERIC(8, 4)   NOT NULL, -- precomputed: ((proposed-original)/original)*100

    -- Governance decision
    status              decision_status NOT NULL DEFAULT 'pending',

    -- Layer 1: static rule result
    layer1_result       TEXT,           -- e.g. "PASS" or "REJECTED: max_discount_exceeded"

    -- Layer 2: LLM agentic analysis
    ai_reasoning        TEXT,           -- LLM chain-of-thought summary
    ai_confidence_score NUMERIC(5, 4),  -- 0.0000 – 1.0000

    -- Human-in-the-loop
    human_reviewer_id   TEXT,           -- SSO / user ID of reviewer
    human_decision      TEXT,           -- 'approved' | 'rejected'

    -- Timestamps (immutable insert / mutable update)
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Dashboard query: fetch all rows where status = 'manual_review'
CREATE INDEX IF NOT EXISTS idx_governance_logs_status
    ON governance_logs (status)
    WHERE status = 'manual_review';

-- Product-level history (recent-first)
CREATE INDEX IF NOT EXISTS idx_governance_logs_product_id
    ON governance_logs (product_id, created_at DESC);

-- Bot-level analytics
CREATE INDEX IF NOT EXISTS idx_governance_logs_bot_id
    ON governance_logs (bot_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger: auto-update updated_at on every status mutation
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_update_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_governance_logs_updated_at ON governance_logs;

CREATE TRIGGER trg_governance_logs_updated_at
    BEFORE UPDATE ON governance_logs
    FOR EACH ROW
    EXECUTE FUNCTION fn_update_updated_at();

-- ---------------------------------------------------------------------------
-- View: manual_review_queue   (convenience view for Streamlit dashboard)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW manual_review_queue AS
    SELECT *
    FROM   governance_logs
    WHERE  status = 'manual_review'
    ORDER  BY created_at ASC; -- oldest-first so urgent items surface at top
