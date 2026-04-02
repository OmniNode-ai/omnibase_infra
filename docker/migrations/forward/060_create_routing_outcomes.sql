-- =============================================================================
-- MIGRATION: Create routing_outcomes and capability_scores tables
-- =============================================================================
-- Ticket: OMN-7277 (Intelligent Model Router — Postgres migration)
-- Epic: OMN-7264 (Intelligent Model Router MVP)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the routing_outcomes table for recording every model routing
--   decision and its outcome, and the capability_scores table for persisting
--   the reducer's rolling capability metrics per (model_key, task_type).
--
--   These tables back the node_routing_score_reducer and provide the live
--   metrics consumed by node_model_router_compute for scoring.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_060_create_routing_outcomes.sql
-- =============================================================================

-- ============================================================
-- Table: routing_outcomes
-- Records every routing decision + outcome for learning.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.routing_outcomes (
    id                      BIGSERIAL       PRIMARY KEY,
    correlation_id          UUID            NOT NULL,
    model_key                TEXT            NOT NULL,
    task_type               TEXT            NOT NULL,
    task_subtype            TEXT            NOT NULL DEFAULT '',
    task_description        TEXT            NOT NULL DEFAULT '',
    selected                BOOLEAN         NOT NULL DEFAULT TRUE,
    success                 BOOLEAN,
    actual_latency_ms       INT,
    actual_tokens_per_sec   DOUBLE PRECISION,
    actual_cost             DOUBLE PRECISION        DEFAULT 0.0,
    input_tokens            INT             DEFAULT 0,
    output_tokens           INT             DEFAULT 0,
    scoring_rationale       TEXT            NOT NULL DEFAULT '',
    composite_score         DOUBLE PRECISION,
    chain_hit               BOOLEAN         NOT NULL DEFAULT FALSE,
    chain_hit_model_key      TEXT            DEFAULT '',
    fallback_model_key       TEXT            DEFAULT '',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ
);

-- Primary lookup: model + task type + time range
CREATE INDEX IF NOT EXISTS idx_routing_outcomes_model_task_time
    ON routing_outcomes (model_key, task_type, created_at DESC);

-- Outcome analysis: success rate per model
CREATE INDEX IF NOT EXISTS idx_routing_outcomes_model_success
    ON routing_outcomes (model_key, success);

-- Correlation lookup
CREATE INDEX IF NOT EXISTS idx_routing_outcomes_correlation
    ON routing_outcomes (correlation_id);

-- Time-range scans for dashboard
CREATE INDEX IF NOT EXISTS idx_routing_outcomes_created_at
    ON routing_outcomes (created_at DESC);

-- Chain hit analysis
CREATE INDEX IF NOT EXISTS idx_routing_outcomes_chain_hit
    ON routing_outcomes (chain_hit, model_key)
    WHERE chain_hit = TRUE;

-- ============================================================
-- Table: capability_scores
-- Persisted reducer state: rolling metrics per (model, task_type).
-- ============================================================

CREATE TABLE IF NOT EXISTS public.capability_scores (
    id                      BIGSERIAL       PRIMARY KEY,
    model_key                TEXT            NOT NULL,
    task_type               TEXT            NOT NULL,
    success_count           INT             NOT NULL DEFAULT 0,
    failure_count           INT             NOT NULL DEFAULT 0,
    total_count             INT             NOT NULL DEFAULT 0,
    success_rate            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    avg_latency_ms          INT             NOT NULL DEFAULT 0,
    avg_tokens_per_sec      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_cost              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    graduated               BOOLEAN         NOT NULL DEFAULT FALSE,
    last_updated            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (model_key, task_type)
);

-- Lookup by model for router input
CREATE INDEX IF NOT EXISTS idx_capability_scores_model
    ON capability_scores (model_key);

-- Graduated models for fast filtering
CREATE INDEX IF NOT EXISTS idx_capability_scores_graduated
    ON capability_scores (graduated)
    WHERE graduated = TRUE;

-- ============================================================
-- COMMENTS
-- ============================================================

COMMENT ON TABLE routing_outcomes IS
    'Records every model routing decision and outcome for the intelligent '
    'model router learning flywheel (OMN-7277). Fed to node_routing_score_reducer.';

COMMENT ON TABLE capability_scores IS
    'Persisted reducer state: rolling capability metrics per (model_key, task_type). '
    'Read by node_model_router_compute for live scoring (OMN-7277).';

COMMENT ON COLUMN routing_outcomes.chain_hit IS
    'Whether a golden chain was found for this task, biasing toward local models.';

COMMENT ON COLUMN capability_scores.graduated IS
    'True when success_rate > 0.9 over 50+ verified attempts for this (model, task_type).';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '060',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
