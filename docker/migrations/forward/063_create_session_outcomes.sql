-- Migration: 063_create_session_outcomes
-- Description: Create session_outcomes projection table (OMN-8540)
-- Created: 2026-04-12
--
-- Purpose: Projection table for onex.evt.omniclaude.session-outcome.v1 events.
-- Consumed by node_projection_session_outcome (omnimarket). Closes the golden
-- chain evaluation chain: the tail table was referenced but never migrated into
-- the omnibase_infra Postgres schema, causing 0/5 golden chain sweep results.
--
-- Schema derived from omnidash migrations 0021_session_outcomes.sql and
-- 0058_session_outcomes_correlation_id.sql, and the handler in
-- omnimarket/src/omnimarket/nodes/node_projection_session_outcome/.
--
-- Idempotency: All statements use IF NOT EXISTS; safe to re-apply.
-- Rollback: See rollback/rollback_063_session_outcomes.sql

-- ============================================================================
-- TABLE: session_outcomes
-- ============================================================================
-- Projects session-outcome events. UPSERT key is session_id (latest-state-wins).
-- correlation_id added for golden chain validation (OMN-8521).

CREATE TABLE IF NOT EXISTS session_outcomes (
    session_id      TEXT        NOT NULL,
    outcome         TEXT        NOT NULL,
    emitted_at      TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    correlation_id  UUID,

    CONSTRAINT pk_session_outcomes PRIMARY KEY (session_id)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_session_outcomes_outcome
    ON session_outcomes (outcome);

CREATE INDEX IF NOT EXISTS idx_session_outcomes_emitted_at
    ON session_outcomes (emitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_outcomes_correlation_id
    ON session_outcomes (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE session_outcomes IS
    'Session outcome projection from onex.evt.omniclaude.session-outcome.v1. '
    'UPSERT key: session_id. Golden chain evaluation tail table (OMN-8540).';

COMMENT ON COLUMN session_outcomes.session_id IS
    'Unique session identifier. Primary key; used as UPSERT conflict target.';

COMMENT ON COLUMN session_outcomes.outcome IS
    'Session outcome: success, failed, abandoned, unknown.';

COMMENT ON COLUMN session_outcomes.emitted_at IS
    'When the session-outcome event was emitted by omniclaude.';

COMMENT ON COLUMN session_outcomes.ingested_at IS
    'When this row was last ingested/updated by the projection consumer.';

COMMENT ON COLUMN session_outcomes.correlation_id IS
    'Distributed tracing correlation ID (UUID). Added for golden chain lookup (OMN-8521).';
