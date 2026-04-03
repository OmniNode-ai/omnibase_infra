-- Migration: 062_context_enrichment_injection_recorded_tables
-- Description: Create tables for context enrichment and injection recorded events (OMN-6158)
-- Created: 2026-04-03
--
-- Purpose: Closes two Kafka consumer pipeline gaps where events fire into Kafka
-- but have no consumers:
--   1. onex.evt.omniclaude.context-enrichment.v1 — per-channel enrichment metrics
--   2. onex.evt.omniclaude.injection-recorded.v1 — injection tracking events
--
-- Idempotency (writer-side ON CONFLICT handling):
--   - context_enrichment_events: UNIQUE on (session_id, channel) enables idempotent inserts
--   - injection_recorded_events: UNIQUE on (session_id, emitted_at) enables idempotent inserts
--
-- Rollback: See rollback/rollback_062_context_enrichment_injection_recorded_tables.sql

-- ============================================================================
-- CONTEXT_ENRICHMENT_EVENTS TABLE
-- ============================================================================
-- Records per-channel enrichment metrics (summarization, code_analysis, similarity).
-- One row per (session_id, channel) pair.

CREATE TABLE IF NOT EXISTS context_enrichment_events (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    correlation_id TEXT,

    -- Timestamp
    emitted_at TEXT NOT NULL,

    -- Channel identification
    channel TEXT NOT NULL,
    model_name TEXT DEFAULT '',

    -- Outcome
    cache_hit BOOLEAN DEFAULT FALSE,
    outcome TEXT NOT NULL,

    -- Metrics
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    tokens_before INTEGER DEFAULT 0,
    tokens_after INTEGER DEFAULT 0,
    net_tokens_saved INTEGER DEFAULT 0,

    -- Scores
    similarity_score DOUBLE PRECISION,
    quality_score DOUBLE PRECISION,

    -- Context
    repo TEXT,
    agent_name TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per session per channel
    CONSTRAINT unique_session_channel UNIQUE (session_id, channel),

    -- Valid outcome values
    CONSTRAINT valid_enrichment_outcome CHECK (
        outcome IN ('hit', 'miss', 'error', 'inflated')
    ),

    -- Non-negative metrics
    CONSTRAINT non_negative_enrichment_latency CHECK (latency_ms >= 0.0),
    CONSTRAINT non_negative_enrichment_tokens_before CHECK (tokens_before >= 0),
    CONSTRAINT non_negative_enrichment_tokens_after CHECK (tokens_after >= 0)
);

-- ============================================================================
-- INJECTION_RECORDED_EVENTS TABLE
-- ============================================================================
-- Records injection tracking events emitted by INJECT-004 (OMN-1673).
-- One row per (session_id, emitted_at) pair.

CREATE TABLE IF NOT EXISTS injection_recorded_events (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    correlation_id TEXT,

    -- Timestamp
    emitted_at TEXT NOT NULL,

    -- Injection metrics
    patterns_injected INTEGER DEFAULT 0,
    total_injected_tokens INTEGER DEFAULT 0,
    injection_latency_ms DOUBLE PRECISION DEFAULT 0.0,

    -- Context
    agent_name TEXT,
    repo TEXT,
    cache_hit BOOLEAN DEFAULT FALSE,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per session per emission timestamp
    CONSTRAINT unique_session_emitted UNIQUE (session_id, emitted_at),

    -- Non-negative metrics
    CONSTRAINT non_negative_injection_patterns CHECK (patterns_injected >= 0),
    CONSTRAINT non_negative_injection_tokens CHECK (total_injected_tokens >= 0),
    CONSTRAINT non_negative_injection_latency CHECK (injection_latency_ms >= 0.0)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- context_enrichment_events indexes
CREATE INDEX IF NOT EXISTS idx_context_enrichment_session_id
    ON context_enrichment_events (session_id);

CREATE INDEX IF NOT EXISTS idx_context_enrichment_created_at
    ON context_enrichment_events (created_at);

CREATE INDEX IF NOT EXISTS idx_context_enrichment_channel
    ON context_enrichment_events (channel);

CREATE INDEX IF NOT EXISTS idx_context_enrichment_outcome
    ON context_enrichment_events (outcome);

CREATE INDEX IF NOT EXISTS idx_context_enrichment_repo
    ON context_enrichment_events (repo)
    WHERE repo IS NOT NULL;

-- injection_recorded_events indexes
CREATE INDEX IF NOT EXISTS idx_injection_recorded_session_id
    ON injection_recorded_events (session_id);

CREATE INDEX IF NOT EXISTS idx_injection_recorded_created_at
    ON injection_recorded_events (created_at);

CREATE INDEX IF NOT EXISTS idx_injection_recorded_repo
    ON injection_recorded_events (repo)
    WHERE repo IS NOT NULL;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE context_enrichment_events IS
    'Per-channel context enrichment metrics (OMN-6158). '
    'One row per (session_id, channel). Idempotent via ON CONFLICT DO NOTHING.';

COMMENT ON TABLE injection_recorded_events IS
    'Injection tracking events from INJECT-004 (OMN-6158). '
    'One row per (session_id, emitted_at). Idempotent via ON CONFLICT DO NOTHING.';
