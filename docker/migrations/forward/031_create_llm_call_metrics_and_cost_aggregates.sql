-- Migration: 031_create_llm_call_metrics_and_cost_aggregates
-- Predecessor: 030_add_schema_fingerprint
-- Description: Create llm_call_metrics and llm_cost_aggregates tables (OMN-2236)
-- Created: 2026-02-15
--
-- Purpose: Stores per-LLM-call token usage, cost data, and rolling window
-- cost aggregations. Enables cost tracking, budget alerting, and model
-- usage analysis across sessions and patterns.
--
-- Tables:
--   1. llm_call_metrics    - Per-call token usage and cost data
--   2. llm_cost_aggregates - Rolling window cost aggregations
--
-- Idempotency: CREATE TABLE IF NOT EXISTS; CREATE INDEX IF NOT EXISTS
-- TTL Column: created_at (llm_call_metrics), updated_at (llm_cost_aggregates)
--
-- Forward-only: Old code ignores new tables. No backfill required.
-- This migration must complete before E1-T2 and E1-T4 can land.
--
-- Rollback: See rollback/rollback_031_llm_call_metrics_and_cost_aggregates.sql

-- ============================================================================
-- ENUM: usage_source_type
-- ============================================================================
-- Tracks provenance of token usage data: reported by API, estimated by us,
-- or missing entirely. Distinct from usage_is_estimated (boolean) because
-- MISSING means no data at all, not an estimate.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'usage_source_type') THEN
        CREATE TYPE usage_source_type AS ENUM ('API', 'ESTIMATED', 'MISSING');
    END IF;
END$$;

-- ============================================================================
-- ENUM: cost_aggregation_window
-- ============================================================================
-- Fixed set of rolling aggregation windows. Adding new windows requires a
-- migration to ALTER TYPE (intentional friction for schema evolution).

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'cost_aggregation_window') THEN
        CREATE TYPE cost_aggregation_window AS ENUM ('24h', '7d', '30d');
    END IF;
END$$;

-- ============================================================================
-- LLM_CALL_METRICS TABLE
-- ============================================================================
-- Stores per-LLM-call token usage and cost data. Each row represents one
-- LLM API call with token counts, estimated cost, and latency. Append-only,
-- write-heavy workload similar to agent_status_events.
--
-- Design decisions:
--   - No FK to session/run tables (async Kafka arrival, same pattern as
--     latency_breakdowns in migration 026)
--   - estimated_cost_usd is NULL for unknown models, NOT 0 (semantic difference)
--   - usage_raw capped at 64KB via CHECK constraint (prevents payload bloat)
--   - input_hash enables deduplication without full payload comparison

CREATE TABLE IF NOT EXISTS llm_call_metrics (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID,

    -- Session/run context (logical references, no FK - async event arrival)
    session_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255),

    -- Model identification
    model_id VARCHAR(255) NOT NULL,

    -- Token usage (nullable: not all providers report all fields)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Cost (NULL for unknown models, NOT 0)
    estimated_cost_usd NUMERIC(12, 6),

    -- Performance
    latency_ms NUMERIC(10, 2) NOT NULL,

    -- Usage provenance
    usage_source usage_source_type NOT NULL DEFAULT 'MISSING',
    usage_is_estimated BOOLEAN NOT NULL DEFAULT FALSE,

    -- Raw provider response (redacted, capped at 64KB)
    usage_raw JSONB,

    -- Provenance columns
    input_hash VARCHAR(64),
    code_version VARCHAR(64),
    contract_version VARCHAR(64),
    source VARCHAR(255),

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT non_negative_prompt_tokens CHECK (
        prompt_tokens IS NULL OR prompt_tokens >= 0
    ),
    CONSTRAINT non_negative_completion_tokens CHECK (
        completion_tokens IS NULL OR completion_tokens >= 0
    ),
    CONSTRAINT non_negative_total_tokens CHECK (
        total_tokens IS NULL OR total_tokens >= 0
    ),
    CONSTRAINT non_negative_estimated_cost_usd CHECK (
        estimated_cost_usd IS NULL OR estimated_cost_usd >= 0
    ),
    CONSTRAINT non_negative_latency_ms CHECK (latency_ms >= 0),
    CONSTRAINT usage_raw_size_limit CHECK (
        usage_raw IS NULL OR pg_column_size(usage_raw) <= 65536
    )
);

-- ============================================================================
-- LLM_COST_AGGREGATES TABLE
-- ============================================================================
-- Stores rolling window cost aggregations keyed by a composite aggregation_key.
-- The aggregation_key encodes the grouping dimension (e.g., session_id, pattern_id,
-- repo, model) and the window defines the time range.
--
-- Design decisions:
--   - aggregation_key is a composite string (e.g., "session:abc-123" or
--     "model:gpt-4o") to support multiple grouping dimensions without
--     separate columns for each
--   - UNIQUE on (aggregation_key, window) enables upsert semantics
--   - estimated_coverage_pct tracks data quality (what fraction is API-reported
--     vs estimated)

CREATE TABLE IF NOT EXISTS llm_cost_aggregates (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Aggregation dimensions
    aggregation_key VARCHAR(512) NOT NULL,
    window cost_aggregation_window NOT NULL,

    -- Aggregated metrics
    total_cost_usd NUMERIC(14, 6) NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    call_count INTEGER NOT NULL DEFAULT 0,

    -- Data quality indicator
    estimated_coverage_pct NUMERIC(5, 2),

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per aggregation_key + window (upsert target)
    CONSTRAINT unique_aggregation_key_window UNIQUE (aggregation_key, window),

    -- Constraints
    CONSTRAINT non_negative_total_cost_usd CHECK (total_cost_usd >= 0),
    CONSTRAINT non_negative_agg_total_tokens CHECK (total_tokens >= 0),
    CONSTRAINT non_negative_call_count CHECK (call_count >= 0),
    CONSTRAINT valid_estimated_coverage_pct CHECK (
        estimated_coverage_pct IS NULL
        OR (estimated_coverage_pct >= 0.00 AND estimated_coverage_pct <= 100.00)
    )
);

-- ============================================================================
-- INDEXES: llm_call_metrics
-- ============================================================================

-- Session lookups (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_session_id
    ON llm_call_metrics (session_id);

-- Run lookups (within-session drill-down)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_run_id
    ON llm_call_metrics (run_id)
    WHERE run_id IS NOT NULL;

-- Model-based analysis (cost per model, usage patterns)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_model_id
    ON llm_call_metrics (model_id);

-- Correlation-based lookups (tracing across services)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_correlation_id
    ON llm_call_metrics (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- Time-based queries and TTL cleanup (descending for recent-first)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_created_at
    ON llm_call_metrics (created_at DESC);

-- Cost analysis: model + time range queries
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_model_created
    ON llm_call_metrics (model_id, created_at DESC);

-- Usage source filtering (find estimated vs API-reported calls)
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_usage_source
    ON llm_call_metrics (usage_source);

-- Deduplication via input_hash
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_input_hash
    ON llm_call_metrics (input_hash)
    WHERE input_hash IS NOT NULL;

-- ============================================================================
-- INDEXES: llm_cost_aggregates
-- ============================================================================

-- Aggregation key lookups (dashboard queries)
CREATE INDEX IF NOT EXISTS idx_llm_cost_aggregates_aggregation_key
    ON llm_cost_aggregates (aggregation_key);

-- Window-based queries (e.g., "all 24h aggregations")
CREATE INDEX IF NOT EXISTS idx_llm_cost_aggregates_window
    ON llm_cost_aggregates (window);

-- Time-based queries and TTL cleanup
CREATE INDEX IF NOT EXISTS idx_llm_cost_aggregates_updated_at
    ON llm_cost_aggregates (updated_at DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================
-- Auto-update updated_at on any row modification for llm_cost_aggregates

CREATE OR REPLACE FUNCTION update_llm_cost_aggregates_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_llm_cost_aggregates_updated_at ON llm_cost_aggregates;
CREATE TRIGGER trigger_llm_cost_aggregates_updated_at
    BEFORE UPDATE ON llm_cost_aggregates
    FOR EACH ROW
    EXECUTE FUNCTION update_llm_cost_aggregates_updated_at();

-- ============================================================================
-- COMMENTS: llm_call_metrics
-- ============================================================================

COMMENT ON TABLE llm_call_metrics IS
    'Per-LLM-call token usage and cost data (OMN-2236). Append-only table for '
    'tracking individual LLM API calls with token counts, estimated cost, and latency.';

COMMENT ON COLUMN llm_call_metrics.id IS
    'Auto-generated primary key (UUID)';
COMMENT ON COLUMN llm_call_metrics.correlation_id IS
    'Request correlation ID for tracing across services';
COMMENT ON COLUMN llm_call_metrics.session_id IS
    'Session identifier (logical reference, no FK - async event arrival)';
COMMENT ON COLUMN llm_call_metrics.run_id IS
    'Run identifier within a session (logical reference, no FK)';
COMMENT ON COLUMN llm_call_metrics.model_id IS
    'LLM model identifier (e.g., gpt-4o, claude-opus-4-20250514, qwen2.5-coder-14b)';
COMMENT ON COLUMN llm_call_metrics.prompt_tokens IS
    'Number of tokens in the prompt (NULL if not reported by provider)';
COMMENT ON COLUMN llm_call_metrics.completion_tokens IS
    'Number of tokens in the completion (NULL if not reported by provider)';
COMMENT ON COLUMN llm_call_metrics.total_tokens IS
    'Total tokens used (NULL if not reported by provider)';
COMMENT ON COLUMN llm_call_metrics.estimated_cost_usd IS
    'Estimated cost in USD (NULL for unknown models, NOT 0)';
COMMENT ON COLUMN llm_call_metrics.latency_ms IS
    'LLM call latency in milliseconds (sub-millisecond precision, 2 decimal places)';
COMMENT ON COLUMN llm_call_metrics.usage_source IS
    'Token usage data provenance: API (provider-reported), ESTIMATED (calculated), MISSING (unavailable)';
COMMENT ON COLUMN llm_call_metrics.usage_is_estimated IS
    'Whether token usage values are estimated (TRUE) or API-reported (FALSE)';
COMMENT ON COLUMN llm_call_metrics.usage_raw IS
    'Redacted provider response payload as JSONB (max 64KB)';
COMMENT ON COLUMN llm_call_metrics.input_hash IS
    'SHA-256 hash of input for deduplication (first 64 hex chars)';
COMMENT ON COLUMN llm_call_metrics.code_version IS
    'Application code version that produced this record';
COMMENT ON COLUMN llm_call_metrics.contract_version IS
    'ONEX contract version that produced this record';
COMMENT ON COLUMN llm_call_metrics.source IS
    'Source system or service that produced this record';
COMMENT ON COLUMN llm_call_metrics.created_at IS
    'Timestamp when record was created (TTL cleanup key)';

-- ============================================================================
-- COMMENTS: llm_cost_aggregates
-- ============================================================================

COMMENT ON TABLE llm_cost_aggregates IS
    'Rolling window cost aggregations for LLM usage (OMN-2236). Supports '
    'session-level, pattern-level, repo-level, and model-level cost tracking.';

COMMENT ON COLUMN llm_cost_aggregates.id IS
    'Auto-generated primary key (UUID)';
COMMENT ON COLUMN llm_cost_aggregates.aggregation_key IS
    'Composite grouping key (e.g., session:<id>, model:<name>, repo:<path>, pattern:<id>)';
COMMENT ON COLUMN llm_cost_aggregates.window IS
    'Aggregation time window: 24h, 7d, or 30d';
COMMENT ON COLUMN llm_cost_aggregates.total_cost_usd IS
    'Total estimated cost in USD for this aggregation key and window';
COMMENT ON COLUMN llm_cost_aggregates.total_tokens IS
    'Total tokens consumed for this aggregation key and window';
COMMENT ON COLUMN llm_cost_aggregates.call_count IS
    'Number of LLM calls in this aggregation key and window';
COMMENT ON COLUMN llm_cost_aggregates.estimated_coverage_pct IS
    'Percentage of calls with estimated (vs API-reported) usage data (0-100)';
COMMENT ON COLUMN llm_cost_aggregates.created_at IS
    'Row creation timestamp';
COMMENT ON COLUMN llm_cost_aggregates.updated_at IS
    'Last update timestamp (TTL key for upsert tables)';

-- ============================================================================
-- UPDATE DB_METADATA
-- ============================================================================

UPDATE public.db_metadata
SET schema_version = '031', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
