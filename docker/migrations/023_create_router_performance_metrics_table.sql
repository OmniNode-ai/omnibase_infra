-- Migration: 023_create_router_performance_metrics_table
-- Description: Create router_performance_metrics table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores performance metrics for the agent router.
-- Captures routing duration, cache effectiveness, and candidate evaluation stats.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: DROP TABLE IF EXISTS router_performance_metrics;

-- ============================================================================
-- ROUTER_PERFORMANCE_METRICS TABLE
-- ============================================================================
-- Records router performance metrics for optimization analysis.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS router_performance_metrics (
    -- Identity
    id UUID PRIMARY KEY,

    -- Query context
    query_text TEXT,

    -- Performance metrics
    routing_duration_ms INTEGER,
    cache_hit BOOLEAN,
    trigger_match_strategy VARCHAR(100),

    -- Detailed metrics
    confidence_components JSONB,
    candidates_evaluated INTEGER,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================
-- Minimal indexing for write-heavy workload - only TTL cleanup index

CREATE INDEX IF NOT EXISTS idx_router_performance_metrics_created_at
    ON router_performance_metrics (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE router_performance_metrics IS 'Router performance metrics for optimization analysis (OMN-1743). Append-only observability table.';
COMMENT ON COLUMN router_performance_metrics.id IS 'Unique metric identifier (UUID) - primary key for idempotent inserts';
COMMENT ON COLUMN router_performance_metrics.query_text IS 'Query/request text that was routed';
COMMENT ON COLUMN router_performance_metrics.routing_duration_ms IS 'Time to complete routing in milliseconds';
COMMENT ON COLUMN router_performance_metrics.cache_hit IS 'Whether routing result was served from cache';
COMMENT ON COLUMN router_performance_metrics.trigger_match_strategy IS 'Strategy used for trigger matching';
COMMENT ON COLUMN router_performance_metrics.confidence_components IS 'JSON breakdown of confidence score components';
COMMENT ON COLUMN router_performance_metrics.candidates_evaluated IS 'Number of agent candidates evaluated during routing';
COMMENT ON COLUMN router_performance_metrics.created_at IS 'Timestamp when metric was recorded (TTL cleanup key)';
