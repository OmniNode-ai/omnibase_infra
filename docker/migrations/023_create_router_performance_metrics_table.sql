-- Migration: 023_create_router_performance_metrics_table
-- Description: Create router_performance_metrics table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores performance metrics for the agent router.
-- Captures timing, throughput, and resource usage metrics with dimensional labels.
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

    -- Metric identification
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Tracing
    correlation_id UUID,

    -- Metric context
    unit VARCHAR(50),
    agent_name VARCHAR(255),

    -- Dimensional data
    labels JSONB,
    metadata JSONB
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
COMMENT ON COLUMN router_performance_metrics.metric_name IS 'Name of the metric being recorded (e.g., routing_latency_ms)';
COMMENT ON COLUMN router_performance_metrics.metric_value IS 'Numeric value of the metric';
COMMENT ON COLUMN router_performance_metrics.created_at IS 'Timestamp when metric was recorded (TTL cleanup key)';
COMMENT ON COLUMN router_performance_metrics.correlation_id IS 'Request correlation ID for trace-specific metrics';
COMMENT ON COLUMN router_performance_metrics.unit IS 'Unit of measurement (ms, bytes, count, etc.)';
COMMENT ON COLUMN router_performance_metrics.agent_name IS 'Agent name if metric is agent-specific';
COMMENT ON COLUMN router_performance_metrics.labels IS 'Key-value labels for metric dimensionality (JSONB)';
COMMENT ON COLUMN router_performance_metrics.metadata IS 'Additional metadata about the metric (JSONB)';
