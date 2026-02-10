-- Migration: 024_create_agent_detection_failures_table
-- Description: Create agent_detection_failures table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent detection failures when the router cannot determine
-- an appropriate agent for a user request. Used for improving detection accuracy.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (correlation_id) DO NOTHING
-- TTL Column: created_at
-- Note: Uses correlation_id as PRIMARY KEY (not a separate id column)
--
-- Rollback: See rollback/rollback_024_agent_detection_failures_table.sql

-- ============================================================================
-- AGENT_DETECTION_FAILURES TABLE
-- ============================================================================
-- Records detection failures for analysis and improvement.
-- Append-only table with correlation_id as primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_detection_failures (
    -- Identity (correlation_id is the unique key)
    correlation_id UUID PRIMARY KEY,

    -- Failure details
    failure_reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Request context
    request_summary TEXT,
    attempted_patterns JSONB,
    fallback_used VARCHAR(255),
    error_code VARCHAR(100),
    metadata JSONB
);

-- ============================================================================
-- INDEXES
-- ============================================================================
-- Minimal indexing for write-heavy workload - only TTL cleanup index

CREATE INDEX IF NOT EXISTS idx_agent_detection_failures_created_at
    ON agent_detection_failures (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_detection_failures IS 'Agent detection failures for accuracy improvement (OMN-1743). Append-only observability table.';
COMMENT ON COLUMN agent_detection_failures.correlation_id IS 'Request correlation ID - PRIMARY KEY for idempotent inserts';
COMMENT ON COLUMN agent_detection_failures.failure_reason IS 'Human-readable explanation of detection failure';
COMMENT ON COLUMN agent_detection_failures.created_at IS 'Timestamp when failure was recorded (TTL cleanup key)';
COMMENT ON COLUMN agent_detection_failures.request_summary IS 'Summary of the request that failed routing';
COMMENT ON COLUMN agent_detection_failures.attempted_patterns IS 'JSON array of patterns attempted during detection';
COMMENT ON COLUMN agent_detection_failures.fallback_used IS 'Name of fallback agent if one was used';
COMMENT ON COLUMN agent_detection_failures.error_code IS 'Error code for categorization';
COMMENT ON COLUMN agent_detection_failures.metadata IS 'Additional JSON metadata about the failure';
