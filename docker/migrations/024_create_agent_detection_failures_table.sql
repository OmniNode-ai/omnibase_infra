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
-- Rollback: DROP TABLE IF EXISTS agent_detection_failures;

-- ============================================================================
-- AGENT_DETECTION_FAILURES TABLE
-- ============================================================================
-- Records detection failures for analysis and improvement.
-- Append-only table with correlation_id as primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_detection_failures (
    -- Identity (correlation_id is the unique key)
    correlation_id UUID PRIMARY KEY,

    -- User prompt context
    user_prompt TEXT,
    prompt_length INTEGER,
    prompt_hash VARCHAR(64),

    -- Detection outcome
    detection_status VARCHAR(50),
    failure_reason TEXT,
    detection_metadata JSONB,
    attempted_methods TEXT[],

    -- Project context
    project_path TEXT,
    project_name VARCHAR(255),

    -- Session context
    claude_session_id VARCHAR(255),

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
COMMENT ON COLUMN agent_detection_failures.user_prompt IS 'User prompt that failed detection';
COMMENT ON COLUMN agent_detection_failures.prompt_length IS 'Length of user prompt in characters';
COMMENT ON COLUMN agent_detection_failures.prompt_hash IS 'SHA-256 hash of prompt for deduplication analysis';
COMMENT ON COLUMN agent_detection_failures.detection_status IS 'Status of detection attempt (e.g., no_match, ambiguous, timeout)';
COMMENT ON COLUMN agent_detection_failures.failure_reason IS 'Human-readable explanation of detection failure';
COMMENT ON COLUMN agent_detection_failures.detection_metadata IS 'JSON metadata about detection attempt';
COMMENT ON COLUMN agent_detection_failures.attempted_methods IS 'Array of detection methods attempted';
COMMENT ON COLUMN agent_detection_failures.project_path IS 'Full filesystem path to the project';
COMMENT ON COLUMN agent_detection_failures.project_name IS 'Human-readable project name';
COMMENT ON COLUMN agent_detection_failures.claude_session_id IS 'Claude Code session identifier';
COMMENT ON COLUMN agent_detection_failures.created_at IS 'Timestamp when failure was recorded (TTL cleanup key)';
