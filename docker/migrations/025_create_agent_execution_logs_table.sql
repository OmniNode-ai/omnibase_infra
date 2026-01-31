-- Migration: 025_create_agent_execution_logs_table
-- Description: Create agent_execution_logs table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent execution logs for complete execution lifecycle tracking.
-- Unlike other observability tables, this table supports UPSERT operations to
-- update execution status as it progresses (started -> running -> completed/failed).
--
-- Idempotency: INSERT ... ON CONFLICT (execution_id) DO UPDATE
-- TTL Column: updated_at (NOT created_at - uses most recent update for TTL)
--
-- Rollback: DROP TABLE IF EXISTS agent_execution_logs;

-- ============================================================================
-- AGENT_EXECUTION_LOGS TABLE
-- ============================================================================
-- Records agent execution lifecycle with upsert support for status updates.
-- Uses execution_id as primary key, supports ON CONFLICT DO UPDATE.

CREATE TABLE IF NOT EXISTS agent_execution_logs (
    -- Identity (execution_id is the unique key for upserts)
    execution_id UUID PRIMARY KEY,
    correlation_id UUID,
    session_id VARCHAR(255),

    -- Agent context
    agent_name VARCHAR(255),
    user_prompt TEXT,
    status VARCHAR(50),

    -- Execution metadata
    metadata JSONB,

    -- Lifecycle timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Performance
    duration_ms INTEGER,
    quality_score DECIMAL(5,2),

    -- Error tracking
    error_message TEXT,
    error_type VARCHAR(100),

    -- Project context
    project_path TEXT,
    project_name VARCHAR(255),

    -- Session context
    claude_session_id VARCHAR(255),
    terminal_id VARCHAR(255),

    -- Audit (TTL keys off updated_at for upsert tables)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================
-- TTL cleanup index on updated_at (not created_at) for upsert tables

CREATE INDEX IF NOT EXISTS idx_agent_execution_logs_updated_at
    ON agent_execution_logs (updated_at);

-- ============================================================================
-- TRIGGERS
-- ============================================================================
-- Auto-update updated_at on any row modification

CREATE OR REPLACE FUNCTION update_agent_execution_logs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_agent_execution_logs_updated_at ON agent_execution_logs;
CREATE TRIGGER trigger_agent_execution_logs_updated_at
    BEFORE UPDATE ON agent_execution_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_execution_logs_updated_at();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_execution_logs IS 'Agent execution lifecycle logs with upsert support (OMN-1743). TTL keys off updated_at.';
COMMENT ON COLUMN agent_execution_logs.execution_id IS 'Unique execution identifier (UUID) - PRIMARY KEY for upsert operations';
COMMENT ON COLUMN agent_execution_logs.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_execution_logs.session_id IS 'Session identifier for grouping related executions';
COMMENT ON COLUMN agent_execution_logs.agent_name IS 'Name of the agent being executed';
COMMENT ON COLUMN agent_execution_logs.user_prompt IS 'User prompt that triggered execution';
COMMENT ON COLUMN agent_execution_logs.status IS 'Execution status: pending, running, completed, failed';
COMMENT ON COLUMN agent_execution_logs.metadata IS 'JSON metadata about execution';
COMMENT ON COLUMN agent_execution_logs.started_at IS 'Timestamp when execution started';
COMMENT ON COLUMN agent_execution_logs.completed_at IS 'Timestamp when execution completed';
COMMENT ON COLUMN agent_execution_logs.duration_ms IS 'Total execution duration in milliseconds';
COMMENT ON COLUMN agent_execution_logs.quality_score IS 'Quality score for the execution (0.00-100.00)';
COMMENT ON COLUMN agent_execution_logs.error_message IS 'Error message if execution failed';
COMMENT ON COLUMN agent_execution_logs.error_type IS 'Error classification if execution failed';
COMMENT ON COLUMN agent_execution_logs.project_path IS 'Full filesystem path to the project';
COMMENT ON COLUMN agent_execution_logs.project_name IS 'Human-readable project name';
COMMENT ON COLUMN agent_execution_logs.claude_session_id IS 'Claude Code session identifier';
COMMENT ON COLUMN agent_execution_logs.terminal_id IS 'Terminal identifier where execution occurred';
COMMENT ON COLUMN agent_execution_logs.created_at IS 'Timestamp when log was first created';
COMMENT ON COLUMN agent_execution_logs.updated_at IS 'Timestamp of most recent update (TTL cleanup key)';
