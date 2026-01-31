-- Migration: 020_create_agent_actions_table
-- Description: Create agent_actions table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent action events from Claude Code sessions.
-- Each action (tool call, decision, error, success) is recorded with correlation tracking.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: DROP TABLE IF EXISTS agent_actions;

-- ============================================================================
-- AGENT_ACTIONS TABLE
-- ============================================================================
-- Records individual agent actions during Claude Code sessions.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_actions (
    -- Identity
    id UUID PRIMARY KEY,
    correlation_id UUID,

    -- Agent context
    agent_name VARCHAR(255),
    action_type VARCHAR(50),
    action_name VARCHAR(255),
    action_details JSONB,
    debug_mode BOOLEAN DEFAULT FALSE,

    -- Performance
    duration_ms INTEGER,

    -- Project context
    project_path TEXT,
    project_name VARCHAR(255),
    working_directory TEXT,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================
-- Minimal indexing for write-heavy workload - only TTL cleanup index

CREATE INDEX IF NOT EXISTS idx_agent_actions_created_at
    ON agent_actions (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_actions IS 'Agent action events from Claude Code sessions (OMN-1743). Append-only observability table.';
COMMENT ON COLUMN agent_actions.id IS 'Unique action identifier (UUID) - primary key for idempotent inserts';
COMMENT ON COLUMN agent_actions.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_actions.agent_name IS 'Name of the agent that performed the action';
COMMENT ON COLUMN agent_actions.action_type IS 'Type of action: tool_call, decision, error, success';
COMMENT ON COLUMN agent_actions.action_name IS 'Specific action name within the type';
COMMENT ON COLUMN agent_actions.action_details IS 'JSON payload with action-specific details';
COMMENT ON COLUMN agent_actions.debug_mode IS 'Whether action was performed in debug mode';
COMMENT ON COLUMN agent_actions.duration_ms IS 'Action execution duration in milliseconds';
COMMENT ON COLUMN agent_actions.project_path IS 'Full filesystem path to the project';
COMMENT ON COLUMN agent_actions.project_name IS 'Human-readable project name';
COMMENT ON COLUMN agent_actions.working_directory IS 'Working directory when action was performed';
COMMENT ON COLUMN agent_actions.created_at IS 'Timestamp when action was recorded (TTL cleanup key)';
