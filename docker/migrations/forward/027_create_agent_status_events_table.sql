-- Migration: 027_create_agent_status_events_table
-- Description: Create agent_status_events table for agent visibility (OMN-1849)
-- Created: 2026-02-07
--
-- Purpose: Stores agent status events consumed from Kafka for real-time agent
-- visibility. Each status event records an agent's current state, progress,
-- and phase within a session. Append-only, write-heavy workload.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: See 027_rollback_agent_status_events_table.sql

-- ============================================================================
-- AGENT_STATUS_EVENTS TABLE
-- ============================================================================
-- Records agent status transitions during Claude Code sessions.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_status_events (
    -- Identity
    id UUID PRIMARY KEY,
    correlation_id UUID NOT NULL,

    -- Agent context
    agent_name VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL,

    -- Schema versioning
    status_schema_version INTEGER NOT NULL DEFAULT 1,

    -- Status details
    message TEXT NOT NULL,
    progress DECIMAL(3, 2),
    current_phase VARCHAR(255),
    current_task TEXT,
    blocking_reason TEXT,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Extensible metadata
    metadata JSONB
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Correlation-based lookups (tracing across services)
CREATE INDEX IF NOT EXISTS idx_agent_status_correlation
    ON agent_status_events (correlation_id);

-- Agent + session lookups (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_agent_status_agent_session
    ON agent_status_events (agent_name, session_id);

-- State filtering (dashboard queries)
CREATE INDEX IF NOT EXISTS idx_agent_status_state
    ON agent_status_events (state);

-- Time-based queries and TTL cleanup (descending for recent-first)
CREATE INDEX IF NOT EXISTS idx_agent_status_created
    ON agent_status_events (created_at DESC);

-- Ordering index for timeline reconstruction per agent session
CREATE INDEX IF NOT EXISTS idx_agent_status_ordering
    ON agent_status_events (agent_name, session_id, created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_status_events IS 'Agent status events from Claude Code sessions (OMN-1849). Append-only observability table for real-time agent visibility.';
COMMENT ON COLUMN agent_status_events.id IS 'Unique event identifier (UUID) - primary key for idempotent inserts';
COMMENT ON COLUMN agent_status_events.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_status_events.agent_name IS 'Name of the reporting agent';
COMMENT ON COLUMN agent_status_events.session_id IS 'Session identifier for grouping status events';
COMMENT ON COLUMN agent_status_events.state IS 'Current agent state (idle, working, blocked, etc.)';
COMMENT ON COLUMN agent_status_events.status_schema_version IS 'Schema version for forward compatibility';
COMMENT ON COLUMN agent_status_events.message IS 'Human-readable status message';
COMMENT ON COLUMN agent_status_events.progress IS 'Progress indicator 0.00-1.00';
COMMENT ON COLUMN agent_status_events.current_phase IS 'Current workflow phase name';
COMMENT ON COLUMN agent_status_events.current_task IS 'Current task description';
COMMENT ON COLUMN agent_status_events.blocking_reason IS 'Reason for blocked state';
COMMENT ON COLUMN agent_status_events.created_at IS 'Timestamp when event was created by agent (TTL cleanup key)';
COMMENT ON COLUMN agent_status_events.received_at IS 'Timestamp when event was received by consumer';
COMMENT ON COLUMN agent_status_events.metadata IS 'JSON payload with additional status details';
