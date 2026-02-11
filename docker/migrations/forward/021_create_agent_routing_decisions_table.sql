-- Migration: 021_create_agent_routing_decisions_table
-- Description: Create agent_routing_decisions table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent routing decisions from the polymorphic agent router.
-- Records which agent was selected, confidence scores, and routing metadata.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: See rollback/rollback_021_agent_routing_decisions_table.sql

-- ============================================================================
-- AGENT_ROUTING_DECISIONS TABLE
-- ============================================================================
-- Records routing decisions made by the polymorphic agent router.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_routing_decisions (
    -- Identity
    id UUID PRIMARY KEY,
    correlation_id UUID,

    -- Routing decision
    selected_agent VARCHAR(255),
    confidence_score DECIMAL(5,4),

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Request context
    request_type VARCHAR(100),
    alternatives JSONB,
    routing_reason TEXT,
    domain VARCHAR(255),
    metadata JSONB,

    -- Project context (absorbed from omniclaude - OMN-2057)
    project_path TEXT,
    project_name VARCHAR(255),
    claude_session_id VARCHAR(255)
);

-- ============================================================================
-- INDEXES
-- ============================================================================
-- Minimal indexing for write-heavy workload - only TTL cleanup index

CREATE INDEX IF NOT EXISTS idx_agent_routing_decisions_created_at
    ON agent_routing_decisions (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_routing_decisions IS 'Agent routing decisions from polymorphic router (OMN-1743). Append-only observability table.';
COMMENT ON COLUMN agent_routing_decisions.id IS 'Unique decision identifier (UUID) - primary key for idempotent inserts';
COMMENT ON COLUMN agent_routing_decisions.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_routing_decisions.selected_agent IS 'Name of the agent selected by the router';
COMMENT ON COLUMN agent_routing_decisions.confidence_score IS 'Router confidence in selection (0.0000-1.0000)';
COMMENT ON COLUMN agent_routing_decisions.created_at IS 'Timestamp when decision was recorded (TTL cleanup key)';
COMMENT ON COLUMN agent_routing_decisions.request_type IS 'Type of request being routed';
COMMENT ON COLUMN agent_routing_decisions.alternatives IS 'JSON array of alternative agents considered';
COMMENT ON COLUMN agent_routing_decisions.routing_reason IS 'Explanation for the routing decision';
COMMENT ON COLUMN agent_routing_decisions.domain IS 'Domain classification for the request';
COMMENT ON COLUMN agent_routing_decisions.metadata IS 'Additional metadata about the decision (JSON object)';
COMMENT ON COLUMN agent_routing_decisions.project_path IS 'Absolute path to the project being worked on (absorbed from omniclaude OMN-2057)';
COMMENT ON COLUMN agent_routing_decisions.project_name IS 'Human-readable project name (absorbed from omniclaude OMN-2057)';
COMMENT ON COLUMN agent_routing_decisions.claude_session_id IS 'Claude Code session identifier (absorbed from omniclaude OMN-2057)';
