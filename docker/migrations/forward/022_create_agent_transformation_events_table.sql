-- Migration: 022_create_agent_transformation_events_table
-- Description: Create agent_transformation_events table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent transformation events when polymorphic agent transforms
-- from one agent type to another.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: See rollback/rollback_022_agent_transformation_events_table.sql

-- ============================================================================
-- AGENT_TRANSFORMATION_EVENTS TABLE
-- ============================================================================
-- Records agent transformation lifecycle events.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_transformation_events (
    -- Identity
    id UUID PRIMARY KEY,
    correlation_id UUID,

    -- Transformation details
    source_agent VARCHAR(255),
    target_agent VARCHAR(255),

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Optional context
    trigger TEXT,
    context TEXT,
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

CREATE INDEX IF NOT EXISTS idx_agent_transformation_events_created_at
    ON agent_transformation_events (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE agent_transformation_events IS 'Agent transformation lifecycle events (OMN-1743). Append-only observability table.';
COMMENT ON COLUMN agent_transformation_events.id IS 'Unique event identifier (UUID) - primary key for idempotent inserts';
COMMENT ON COLUMN agent_transformation_events.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_transformation_events.source_agent IS 'Agent type before transformation (e.g., polymorphic-agent)';
COMMENT ON COLUMN agent_transformation_events.target_agent IS 'Agent type after transformation (e.g., api-architect)';
COMMENT ON COLUMN agent_transformation_events.created_at IS 'Timestamp when event was recorded (TTL cleanup key)';
COMMENT ON COLUMN agent_transformation_events.trigger IS 'Trigger that caused the transformation';
COMMENT ON COLUMN agent_transformation_events.context IS 'Context information about the transformation';
COMMENT ON COLUMN agent_transformation_events.metadata IS 'Additional metadata about the transformation (JSONB)';
COMMENT ON COLUMN agent_transformation_events.project_path IS 'Absolute path to the project being worked on (absorbed from omniclaude OMN-2057)';
COMMENT ON COLUMN agent_transformation_events.project_name IS 'Human-readable project name (absorbed from omniclaude OMN-2057)';
COMMENT ON COLUMN agent_transformation_events.claude_session_id IS 'Claude Code session identifier (absorbed from omniclaude OMN-2057)';
