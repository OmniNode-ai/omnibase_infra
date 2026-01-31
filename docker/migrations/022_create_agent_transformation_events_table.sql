-- Migration: 022_create_agent_transformation_events_table
-- Description: Create agent_transformation_events table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent transformation events when polymorphic agent transforms
-- from one agent type to another. Captures full lifecycle including initialization,
-- execution, and completion with quality metrics.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: DROP TABLE IF EXISTS agent_transformation_events;

-- ============================================================================
-- AGENT_TRANSFORMATION_EVENTS TABLE
-- ============================================================================
-- Records agent transformation lifecycle events.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_transformation_events (
    -- Identity
    id UUID PRIMARY KEY,
    event_type VARCHAR(50),
    correlation_id UUID,
    session_id VARCHAR(255),

    -- Transformation details
    source_agent VARCHAR(255),
    target_agent VARCHAR(255),
    transformation_reason TEXT,
    user_request TEXT,

    -- Routing context
    routing_confidence DECIMAL(5,4),
    routing_strategy VARCHAR(100),

    -- Performance timing
    transformation_duration_ms INTEGER,
    initialization_duration_ms INTEGER,
    total_execution_duration_ms INTEGER,

    -- Outcome
    success BOOLEAN,
    error_message TEXT,
    error_type VARCHAR(100),
    quality_score DECIMAL(5,2),

    -- Context snapshot
    context_snapshot JSONB,
    context_keys TEXT[],
    context_size_bytes INTEGER,

    -- Agent definition
    agent_definition_id VARCHAR(255),

    -- Event hierarchy
    parent_event_id UUID,

    -- Lifecycle timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
COMMENT ON COLUMN agent_transformation_events.event_type IS 'Type of transformation event: started, completed, failed';
COMMENT ON COLUMN agent_transformation_events.correlation_id IS 'Request correlation ID for tracing across services';
COMMENT ON COLUMN agent_transformation_events.session_id IS 'Claude Code session identifier';
COMMENT ON COLUMN agent_transformation_events.source_agent IS 'Agent type before transformation (e.g., polymorphic)';
COMMENT ON COLUMN agent_transformation_events.target_agent IS 'Agent type after transformation';
COMMENT ON COLUMN agent_transformation_events.transformation_reason IS 'Explanation for why transformation occurred';
COMMENT ON COLUMN agent_transformation_events.user_request IS 'User request that triggered transformation';
COMMENT ON COLUMN agent_transformation_events.routing_confidence IS 'Confidence score for the transformation (0.0000-1.0000)';
COMMENT ON COLUMN agent_transformation_events.routing_strategy IS 'Strategy used for transformation decision';
COMMENT ON COLUMN agent_transformation_events.transformation_duration_ms IS 'Time to complete transformation in milliseconds';
COMMENT ON COLUMN agent_transformation_events.initialization_duration_ms IS 'Time to initialize target agent in milliseconds';
COMMENT ON COLUMN agent_transformation_events.total_execution_duration_ms IS 'Total execution time including transformation in milliseconds';
COMMENT ON COLUMN agent_transformation_events.success IS 'Whether transformation completed successfully';
COMMENT ON COLUMN agent_transformation_events.error_message IS 'Error message if transformation failed';
COMMENT ON COLUMN agent_transformation_events.error_type IS 'Error classification if transformation failed';
COMMENT ON COLUMN agent_transformation_events.quality_score IS 'Quality score for the transformation (0.00-100.00)';
COMMENT ON COLUMN agent_transformation_events.context_snapshot IS 'JSON snapshot of context at transformation time';
COMMENT ON COLUMN agent_transformation_events.context_keys IS 'Array of context keys present during transformation';
COMMENT ON COLUMN agent_transformation_events.context_size_bytes IS 'Size of context snapshot in bytes';
COMMENT ON COLUMN agent_transformation_events.agent_definition_id IS 'Identifier of the agent definition used';
COMMENT ON COLUMN agent_transformation_events.parent_event_id IS 'UUID of parent event for nested transformations';
COMMENT ON COLUMN agent_transformation_events.started_at IS 'Timestamp when transformation started';
COMMENT ON COLUMN agent_transformation_events.completed_at IS 'Timestamp when transformation completed';
COMMENT ON COLUMN agent_transformation_events.created_at IS 'Timestamp when event was recorded (TTL cleanup key)';
