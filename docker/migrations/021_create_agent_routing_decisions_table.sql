-- Migration: 021_create_agent_routing_decisions_table
-- Description: Create agent_routing_decisions table for observability (OMN-1743)
-- Created: 2026-01-31
--
-- Purpose: Stores agent routing decisions from the polymorphic agent router.
-- Records which agent was selected, confidence scores, and routing strategy.
-- Write-heavy, append-only workload - minimal indexing for TTL cleanup only.
--
-- Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING
-- TTL Column: created_at
--
-- Rollback: DROP TABLE IF EXISTS agent_routing_decisions;

-- ============================================================================
-- AGENT_ROUTING_DECISIONS TABLE
-- ============================================================================
-- Records routing decisions made by the polymorphic agent router.
-- Append-only table with UUID primary key for idempotent inserts.

CREATE TABLE IF NOT EXISTS agent_routing_decisions (
    -- Identity
    id UUID PRIMARY KEY,
    correlation_id UUID,

    -- Request context
    user_request TEXT,

    -- Routing decision
    selected_agent VARCHAR(255),
    confidence_score DECIMAL(5,4),
    alternatives JSONB,
    reasoning TEXT,
    routing_strategy VARCHAR(100),

    -- Context snapshot for debugging
    context_snapshot JSONB,

    -- Performance
    routing_time_ms INTEGER,

    -- Audit (TTL keys off created_at)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
COMMENT ON COLUMN agent_routing_decisions.user_request IS 'Original user request that triggered routing';
COMMENT ON COLUMN agent_routing_decisions.selected_agent IS 'Name of the agent selected by the router';
COMMENT ON COLUMN agent_routing_decisions.confidence_score IS 'Router confidence in selection (0.0000-1.0000)';
COMMENT ON COLUMN agent_routing_decisions.alternatives IS 'JSON array of alternative agents considered with scores';
COMMENT ON COLUMN agent_routing_decisions.reasoning IS 'Human-readable explanation of routing decision';
COMMENT ON COLUMN agent_routing_decisions.routing_strategy IS 'Strategy used for routing (e.g., trigger_match, semantic, hybrid)';
COMMENT ON COLUMN agent_routing_decisions.context_snapshot IS 'JSON snapshot of context at decision time for debugging';
COMMENT ON COLUMN agent_routing_decisions.routing_time_ms IS 'Time taken to make routing decision in milliseconds';
COMMENT ON COLUMN agent_routing_decisions.created_at IS 'Timestamp when decision was recorded (TTL cleanup key)';
