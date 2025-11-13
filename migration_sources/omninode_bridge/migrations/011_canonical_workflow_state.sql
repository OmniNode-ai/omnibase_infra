-- Migration: 011_canonical_workflow_state
-- Description: Create canonical workflow state table for pure reducer refactor
-- Dependencies: None (Foundation Wave 1, Workstream 1A)
-- Created: 2025-10-21
-- Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md (Wave 1, Workstream 1A)

-- Create workflow_state table
-- This table stores the canonical state for all workflows with optimistic concurrency control
-- using version-based conflict detection. It is the single source of truth for workflow state.
CREATE TABLE IF NOT EXISTS workflow_state (
    -- Primary key: human-readable workflow identifier
    workflow_key TEXT PRIMARY KEY,

    -- Version number for optimistic concurrency control
    -- Incremented on each state update to detect conflicts
    version BIGINT NOT NULL DEFAULT 1 CHECK (version >= 1),

    -- Current workflow state as JSONB (flexible, queryable)
    state JSONB NOT NULL,

    -- Timestamp of last state update (auto-managed)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Schema version for future migrations
    schema_version INT NOT NULL DEFAULT 1 CHECK (schema_version >= 1),

    -- Provenance metadata (effect_id, timestamp, action_id, etc.)
    provenance JSONB NOT NULL
);

-- Performance-optimized indexes

-- Index for version-based queries and conflict detection
CREATE INDEX IF NOT EXISTS idx_workflow_state_version
    ON workflow_state(workflow_key, version);

-- Index for temporal queries (most recent first)
CREATE INDEX IF NOT EXISTS idx_workflow_state_updated
    ON workflow_state(updated_at DESC);

-- Optional: GIN index for state JSONB queries (future enhancement)
-- CREATE INDEX IF NOT EXISTS idx_workflow_state_state_gin
--     ON workflow_state USING GIN(state);

-- Optional: GIN index for provenance JSONB queries (future enhancement)
-- CREATE INDEX IF NOT EXISTS idx_workflow_state_provenance_gin
--     ON workflow_state USING GIN(provenance);

-- Table and column comments
COMMENT ON TABLE workflow_state IS 'Canonical workflow state storage with version-based optimistic concurrency control';
COMMENT ON COLUMN workflow_state.workflow_key IS 'Human-readable workflow identifier (PRIMARY KEY)';
COMMENT ON COLUMN workflow_state.version IS 'Version number for optimistic locking (incremented on each update)';
COMMENT ON COLUMN workflow_state.state IS 'Current workflow state as JSONB';
COMMENT ON COLUMN workflow_state.updated_at IS 'Timestamp of last state update (auto-managed)';
COMMENT ON COLUMN workflow_state.schema_version IS 'Schema version for future migrations';
COMMENT ON COLUMN workflow_state.provenance IS 'Provenance metadata (effect_id, timestamp, action_id, etc.)';
