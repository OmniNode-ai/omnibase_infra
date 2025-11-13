-- Migration: 010_create_workflow_state
-- Description: Create workflow_state table for canonical state management
-- Dependencies: None (standalone canonical store table)
-- Created: 2025-10-28
-- Purpose: Canonical workflow state storage with optimistic concurrency control

-- Canonical workflow state table
CREATE TABLE IF NOT EXISTS workflow_state (
    workflow_key VARCHAR(255) PRIMARY KEY,
    version BIGINT NOT NULL DEFAULT 1 CHECK (version >= 1),
    state JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version >= 1),
    provenance JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_workflow_state_updated_at
    ON workflow_state(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_workflow_state_version
    ON workflow_state(version);

-- Composite index for version-based queries
CREATE INDEX IF NOT EXISTS idx_workflow_state_key_version
    ON workflow_state(workflow_key, version);

-- GIN indexes for JSONB fields (faster lookups)
CREATE INDEX IF NOT EXISTS idx_workflow_state_state
    ON workflow_state USING GIN(state);

CREATE INDEX IF NOT EXISTS idx_workflow_state_provenance
    ON workflow_state USING GIN(provenance);

-- Comments for documentation
COMMENT ON TABLE workflow_state IS 'Canonical workflow state storage with optimistic concurrency control';
COMMENT ON COLUMN workflow_state.workflow_key IS 'Unique workflow identifier (PRIMARY KEY)';
COMMENT ON COLUMN workflow_state.version IS 'Version number for optimistic locking (incremented on each update, starts at 1)';
COMMENT ON COLUMN workflow_state.state IS 'Current workflow state as JSONB (cannot be empty)';
COMMENT ON COLUMN workflow_state.updated_at IS 'Timestamp of last state update (auto-managed)';
COMMENT ON COLUMN workflow_state.schema_version IS 'Schema version for future migrations (default 1)';
COMMENT ON COLUMN workflow_state.provenance IS 'Provenance metadata (must contain effect_id and timestamp)';
COMMENT ON COLUMN workflow_state.created_at IS 'Timestamp of workflow state creation';
