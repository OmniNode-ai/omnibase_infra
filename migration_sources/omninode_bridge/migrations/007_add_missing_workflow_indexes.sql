-- Migration: 007_add_missing_workflow_indexes
-- Description: Add missing database indexes for workflow_executions table
-- Dependencies: 001_create_workflow_executions
-- Created: 2025-10-08

-- Add missing index on workflow_type column for workflow type filtering
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_type
    ON workflow_executions(workflow_type);

-- Add composite index on (namespace, current_state) for common multi-tenant state filtering
-- This index will significantly improve performance for queries that filter by both namespace and state
CREATE INDEX IF NOT EXISTS idx_workflow_executions_namespace_state
    ON workflow_executions(namespace, current_state);

-- Add index on created_at for time-based queries (if not already present)
-- Note: Some migration versions already have this, but adding it here ensures consistency
CREATE INDEX IF NOT EXISTS idx_workflow_executions_created_at
    ON workflow_executions(created_at DESC);

-- Add comments for documentation
COMMENT ON INDEX idx_workflow_executions_workflow_type IS 'Index for workflow type filtering and queries';
COMMENT ON INDEX idx_workflow_executions_namespace_state IS 'Composite index for multi-tenant state filtering (namespace + current_state)';
COMMENT ON INDEX idx_workflow_executions_created_at IS 'Index for time-based queries and sorting by creation time';
