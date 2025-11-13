-- Migration: 008_add_composite_indexes
-- Description: Add composite indexes for common query patterns
-- Dependencies: 001_create_workflow_executions, 002_create_workflow_steps
-- Created: 2025-10-09
-- PR: #23

-- Composite index for namespace and state filtering queries
-- This index optimizes queries that filter workflows by namespace and state simultaneously
-- Common use case: Finding all workflows in a specific namespace with a particular state
CREATE INDEX IF NOT EXISTS idx_workflow_executions_namespace_state
    ON workflow_executions(namespace, current_state);

-- Composite index for workflow steps ordered by workflow and step order
-- This index optimizes queries that retrieve steps for a specific workflow in order
-- Common use case: Fetching all steps for a workflow execution in sequential order
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id_order
    ON workflow_steps(workflow_id, step_order);

-- Add comments for documentation
COMMENT ON INDEX idx_workflow_executions_namespace_state IS 'Composite index for multi-tenant state filtering (namespace + current_state)';
COMMENT ON INDEX idx_workflow_steps_workflow_id_order IS 'Composite index for efficient workflow step retrieval in order (workflow_id + step_order)';
