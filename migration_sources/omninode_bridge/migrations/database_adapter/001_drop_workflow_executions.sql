-- Rollback Migration: 001_drop_workflow_executions
-- Description: Drop workflow_executions table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_executions_created_at;
DROP INDEX IF EXISTS idx_workflow_executions_state;
DROP INDEX IF EXISTS idx_workflow_executions_namespace;
DROP INDEX IF EXISTS idx_workflow_executions_correlation_id;

-- Drop table (CASCADE will handle dependent foreign keys)
DROP TABLE IF EXISTS workflow_executions CASCADE;
