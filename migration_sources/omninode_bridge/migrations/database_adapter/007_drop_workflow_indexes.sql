-- Rollback: 007_add_missing_workflow_indexes
-- Description: Remove indexes added in migration 007
-- Dependencies: 007_add_missing_workflow_indexes
-- Created: 2025-10-08

-- Drop the indexes created in migration 007
DROP INDEX IF EXISTS idx_workflow_executions_workflow_type;
DROP INDEX IF EXISTS idx_workflow_executions_namespace_state;
