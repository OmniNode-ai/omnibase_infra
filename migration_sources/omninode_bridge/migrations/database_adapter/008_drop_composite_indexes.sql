-- Rollback: 008_add_composite_indexes
-- Description: Remove composite indexes added in migration 008
-- Dependencies: 008_add_composite_indexes
-- Created: 2025-10-09

-- Drop the composite indexes created in migration 008
DROP INDEX IF EXISTS idx_workflow_executions_namespace_state;
DROP INDEX IF EXISTS idx_workflow_steps_workflow_id_order;
