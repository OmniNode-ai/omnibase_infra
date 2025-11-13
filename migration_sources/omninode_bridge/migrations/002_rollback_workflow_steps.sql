-- Rollback: 002_create_workflow_steps
-- Description: Rollback workflow_steps table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_steps_workflow_id_order;
DROP INDEX IF EXISTS idx_workflow_steps_created_at;
DROP INDEX IF EXISTS idx_workflow_steps_status;
DROP INDEX IF EXISTS idx_workflow_steps_workflow_id;

-- Drop table (CASCADE will remove foreign key constraints)
DROP TABLE IF EXISTS workflow_steps CASCADE;
