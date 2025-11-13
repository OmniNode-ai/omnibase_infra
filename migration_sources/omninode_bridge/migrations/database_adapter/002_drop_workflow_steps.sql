-- Rollback Migration: 002_drop_workflow_steps
-- Description: Drop workflow_steps table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_steps_step_order;
DROP INDEX IF EXISTS idx_workflow_steps_created_at;
DROP INDEX IF EXISTS idx_workflow_steps_status;
DROP INDEX IF EXISTS idx_workflow_steps_workflow_id;

-- Drop table
DROP TABLE IF EXISTS workflow_steps CASCADE;
