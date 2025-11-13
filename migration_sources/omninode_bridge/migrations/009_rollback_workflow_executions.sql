-- Rollback Migration: 009_enhance_workflow_executions
-- Description: Remove orchestrator-specific enhancements from workflow_executions
-- Created: 2025-10-15

-- Drop indexes first (to avoid dependency issues)
DROP INDEX IF EXISTS idx_workflow_executions_intelligence_data_gin;
DROP INDEX IF EXISTS idx_workflow_executions_workflow_steps_gin;
DROP INDEX IF EXISTS idx_workflow_executions_session_id;
DROP INDEX IF EXISTS idx_workflow_executions_file_hash;
DROP INDEX IF EXISTS idx_workflow_executions_stamp_id;

-- Remove orchestrator-specific columns
ALTER TABLE workflow_executions
DROP COLUMN IF EXISTS session_id,
DROP COLUMN IF EXISTS workflow_steps_executed,
DROP COLUMN IF EXISTS hash_generation_time_ms,
DROP COLUMN IF EXISTS intelligence_data,
DROP COLUMN IF EXISTS workflow_steps,
DROP COLUMN IF EXISTS file_hash,
DROP COLUMN IF EXISTS stamp_id;
