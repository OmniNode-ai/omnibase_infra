-- Rollback Migration: 010_create_workflow_state
-- Description: Rollback workflow_state table creation
-- Created: 2025-10-28

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_state_provenance;
DROP INDEX IF EXISTS idx_workflow_state_state;
DROP INDEX IF EXISTS idx_workflow_state_key_version;
DROP INDEX IF EXISTS idx_workflow_state_version;
DROP INDEX IF EXISTS idx_workflow_state_updated_at;

-- Drop table
DROP TABLE IF EXISTS workflow_state;
