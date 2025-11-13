-- Rollback Migration: 011_canonical_workflow_state
-- Description: Rollback canonical workflow state table
-- Created: 2025-10-21

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_state_updated;
DROP INDEX IF EXISTS idx_workflow_state_version;

-- Drop table
DROP TABLE IF EXISTS workflow_state;
