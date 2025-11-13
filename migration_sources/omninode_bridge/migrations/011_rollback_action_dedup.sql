-- Rollback: 011_create_action_dedup
-- Description: Rollback action_dedup_log table and indexes
-- Created: 2025-10-21

-- Drop indexes first
DROP INDEX IF EXISTS idx_action_dedup_expires;

-- Drop table
DROP TABLE IF EXISTS action_dedup_log CASCADE;
