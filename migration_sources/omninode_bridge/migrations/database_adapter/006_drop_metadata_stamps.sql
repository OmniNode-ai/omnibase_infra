-- Rollback Migration: 006_drop_metadata_stamps
-- Description: Drop metadata_stamps table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_metadata_stamps_created_at;
DROP INDEX IF EXISTS idx_metadata_stamps_namespace;
DROP INDEX IF EXISTS idx_metadata_stamps_file_hash;
DROP INDEX IF EXISTS idx_metadata_stamps_workflow_id;

-- Drop table
DROP TABLE IF EXISTS metadata_stamps CASCADE;
