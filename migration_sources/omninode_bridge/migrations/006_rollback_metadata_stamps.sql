-- Rollback: 006_create_metadata_stamps
-- Description: Rollback metadata_stamps table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_metadata_stamps_namespace_created_at;
DROP INDEX IF EXISTS idx_metadata_stamps_created_at;
DROP INDEX IF EXISTS idx_metadata_stamps_namespace;
DROP INDEX IF EXISTS idx_metadata_stamps_file_hash;
DROP INDEX IF EXISTS idx_metadata_stamps_workflow_id;

-- Drop table (CASCADE will remove foreign key constraints)
DROP TABLE IF EXISTS metadata_stamps CASCADE;
