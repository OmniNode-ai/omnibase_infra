-- Rollback: 012_add_health_endpoint_to_node_registrations
-- Description: Remove health_endpoint column from node_registrations table
-- Created: 2025-10-28

-- Drop index first
DROP INDEX IF EXISTS idx_node_registrations_health_endpoint;

-- Remove health_endpoint column
ALTER TABLE node_registrations
DROP COLUMN IF EXISTS health_endpoint;
