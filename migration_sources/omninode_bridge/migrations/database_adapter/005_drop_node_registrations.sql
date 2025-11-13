-- Rollback Migration: 005_drop_node_registrations
-- Description: Drop node_registrations table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_node_registrations_updated_at;
DROP INDEX IF EXISTS idx_node_registrations_last_heartbeat;
DROP INDEX IF EXISTS idx_node_registrations_node_type;
DROP INDEX IF EXISTS idx_node_registrations_health;

-- Drop table
DROP TABLE IF EXISTS node_registrations CASCADE;
