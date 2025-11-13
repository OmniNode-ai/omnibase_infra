-- Rollback: 005_create_node_registrations
-- Description: Rollback node_registrations table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_node_registrations_type_health;
DROP INDEX IF EXISTS idx_node_registrations_last_heartbeat;
DROP INDEX IF EXISTS idx_node_registrations_type;
DROP INDEX IF EXISTS idx_node_registrations_health;

-- Drop table
DROP TABLE IF EXISTS node_registrations CASCADE;
