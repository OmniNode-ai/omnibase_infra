-- Rollback: 004_create_bridge_states
-- Description: Rollback bridge_states table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_bridge_states_namespace_state;
DROP INDEX IF EXISTS idx_bridge_states_last_aggregation;
DROP INDEX IF EXISTS idx_bridge_states_fsm_state;
DROP INDEX IF EXISTS idx_bridge_states_namespace;

-- Drop table
DROP TABLE IF EXISTS bridge_states CASCADE;
