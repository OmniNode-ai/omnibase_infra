-- Rollback Migration: 004_drop_bridge_states
-- Description: Drop bridge_states table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_bridge_states_updated_at;
DROP INDEX IF EXISTS idx_bridge_states_fsm_state;
DROP INDEX IF EXISTS idx_bridge_states_namespace;

-- Drop table
DROP TABLE IF EXISTS bridge_states CASCADE;
