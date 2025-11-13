-- Rollback: 003_create_fsm_transitions
-- Description: Rollback fsm_transitions table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_fsm_transitions_states;
DROP INDEX IF EXISTS idx_fsm_transitions_created_at;
DROP INDEX IF EXISTS idx_fsm_transitions_entity_type;
DROP INDEX IF EXISTS idx_fsm_transitions_entity;

-- Drop table
DROP TABLE IF EXISTS fsm_transitions CASCADE;
