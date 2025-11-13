-- Rollback Migration: 003_drop_fsm_transitions
-- Description: Drop fsm_transitions table and related objects
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_fsm_transitions_to_state;
DROP INDEX IF EXISTS idx_fsm_transitions_created_at;
DROP INDEX IF EXISTS idx_fsm_transitions_entity_type;
DROP INDEX IF EXISTS idx_fsm_transitions_entity_id;
DROP INDEX IF EXISTS idx_fsm_transitions_entity;

-- Drop table
DROP TABLE IF EXISTS fsm_transitions CASCADE;
