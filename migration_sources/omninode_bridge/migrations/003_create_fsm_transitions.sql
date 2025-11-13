-- Migration: 003_create_fsm_transitions
-- Description: Create fsm_transitions table for tracking state machine transitions
-- Dependencies: None (entity_id is generic reference, not foreign key)
-- Created: 2025-10-07

-- FSM state transition history table
CREATE TABLE IF NOT EXISTS fsm_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    from_state VARCHAR(50),
    to_state VARCHAR(50) NOT NULL,
    transition_event VARCHAR(100) NOT NULL,
    transition_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_entity
    ON fsm_transitions(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_entity_type
    ON fsm_transitions(entity_type);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_created_at
    ON fsm_transitions(created_at DESC);

-- Index for state transition analysis
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_states
    ON fsm_transitions(from_state, to_state);

-- Comments for documentation
COMMENT ON TABLE fsm_transitions IS 'Tracks all state machine transitions across entities';
COMMENT ON COLUMN fsm_transitions.entity_id IS 'ID of entity undergoing transition (workflow, bridge, etc)';
COMMENT ON COLUMN fsm_transitions.entity_type IS 'Type of entity (workflow_execution, bridge_state, etc)';
COMMENT ON COLUMN fsm_transitions.from_state IS 'Previous state (NULL for initial state)';
COMMENT ON COLUMN fsm_transitions.to_state IS 'New state after transition';
COMMENT ON COLUMN fsm_transitions.transition_event IS 'Event that triggered the transition';
COMMENT ON COLUMN fsm_transitions.transition_data IS 'Additional transition context as JSON';
