-- Migration: 003_create_fsm_transitions
-- Description: Create fsm_transitions table for tracking FSM state transitions
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_entity ON fsm_transitions(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_entity_id ON fsm_transitions(entity_id);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_entity_type ON fsm_transitions(entity_type);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_created_at ON fsm_transitions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fsm_transitions_to_state ON fsm_transitions(to_state);

-- Add comments for documentation
COMMENT ON TABLE fsm_transitions IS 'Tracks finite state machine state transitions for workflows and aggregations';
COMMENT ON COLUMN fsm_transitions.entity_id IS 'ID of the entity (workflow_id, bridge_id, etc.)';
COMMENT ON COLUMN fsm_transitions.entity_type IS 'Type of entity (workflow, bridge_state, etc.)';
COMMENT ON COLUMN fsm_transitions.from_state IS 'Previous state (NULL for initial state)';
COMMENT ON COLUMN fsm_transitions.to_state IS 'New state after transition';
COMMENT ON COLUMN fsm_transitions.transition_event IS 'Event that triggered the transition';
COMMENT ON COLUMN fsm_transitions.transition_data IS 'Additional transition context as JSONB';
