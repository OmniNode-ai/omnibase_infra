-- Migration: 004_create_bridge_states
-- Description: Create bridge_states table for tracking reducer aggregation state
-- Created: 2025-10-07

-- Bridge aggregation state table (from ModelBridgeState)
CREATE TABLE IF NOT EXISTS bridge_states (
    bridge_id UUID PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    total_workflows_processed INTEGER NOT NULL DEFAULT 0,
    total_items_aggregated INTEGER NOT NULL DEFAULT 0,
    aggregation_metadata JSONB DEFAULT '{}',
    current_fsm_state VARCHAR(50) NOT NULL,
    last_aggregation_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_bridge_states_namespace ON bridge_states(namespace);
CREATE INDEX IF NOT EXISTS idx_bridge_states_fsm_state ON bridge_states(current_fsm_state);
CREATE INDEX IF NOT EXISTS idx_bridge_states_updated_at ON bridge_states(updated_at DESC);

-- Add comments for documentation
COMMENT ON TABLE bridge_states IS 'Tracks aggregation state for NodeBridgeReducer instances';
COMMENT ON COLUMN bridge_states.bridge_id IS 'Unique identifier for the bridge reducer instance';
COMMENT ON COLUMN bridge_states.namespace IS 'Multi-tenant namespace for aggregation grouping';
COMMENT ON COLUMN bridge_states.total_workflows_processed IS 'Counter of workflows processed by this bridge';
COMMENT ON COLUMN bridge_states.total_items_aggregated IS 'Counter of total items aggregated';
COMMENT ON COLUMN bridge_states.aggregation_metadata IS 'Additional aggregation metadata as JSONB';
COMMENT ON COLUMN bridge_states.current_fsm_state IS 'Current FSM state of the bridge';
COMMENT ON COLUMN bridge_states.last_aggregation_timestamp IS 'Timestamp of last aggregation window completion';
