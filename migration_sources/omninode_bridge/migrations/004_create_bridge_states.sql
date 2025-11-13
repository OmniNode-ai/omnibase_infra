-- Migration: 004_create_bridge_states
-- Description: Create bridge_states table for tracking aggregation state
-- Dependencies: None (standalone state table)
-- Created: 2025-10-07

-- Bridge aggregation state table (from ModelBridgeState)
CREATE TABLE IF NOT EXISTS bridge_states (
    bridge_id UUID PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    total_workflows_processed INTEGER NOT NULL DEFAULT 0 CHECK (total_workflows_processed >= 0),
    total_items_aggregated INTEGER NOT NULL DEFAULT 0 CHECK (total_items_aggregated >= 0),
    aggregation_metadata JSONB DEFAULT '{}',
    current_fsm_state VARCHAR(50) NOT NULL,
    last_aggregation_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_bridge_states_namespace
    ON bridge_states(namespace);
CREATE INDEX IF NOT EXISTS idx_bridge_states_fsm_state
    ON bridge_states(current_fsm_state);
CREATE INDEX IF NOT EXISTS idx_bridge_states_last_aggregation
    ON bridge_states(last_aggregation_timestamp DESC);

-- Compound index for common queries
CREATE INDEX IF NOT EXISTS idx_bridge_states_namespace_state
    ON bridge_states(namespace, current_fsm_state);

-- Comments for documentation
COMMENT ON TABLE bridge_states IS 'Tracks aggregation state for bridge reducer nodes';
COMMENT ON COLUMN bridge_states.bridge_id IS 'Unique identifier for bridge instance';
COMMENT ON COLUMN bridge_states.namespace IS 'Multi-tenant isolation namespace';
COMMENT ON COLUMN bridge_states.total_workflows_processed IS 'Cumulative count of workflows processed';
COMMENT ON COLUMN bridge_states.total_items_aggregated IS 'Cumulative count of items aggregated';
COMMENT ON COLUMN bridge_states.aggregation_metadata IS 'Aggregation statistics and metadata as JSON';
COMMENT ON COLUMN bridge_states.current_fsm_state IS 'Current finite state machine state';
COMMENT ON COLUMN bridge_states.last_aggregation_timestamp IS 'Timestamp of most recent aggregation';
