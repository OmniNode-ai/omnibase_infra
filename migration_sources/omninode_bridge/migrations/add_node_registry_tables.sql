-- Migration: Add Node Registry Tables
-- Purpose: Support dual registration system (Consul + PostgreSQL)
-- Date: 2025-10-24
-- ONEX v2.0 Compliance: Registry node support

-- ============================================================
-- Table: registered_nodes
-- Purpose: Store registered node metadata for tool orchestration
-- ============================================================

CREATE TABLE IF NOT EXISTS registered_nodes (
    -- Primary identification
    node_id VARCHAR(255) PRIMARY KEY,
    node_name VARCHAR(255) NOT NULL,
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('effect', 'compute', 'reducer', 'orchestrator')),
    version VARCHAR(50) NOT NULL,

    -- Capabilities (JSONB for flexible structure)
    capabilities JSONB DEFAULT '[]'::jsonb,

    -- Endpoints configuration
    endpoints JSONB DEFAULT '{}'::jsonb,

    -- Health monitoring
    health_status VARCHAR(50) NOT NULL DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    health_check_config JSONB,

    -- Registration metadata
    registration_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    consul_registered BOOLEAN DEFAULT FALSE,
    postgres_registered BOOLEAN DEFAULT TRUE,
    registration_source VARCHAR(100),

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for efficient queries
CREATE INDEX idx_registered_nodes_node_type ON registered_nodes(node_type);
CREATE INDEX idx_registered_nodes_health_status ON registered_nodes(health_status);
CREATE INDEX idx_registered_nodes_last_heartbeat ON registered_nodes(last_heartbeat);
CREATE INDEX idx_registered_nodes_created_at ON registered_nodes(created_at);

-- GIN index for capabilities search
CREATE INDEX idx_registered_nodes_capabilities ON registered_nodes USING GIN (capabilities);

-- GIN index for metadata search
CREATE INDEX idx_registered_nodes_metadata ON registered_nodes USING GIN (metadata);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_registered_nodes_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER trigger_update_registered_nodes_updated_at
    BEFORE UPDATE ON registered_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_registered_nodes_updated_at();

-- ============================================================
-- Table: node_heartbeats
-- Purpose: Track node heartbeat history for health monitoring
-- ============================================================

CREATE TABLE IF NOT EXISTS node_heartbeats (
    heartbeat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) NOT NULL REFERENCES registered_nodes(node_id) ON DELETE CASCADE,
    heartbeat_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    health_status VARCHAR(50) NOT NULL,
    health_metrics JSONB DEFAULT '{}'::jsonb,
    response_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for heartbeat queries
CREATE INDEX idx_node_heartbeats_node_id ON node_heartbeats(node_id);
CREATE INDEX idx_node_heartbeats_timestamp ON node_heartbeats(heartbeat_timestamp DESC);

-- ============================================================
-- Table: node_capability_index
-- Purpose: Denormalized index for fast capability lookups
-- ============================================================

CREATE TABLE IF NOT EXISTS node_capability_index (
    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) NOT NULL REFERENCES registered_nodes(node_id) ON DELETE CASCADE,
    capability_name VARCHAR(255) NOT NULL,
    capability_description TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint to prevent duplicate capability entries
    UNIQUE (node_id, capability_name)
);

-- Indexes for capability search
CREATE INDEX idx_node_capability_index_node_id ON node_capability_index(node_id);
CREATE INDEX idx_node_capability_index_capability_name ON node_capability_index(capability_name);

-- ============================================================
-- Table: codegen_metrics_aggregated
-- Purpose: Store aggregated code generation metrics
-- ============================================================

CREATE TABLE IF NOT EXISTS codegen_metrics_aggregated (
    aggregation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Window configuration
    window_type VARCHAR(50) NOT NULL CHECK (window_type IN ('hourly', 'daily', 'weekly', 'monthly')),
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Aggregation configuration
    aggregation_type VARCHAR(50) NOT NULL CHECK (aggregation_type IN ('node_type_grouping', 'quality_buckets', 'time_window', 'domain_grouping')),
    group_key VARCHAR(255) NOT NULL,

    -- Counts
    total_generations INTEGER DEFAULT 0,
    successful_generations INTEGER DEFAULT 0,
    failed_generations INTEGER DEFAULT 0,
    success_rate NUMERIC(5, 4) DEFAULT 0.0,

    -- Duration statistics
    duration_statistics JSONB,

    -- Quality statistics
    quality_statistics JSONB,

    -- Cost statistics
    cost_statistics JSONB,

    -- Stage statistics
    stage_statistics JSONB,

    -- Node type breakdown
    node_type_breakdown JSONB,

    -- Aggregation metadata
    aggregation_metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint to prevent duplicate aggregations
    UNIQUE (window_type, window_start, aggregation_type, group_key)
);

-- Indexes for metrics queries
CREATE INDEX idx_codegen_metrics_window_type ON codegen_metrics_aggregated(window_type);
CREATE INDEX idx_codegen_metrics_window_start ON codegen_metrics_aggregated(window_start DESC);
CREATE INDEX idx_codegen_metrics_aggregation_type ON codegen_metrics_aggregated(aggregation_type);
CREATE INDEX idx_codegen_metrics_group_key ON codegen_metrics_aggregated(group_key);

-- ============================================================
-- Table: workflow_state_history
-- Purpose: Track FSM state transitions for workflows
-- ============================================================

CREATE TABLE IF NOT EXISTS workflow_state_history (
    transition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    from_state VARCHAR(50) NOT NULL,
    to_state VARCHAR(50) NOT NULL,
    event VARCHAR(50) NOT NULL,
    transition_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    guard_conditions_met BOOLEAN DEFAULT TRUE,
    reason TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for state history queries
CREATE INDEX idx_workflow_state_history_workflow_id ON workflow_state_history(workflow_id);
CREATE INDEX idx_workflow_state_history_timestamp ON workflow_state_history(transition_timestamp DESC);
CREATE INDEX idx_workflow_state_history_to_state ON workflow_state_history(to_state);

-- ============================================================
-- Table: workflow_state_current
-- Purpose: Track current workflow states
-- ============================================================

CREATE TABLE IF NOT EXISTS workflow_state_current (
    workflow_id UUID PRIMARY KEY,
    current_state VARCHAR(50) NOT NULL,
    previous_state VARCHAR(50),
    state_entry_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    transition_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for state queries
CREATE INDEX idx_workflow_state_current_state ON workflow_state_current(current_state);
CREATE INDEX idx_workflow_state_current_entry_time ON workflow_state_current(state_entry_time DESC);

-- Trigger to automatically update updated_at
CREATE TRIGGER trigger_update_workflow_state_current_updated_at
    BEFORE UPDATE ON workflow_state_current
    FOR EACH ROW
    EXECUTE FUNCTION update_registered_nodes_updated_at();

-- ============================================================
-- Views for common queries
-- ============================================================

-- View: Active healthy nodes by type
CREATE OR REPLACE VIEW v_active_healthy_nodes AS
SELECT
    node_id,
    node_name,
    node_type,
    version,
    capabilities,
    endpoints,
    last_heartbeat,
    registration_timestamp
FROM registered_nodes
WHERE health_status = 'healthy'
  AND deleted_at IS NULL
  AND last_heartbeat > (CURRENT_TIMESTAMP - INTERVAL '5 minutes')
ORDER BY node_type, node_name;

-- View: Node capability summary
CREATE OR REPLACE VIEW v_node_capability_summary AS
SELECT
    rn.node_type,
    nci.capability_name,
    COUNT(DISTINCT rn.node_id) as node_count
FROM registered_nodes rn
JOIN node_capability_index nci ON rn.node_id = nci.node_id
WHERE rn.deleted_at IS NULL
GROUP BY rn.node_type, nci.capability_name
ORDER BY rn.node_type, node_count DESC;

-- View: Recent metrics by aggregation type
CREATE OR REPLACE VIEW v_recent_metrics_summary AS
SELECT
    window_type,
    aggregation_type,
    group_key,
    AVG(success_rate) as avg_success_rate,
    SUM(total_generations) as total_generations,
    MAX(window_start) as latest_window
FROM codegen_metrics_aggregated
WHERE window_start > (CURRENT_TIMESTAMP - INTERVAL '24 hours')
GROUP BY window_type, aggregation_type, group_key
ORDER BY window_type, aggregation_type, group_key;

-- ============================================================
-- Comments for documentation
-- ============================================================

COMMENT ON TABLE registered_nodes IS 'Registry of all ONEX nodes with dual registration support (Consul + PostgreSQL)';
COMMENT ON TABLE node_heartbeats IS 'Historical heartbeat data for node health monitoring';
COMMENT ON TABLE node_capability_index IS 'Denormalized index for fast capability-based node discovery';
COMMENT ON TABLE codegen_metrics_aggregated IS 'Aggregated code generation metrics across multiple dimensions';
COMMENT ON TABLE workflow_state_history IS 'Complete FSM state transition history for all workflows';
COMMENT ON TABLE workflow_state_current IS 'Current FSM state for active workflows';

COMMENT ON COLUMN registered_nodes.consul_registered IS 'Indicates if node is registered in Consul service discovery';
COMMENT ON COLUMN registered_nodes.postgres_registered IS 'Indicates if node is registered in PostgreSQL tool registry';
COMMENT ON COLUMN codegen_metrics_aggregated.aggregation_type IS 'Type of aggregation: node_type_grouping, quality_buckets, time_window, or domain_grouping';
COMMENT ON COLUMN workflow_state_history.guard_conditions_met IS 'Whether all guard conditions were satisfied for the state transition';
