-- Migration: 005_create_node_registrations
-- Description: Create node_registrations table for service discovery and health tracking
-- Dependencies: None (standalone registry table)
-- Created: 2025-10-07

-- Node registration table (already exists in NodeBridgeRegistry)
CREATE TABLE IF NOT EXISTS node_registrations (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100) NOT NULL,
    node_version VARCHAR(50) NOT NULL,
    capabilities JSONB DEFAULT '{}',
    endpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    health_status VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_node_registrations_health
    ON node_registrations(health_status);
CREATE INDEX IF NOT EXISTS idx_node_registrations_type
    ON node_registrations(node_type);
CREATE INDEX IF NOT EXISTS idx_node_registrations_last_heartbeat
    ON node_registrations(last_heartbeat DESC);

-- Compound index for common queries
CREATE INDEX IF NOT EXISTS idx_node_registrations_type_health
    ON node_registrations(node_type, health_status);

-- Comments for documentation
COMMENT ON TABLE node_registrations IS 'Tracks registered nodes for service discovery and health monitoring';
COMMENT ON COLUMN node_registrations.node_id IS 'Unique identifier for node instance';
COMMENT ON COLUMN node_registrations.node_type IS 'Type of node (orchestrator, reducer, registry, etc)';
COMMENT ON COLUMN node_registrations.node_version IS 'Node version for compatibility tracking';
COMMENT ON COLUMN node_registrations.capabilities IS 'Node capabilities and feature flags as JSON';
COMMENT ON COLUMN node_registrations.endpoints IS 'Network endpoints for node communication as JSON';
COMMENT ON COLUMN node_registrations.metadata IS 'Additional node metadata as JSON';
COMMENT ON COLUMN node_registrations.health_status IS 'Current health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)';
COMMENT ON COLUMN node_registrations.last_heartbeat IS 'Timestamp of most recent heartbeat signal';
