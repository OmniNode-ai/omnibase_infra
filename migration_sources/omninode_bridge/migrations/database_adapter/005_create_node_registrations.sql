-- Migration: 005_create_node_registrations
-- Description: Create node_registrations table for service discovery and health tracking
-- Created: 2025-10-07

-- Node registration table for service discovery
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_node_registrations_health ON node_registrations(health_status);
CREATE INDEX IF NOT EXISTS idx_node_registrations_node_type ON node_registrations(node_type);
CREATE INDEX IF NOT EXISTS idx_node_registrations_last_heartbeat ON node_registrations(last_heartbeat DESC);
CREATE INDEX IF NOT EXISTS idx_node_registrations_updated_at ON node_registrations(updated_at DESC);

-- Add comments for documentation
COMMENT ON TABLE node_registrations IS 'Service discovery and health tracking for bridge nodes';
COMMENT ON COLUMN node_registrations.node_id IS 'Unique node identifier (format: node_type-version-instance_id)';
COMMENT ON COLUMN node_registrations.node_type IS 'Type of node (orchestrator, reducer, registry, etc.)';
COMMENT ON COLUMN node_registrations.node_version IS 'Node version (e.g., v1.0.0)';
COMMENT ON COLUMN node_registrations.capabilities IS 'Node capabilities and features as JSONB';
COMMENT ON COLUMN node_registrations.endpoints IS 'Node endpoints for communication as JSONB';
COMMENT ON COLUMN node_registrations.health_status IS 'Current health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)';
COMMENT ON COLUMN node_registrations.last_heartbeat IS 'Timestamp of last heartbeat signal';
