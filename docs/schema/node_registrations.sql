-- SPDX-License-Identifier: MIT
-- Copyright (c) 2025 OmniNode Team

-- =============================================================================
-- Node Registrations Table Schema
-- =============================================================================
-- Used by NodeRegistryEffect for persistent node registry state.
-- Supports UPSERT pattern for idempotent re-registration.
--
-- Related source files:
--   - src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py
--   - src/omnibase_infra/nodes/node_registry_effect/v1_0_0/models/model_node_registration.py
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_registrations (
    -- Primary identifier (unique node ID)
    node_id VARCHAR(255) PRIMARY KEY,

    -- Node classification
    node_type VARCHAR(50) NOT NULL,
    node_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',

    -- Node capabilities and configuration (stored as JSONB for flexibility)
    capabilities JSONB NOT NULL DEFAULT '{}',
    endpoints JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Health monitoring
    health_endpoint VARCHAR(512),
    last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Audit timestamps
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for filtering by node type (common query pattern)
CREATE INDEX IF NOT EXISTS idx_node_registrations_node_type
ON node_registrations(node_type);

-- Index for filtering by node version
CREATE INDEX IF NOT EXISTS idx_node_registrations_node_version
ON node_registrations(node_version);

-- Index for finding recently updated nodes
CREATE INDEX IF NOT EXISTS idx_node_registrations_updated_at
ON node_registrations(updated_at DESC);

-- Partial index for nodes with health endpoints
CREATE INDEX IF NOT EXISTS idx_node_registrations_health_endpoint
ON node_registrations(health_endpoint)
WHERE health_endpoint IS NOT NULL;

-- GIN index for JSONB capability queries
CREATE INDEX IF NOT EXISTS idx_node_registrations_capabilities
ON node_registrations USING GIN (capabilities);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE node_registrations IS 'ONEX node registry for service discovery and persistent state management';

COMMENT ON COLUMN node_registrations.node_id IS 'Unique identifier for the registered node';
COMMENT ON COLUMN node_registrations.node_type IS 'Node type: effect, compute, reducer, orchestrator';
COMMENT ON COLUMN node_registrations.node_version IS 'Semantic version of the node (e.g., 1.0.0)';
COMMENT ON COLUMN node_registrations.capabilities IS 'JSON object describing node capabilities';
COMMENT ON COLUMN node_registrations.endpoints IS 'JSON object mapping endpoint names to URLs';
COMMENT ON COLUMN node_registrations.metadata IS 'Additional node metadata as JSON';
COMMENT ON COLUMN node_registrations.health_endpoint IS 'URL for health check endpoint';
COMMENT ON COLUMN node_registrations.last_heartbeat IS 'Timestamp of last successful health check';
COMMENT ON COLUMN node_registrations.registered_at IS 'Initial registration timestamp';
COMMENT ON COLUMN node_registrations.updated_at IS 'Last update timestamp (auto-updated on UPSERT)';

-- =============================================================================
-- Example Queries
-- =============================================================================

-- Find all effect nodes
-- SELECT * FROM node_registrations WHERE node_type = 'effect';

-- Find nodes with specific capability
-- SELECT * FROM node_registrations WHERE capabilities @> '{"feature": "logging"}';

-- Find recently updated nodes (last hour)
-- SELECT * FROM node_registrations WHERE updated_at > NOW() - INTERVAL '1 hour';

-- Find nodes without recent heartbeat (potential issues)
-- SELECT * FROM node_registrations
-- WHERE last_heartbeat < NOW() - INTERVAL '5 minutes'
--   AND health_endpoint IS NOT NULL;
