-- Create the dashboard-compatible node service registry projection table.
-- Runtime projection handlers upsert node introspection and heartbeat events here.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS node_service_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name TEXT UNIQUE NOT NULL,
    service_url TEXT NOT NULL DEFAULT '',
    service_type TEXT NOT NULL DEFAULT 'api',
    health_status TEXT NOT NULL DEFAULT 'unknown',
    last_health_check TIMESTAMPTZ,
    last_heartbeat_at TIMESTAMPTZ,
    uptime_seconds BIGINT NOT NULL DEFAULT 0,
    health_check_interval_seconds INTEGER NOT NULL DEFAULT 60,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_node_service_registry_health_status
    ON node_service_registry (health_status);

CREATE INDEX IF NOT EXISTS idx_node_service_registry_service_type
    ON node_service_registry (service_type);

CREATE INDEX IF NOT EXISTS idx_node_service_registry_last_heartbeat
    ON node_service_registry (last_heartbeat_at DESC);
