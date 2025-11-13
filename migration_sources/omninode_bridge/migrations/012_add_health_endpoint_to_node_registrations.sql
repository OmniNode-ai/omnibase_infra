-- Migration: 012_add_health_endpoint_to_node_registrations
-- Description: Add health_endpoint column to node_registrations table
-- Dependencies: 005_create_node_registrations.sql
-- Created: 2025-10-28

-- Add health_endpoint column
ALTER TABLE node_registrations
ADD COLUMN IF NOT EXISTS health_endpoint VARCHAR(500);

-- Add index for health endpoint queries
CREATE INDEX IF NOT EXISTS idx_node_registrations_health_endpoint
    ON node_registrations(health_endpoint) WHERE health_endpoint IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN node_registrations.health_endpoint IS 'Optional health check endpoint URL for node monitoring';
