-- Migration: 001_init_infrastructure_schema.sql
-- Description: Initialize omnibase_infrastructure database with core infrastructure schema
-- Version: 1.0.0
-- Created: Initial infrastructure setup
CREATE SCHEMA IF NOT EXISTS infrastructure;

-- Create service registry table
CREATE TABLE IF NOT EXISTS infrastructure.service_registry (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    service_version VARCHAR(50) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    health_endpoint VARCHAR(500),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert some test data
INSERT INTO infrastructure.service_registry (service_name, service_version, endpoint, health_endpoint)
VALUES
    ('postgres-adapter', 'v1.0.0', 'http://localhost:8080', 'http://localhost:8080/health'),
    ('consul-adapter', 'v1.0.0', 'http://localhost:8081', 'http://localhost:8081/health')
ON CONFLICT DO NOTHING;
