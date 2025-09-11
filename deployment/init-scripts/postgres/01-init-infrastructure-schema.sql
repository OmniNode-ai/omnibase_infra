-- ONEX Infrastructure Database Initialization Script
-- Creates infrastructure schema and tables for PostgreSQL adapter testing

-- Create infrastructure schema
CREATE SCHEMA IF NOT EXISTS infrastructure;

-- Set search path to include infrastructure schema
ALTER DATABASE omnibase_infrastructure SET search_path TO infrastructure, public;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Grant usage on schema
GRANT USAGE ON SCHEMA infrastructure TO omnibase;
GRANT CREATE ON SCHEMA infrastructure TO omnibase;

-- === ADAPTER TESTING TABLES ===

-- Service registry table for testing service discovery patterns
CREATE TABLE infrastructure.service_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(255) NOT NULL,
    service_type VARCHAR(100) NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'unknown',
    metadata JSONB DEFAULT '{}',
    health_check_url VARCHAR(500),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_service_host_port UNIQUE (service_name, hostname, port)
);

-- Configuration management table for testing adapter configuration
CREATE TABLE infrastructure.configuration (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    config_type VARCHAR(100) NOT NULL DEFAULT 'application',
    environment VARCHAR(50) NOT NULL DEFAULT 'development',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

-- Event log table for testing message bus integration
CREATE TABLE infrastructure.event_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    correlation_id UUID,
    source_service VARCHAR(255) NOT NULL,
    target_service VARCHAR(255),
    event_data JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms NUMERIC(10,2)
);

-- Health check results table for testing health monitoring
CREATE TABLE infrastructure.health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(255) NOT NULL,
    check_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms NUMERIC(10,2),
    details JSONB DEFAULT '{}',
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_health_service_time ON (service_name, checked_at)
);

-- Connection pool metrics table for testing connection management
CREATE TABLE infrastructure.connection_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pool_name VARCHAR(255) NOT NULL,
    total_connections INTEGER NOT NULL,
    active_connections INTEGER NOT NULL,
    idle_connections INTEGER NOT NULL,
    failed_connections INTEGER DEFAULT 0,
    average_response_time_ms NUMERIC(10,2),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- === INDEXES FOR PERFORMANCE ===
CREATE INDEX idx_service_registry_name_status ON infrastructure.service_registry(service_name, status);
CREATE INDEX idx_service_registry_last_seen ON infrastructure.service_registry(last_seen);
CREATE INDEX idx_configuration_environment ON infrastructure.configuration(environment);
CREATE INDEX idx_event_log_correlation ON infrastructure.event_log(correlation_id);
CREATE INDEX idx_event_log_type_status ON infrastructure.event_log(event_type, status);
CREATE INDEX idx_event_log_created ON infrastructure.event_log(created_at);

-- === UPDATE TRIGGERS ===
CREATE OR REPLACE FUNCTION infrastructure.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_service_registry_updated_at 
    BEFORE UPDATE ON infrastructure.service_registry 
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_updated_at_column();

CREATE TRIGGER update_configuration_updated_at 
    BEFORE UPDATE ON infrastructure.configuration 
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_updated_at_column();

-- === SAMPLE DATA FOR TESTING ===

-- Insert sample service registry entries
INSERT INTO infrastructure.service_registry (service_name, service_type, hostname, port, status, metadata) VALUES
('postgres-adapter', 'effect', 'postgres-adapter', 8080, 'healthy', '{"version": "1.0.0", "domain": "infrastructure"}'),
('postgres', 'database', 'postgres', 5432, 'healthy', '{"version": "15", "type": "postgresql"}'),
('event-bus', 'messaging', 'event-bus', 8083, 'unknown', '{"version": "1.0.0", "type": "http"}');

-- Insert sample configuration
INSERT INTO infrastructure.configuration (config_key, config_value, config_type, environment, description) VALUES
('postgres.connection.pool_size', '{"min": 5, "max": 20}', 'database', 'development', 'PostgreSQL connection pool configuration'),
('postgres.connection.timeout', '60', 'database', 'development', 'Database connection timeout in seconds'),
('adapter.health_check.interval', '30', 'application', 'development', 'Health check interval in seconds'),
('event_bus.retry.attempts', '3', 'messaging', 'development', 'Maximum retry attempts for event publishing');

-- Insert sample event log entries
INSERT INTO infrastructure.event_log (event_type, correlation_id, source_service, target_service, event_data, status) VALUES
('postgres_health_check_request', uuid_generate_v4(), 'health-monitor', 'postgres-adapter', '{"include_connection_stats": true}', 'completed'),
('postgres_query_request', uuid_generate_v4(), 'api-service', 'postgres-adapter', '{"query": "SELECT COUNT(*) FROM infrastructure.service_registry", "timeout": 30}', 'completed'),
('postgres_adapter_initialized', uuid_generate_v4(), 'postgres-adapter', null, '{"startup_time": "2025-09-11T07:00:00Z"}', 'completed');

-- Grant permissions on all tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA infrastructure TO omnibase;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA infrastructure TO omnibase;

-- Final message
DO $$
BEGIN
    RAISE NOTICE 'ONEX Infrastructure schema initialized successfully';
    RAISE NOTICE 'Created tables: service_registry, configuration, event_log, health_checks, connection_metrics';
    RAISE NOTICE 'Database ready for PostgreSQL adapter testing';
END $$;