-- Migration: 001_create_workflow_executions
-- Description: Create workflow_executions table for tracking workflow execution lifecycle
-- Dependencies: None (base table)
-- Created: 2025-10-07

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Workflow execution tracking table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL UNIQUE,
    workflow_type VARCHAR(100) NOT NULL,
    current_state VARCHAR(50) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER CHECK (execution_time_ms >= 0),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_workflow_executions_correlation_id
    ON workflow_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_namespace
    ON workflow_executions(namespace);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_state
    ON workflow_executions(current_state);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at
    ON workflow_executions(started_at DESC);

-- Comments for documentation
COMMENT ON TABLE workflow_executions IS 'Tracks workflow execution lifecycle and state';
COMMENT ON COLUMN workflow_executions.correlation_id IS 'Unique identifier for request correlation across services';
COMMENT ON COLUMN workflow_executions.current_state IS 'Current FSM state (PENDING, PROCESSING, COMPLETED, FAILED)';
COMMENT ON COLUMN workflow_executions.namespace IS 'Multi-tenant isolation namespace';
COMMENT ON COLUMN workflow_executions.metadata IS 'Additional workflow metadata as JSON';
