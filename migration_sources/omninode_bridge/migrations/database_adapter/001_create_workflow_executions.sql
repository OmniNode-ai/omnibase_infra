-- Migration: 001_create_workflow_executions
-- Description: Create workflow_executions table for tracking orchestrator workflow execution
-- Created: 2025-10-07

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Workflow execution tracking table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL UNIQUE,
    workflow_type VARCHAR(100) NOT NULL,
    current_state VARCHAR(50) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflow_executions_correlation_id ON workflow_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_namespace ON workflow_executions(namespace);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_state ON workflow_executions(current_state);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_created_at ON workflow_executions(created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE workflow_executions IS 'Tracks workflow execution state for orchestrator workflows';
COMMENT ON COLUMN workflow_executions.correlation_id IS 'Unique correlation ID for tracking across services';
COMMENT ON COLUMN workflow_executions.workflow_type IS 'Type of workflow (e.g., metadata_stamping)';
COMMENT ON COLUMN workflow_executions.current_state IS 'Current FSM state (PENDING, PROCESSING, COMPLETED, FAILED)';
COMMENT ON COLUMN workflow_executions.namespace IS 'Multi-tenant namespace identifier';
COMMENT ON COLUMN workflow_executions.metadata IS 'Additional workflow metadata as JSONB';
