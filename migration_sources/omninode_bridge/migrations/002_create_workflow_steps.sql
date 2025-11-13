-- Migration: 002_create_workflow_steps
-- Description: Create workflow_steps table for tracking individual workflow step history
-- Dependencies: 001_create_workflow_executions (foreign key reference)
-- Created: 2025-10-07

-- Workflow step history table
CREATE TABLE IF NOT EXISTS workflow_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_name VARCHAR(100) NOT NULL,
    step_order INTEGER NOT NULL CHECK (step_order >= 0),
    status VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER CHECK (execution_time_ms >= 0),
    step_data JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id
    ON workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status
    ON workflow_steps(status);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_created_at
    ON workflow_steps(created_at DESC);

-- Compound index for common queries
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id_order
    ON workflow_steps(workflow_id, step_order);

-- Comments for documentation
COMMENT ON TABLE workflow_steps IS 'Tracks individual step execution within workflows';
COMMENT ON COLUMN workflow_steps.workflow_id IS 'Foreign key reference to parent workflow execution';
COMMENT ON COLUMN workflow_steps.step_order IS 'Sequential order of step execution';
COMMENT ON COLUMN workflow_steps.step_data IS 'Step-specific data and parameters as JSON';
