-- Migration: 002_create_workflow_steps
-- Description: Create workflow_steps table for tracking individual workflow step execution
-- Created: 2025-10-07

-- Workflow step history table
CREATE TABLE IF NOT EXISTS workflow_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_name VARCHAR(100) NOT NULL,
    step_order INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER,
    step_data JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps(status);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_created_at ON workflow_steps(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_step_order ON workflow_steps(workflow_id, step_order);

-- Add comments for documentation
COMMENT ON TABLE workflow_steps IS 'Tracks individual steps within workflow executions';
COMMENT ON COLUMN workflow_steps.workflow_id IS 'Foreign key reference to parent workflow execution';
COMMENT ON COLUMN workflow_steps.step_name IS 'Name of the workflow step (e.g., hash_generation, stamp_creation)';
COMMENT ON COLUMN workflow_steps.step_order IS 'Sequential order of step within workflow';
COMMENT ON COLUMN workflow_steps.status IS 'Step status (PENDING, RUNNING, COMPLETED, FAILED)';
COMMENT ON COLUMN workflow_steps.step_data IS 'Step-specific data as JSONB';
