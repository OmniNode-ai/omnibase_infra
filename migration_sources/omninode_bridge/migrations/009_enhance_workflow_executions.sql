-- Migration: 009_enhance_workflow_executions
-- Description: Enhance workflow_executions table for orchestrator-specific fields
-- Dependencies: 001_create_workflow_executions.sql
-- Created: 2025-10-15

-- Add orchestrator-specific columns to workflow_executions table
-- These fields track stamping workflow results and metadata

-- Stamp results
ALTER TABLE workflow_executions
ADD COLUMN IF NOT EXISTS stamp_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);

-- Workflow execution details
ALTER TABLE workflow_executions
ADD COLUMN IF NOT EXISTS workflow_steps JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS intelligence_data JSONB;

-- Performance metrics
ALTER TABLE workflow_executions
ADD COLUMN IF NOT EXISTS hash_generation_time_ms INTEGER CHECK (hash_generation_time_ms >= 0),
ADD COLUMN IF NOT EXISTS workflow_steps_executed INTEGER DEFAULT 0 CHECK (workflow_steps_executed >= 0);

-- Session tracking (optional - for multi-session workflows)
ALTER TABLE workflow_executions
ADD COLUMN IF NOT EXISTS session_id UUID;

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS idx_workflow_executions_stamp_id
    ON workflow_executions(stamp_id)
    WHERE stamp_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_workflow_executions_file_hash
    ON workflow_executions(file_hash)
    WHERE file_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_id
    ON workflow_executions(session_id)
    WHERE session_id IS NOT NULL;

-- GIN indexes for JSONB columns (fast JSON queries)
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_steps_gin
    ON workflow_executions USING GIN(workflow_steps)
    WHERE workflow_steps IS NOT NULL AND workflow_steps != '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_workflow_executions_intelligence_data_gin
    ON workflow_executions USING GIN(intelligence_data)
    WHERE intelligence_data IS NOT NULL AND intelligence_data != '{}'::jsonb;

-- Comments for new columns
COMMENT ON COLUMN workflow_executions.stamp_id IS 'Unique stamp identifier (set when workflow completes successfully)';
COMMENT ON COLUMN workflow_executions.file_hash IS 'BLAKE3 hash of stamped file content';
COMMENT ON COLUMN workflow_executions.workflow_steps IS 'Array of workflow step execution details (step_type, status, duration_ms, etc.)';
COMMENT ON COLUMN workflow_executions.intelligence_data IS 'OnexTree intelligence analysis results (optional, NULL if not available)';
COMMENT ON COLUMN workflow_executions.hash_generation_time_ms IS 'BLAKE3 hash generation time in milliseconds';
COMMENT ON COLUMN workflow_executions.workflow_steps_executed IS 'Total number of workflow steps executed';
COMMENT ON COLUMN workflow_executions.session_id IS 'Session identifier for multi-session workflow grouping (optional)';
