-- Migration: 006_create_metadata_stamps
-- Description: Create metadata_stamps table for audit trail of stamp operations
-- Created: 2025-10-07

-- Metadata stamps audit trail table
CREATE TABLE IF NOT EXISTS metadata_stamps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID REFERENCES workflow_executions(id) ON DELETE SET NULL,
    file_hash VARCHAR(128) NOT NULL,
    stamp_data JSONB NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_workflow_id ON metadata_stamps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_hash ON metadata_stamps(file_hash);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_namespace ON metadata_stamps(namespace);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_created_at ON metadata_stamps(created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE metadata_stamps IS 'Audit trail for metadata stamp operations';
COMMENT ON COLUMN metadata_stamps.workflow_id IS 'Optional foreign key to parent workflow execution';
COMMENT ON COLUMN metadata_stamps.file_hash IS 'BLAKE3 hash of stamped content';
COMMENT ON COLUMN metadata_stamps.stamp_data IS 'Complete stamp data as JSONB';
COMMENT ON COLUMN metadata_stamps.namespace IS 'Multi-tenant namespace identifier';
