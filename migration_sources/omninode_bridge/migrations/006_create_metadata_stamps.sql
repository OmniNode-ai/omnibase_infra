-- Migration: 006_create_metadata_stamps
-- Description: Create metadata_stamps table for audit trail of metadata stamping operations
-- Dependencies: 001_create_workflow_executions (optional foreign key reference)
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

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_workflow_id
    ON metadata_stamps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_hash
    ON metadata_stamps(file_hash);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_namespace
    ON metadata_stamps(namespace);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_created_at
    ON metadata_stamps(created_at DESC);

-- Compound index for common queries
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_namespace_created_at
    ON metadata_stamps(namespace, created_at DESC);

-- Comments for documentation
COMMENT ON TABLE metadata_stamps IS 'Audit trail for metadata stamping operations';
COMMENT ON COLUMN metadata_stamps.workflow_id IS 'Optional foreign key to associated workflow execution';
COMMENT ON COLUMN metadata_stamps.file_hash IS 'BLAKE3 hash of stamped file';
COMMENT ON COLUMN metadata_stamps.stamp_data IS 'Complete stamp data including metadata as JSON';
COMMENT ON COLUMN metadata_stamps.namespace IS 'Multi-tenant isolation namespace';
