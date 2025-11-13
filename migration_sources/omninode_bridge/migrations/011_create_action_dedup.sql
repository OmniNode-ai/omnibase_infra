-- Migration: 011_create_action_dedup
-- Description: Create action_dedup_log table for preventing duplicate action processing
-- Dependencies: None (standalone deduplication table)
-- Created: 2025-10-21

-- Action deduplication log table (from ModelActionDedup)
-- Prevents duplicate processing from retries or at-least-once delivery guarantees
CREATE TABLE IF NOT EXISTS action_dedup_log (
    workflow_key TEXT NOT NULL,
    action_id UUID NOT NULL,
    result_hash TEXT NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (workflow_key, action_id)
);

-- Performance index for TTL-based cleanup
CREATE INDEX IF NOT EXISTS idx_action_dedup_expires
    ON action_dedup_log(expires_at);

-- Comments for documentation
COMMENT ON TABLE action_dedup_log IS 'Action deduplication log for preventing duplicate processing from retries';
COMMENT ON COLUMN action_dedup_log.workflow_key IS 'Workflow identifier for grouping actions';
COMMENT ON COLUMN action_dedup_log.action_id IS 'Unique action identifier (UUID)';
COMMENT ON COLUMN action_dedup_log.result_hash IS 'SHA256 hash of action result for validation on replay';
COMMENT ON COLUMN action_dedup_log.processed_at IS 'Timestamp when action was first processed';
COMMENT ON COLUMN action_dedup_log.expires_at IS 'TTL expiration timestamp (default 6 hours), enables efficient cleanup';

-- Cleanup job runs periodically: DELETE FROM action_dedup_log WHERE expires_at < NOW()
-- Default TTL: 6 hours (21600 seconds)
-- Composite primary key (workflow_key, action_id) ensures uniqueness and prevents duplicates
