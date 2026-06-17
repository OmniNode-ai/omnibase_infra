-- OMN-12633: Create the DLQ replay history table.
-- DDL owner: node_dlq_replay_effect (CONTRACT/NODE/HANDLER pattern — OMN-12525).
--
-- Previously this table was created at runtime by ServiceDlqTracking via an
-- imperative CREATE TABLE IF NOT EXISTS inside _ensure_table_exists().  That
-- method has been removed (OMN-12633).  The table is now provisioned here,
-- through the canonical forward migration runner, and is therefore visible to
-- public.schema_migrations.

CREATE TABLE IF NOT EXISTS dlq_replay_history (
    id UUID PRIMARY KEY,
    original_message_id UUID NOT NULL,
    replay_correlation_id UUID NOT NULL,
    original_topic VARCHAR(255) NOT NULL,
    target_topic VARCHAR(255) NOT NULL,
    replay_status VARCHAR(50) NOT NULL,
    replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    dlq_offset BIGINT NOT NULL,
    dlq_partition INTEGER NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_dlq_replay_history_message_id
    ON dlq_replay_history (original_message_id);

CREATE INDEX IF NOT EXISTS idx_dlq_replay_history_timestamp
    ON dlq_replay_history (replay_timestamp);
