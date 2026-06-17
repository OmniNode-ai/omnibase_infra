-- ONEX DLQ Replay History Schema
-- Ticket: OMN-12633 (retire ServiceDlqTracking runtime DDL)
-- Node: node_dlq_replay_effect
-- Version: 1.0.0
--
-- Design Notes:
--   - dlq_replay_history is the audit ledger for every DLQ replay attempt.
--   - Owned by node_dlq_replay_effect (CONTRACT/NODE/HANDLER pattern — OMN-12525).
--   - This file replaces the runtime CREATE TABLE IF NOT EXISTS that was
--     previously executed by ServiceDlqTracking._ensure_table_exists() on
--     plugin initialisation.  The imperative DDL has been removed from
--     ServiceDlqTracking; the forward migration runner is the canonical path.
--   - Schema is idempotent (IF NOT EXISTS) — safe to re-run.
--
-- Related Tickets:
--   - OMN-12633: retire ServiceDlqTracking runtime CREATE TABLE
--   - OMN-12619: contract-native DLQ replay node + quarantine
--   - OMN-1032:  PostgreSQL tracking integration

-- =============================================================================
-- MAIN TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS dlq_replay_history (
    -- Identity
    id UUID PRIMARY KEY,

    -- Source message identity
    original_message_id UUID NOT NULL,
    replay_correlation_id UUID NOT NULL,

    -- Routing
    original_topic VARCHAR(255) NOT NULL,
    target_topic VARCHAR(255) NOT NULL,

    -- Outcome
    replay_status VARCHAR(50) NOT NULL,
    replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    success BOOLEAN NOT NULL,
    error_message TEXT,

    -- Source position (for audit / dedup)
    dlq_offset BIGINT NOT NULL,
    dlq_partition INTEGER NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Message-level dedup and lookup (primary replay query path)
CREATE INDEX IF NOT EXISTS idx_dlq_replay_history_message_id
    ON dlq_replay_history (original_message_id);

-- Time-range scans for operator audit queries
CREATE INDEX IF NOT EXISTS idx_dlq_replay_history_timestamp
    ON dlq_replay_history (replay_timestamp);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE dlq_replay_history IS
    'Audit ledger for DLQ replay attempts (OMN-12633). '
    'One row per replay attempt; owned by node_dlq_replay_effect.';

COMMENT ON COLUMN dlq_replay_history.original_message_id IS
    'Kafka message key / correlation id from the original DLQ message.';

COMMENT ON COLUMN dlq_replay_history.replay_correlation_id IS
    'Correlation id of the replay run that processed this message.';

COMMENT ON COLUMN dlq_replay_history.replay_status IS
    'Terminal replay status (EnumReplayStatus): COMPLETED | FAILED | QUARANTINED | SKIPPED.';

COMMENT ON COLUMN dlq_replay_history.dlq_offset IS
    'Kafka offset in the DLQ topic at which the original message was consumed.';

COMMENT ON COLUMN dlq_replay_history.retry_count IS
    'Number of prior replay attempts for this message before this record.';
