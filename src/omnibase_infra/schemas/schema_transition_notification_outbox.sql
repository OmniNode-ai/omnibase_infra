-- SPDX-License-Identifier: MIT
-- Copyright (c) 2025 OmniNode Team
--
-- Transition Notification Outbox Schema
-- Ticket: OMN-1139 (TransitionNotificationOutbox implementation)
-- Version: 1.0.0
--
-- Design Notes:
--   - Implements the outbox pattern for guaranteed notification delivery
--   - Stores notifications in same transaction as projections for atomicity
--   - Supports concurrent processing via SELECT FOR UPDATE SKIP LOCKED
--   - Includes retry tracking with error recording
--   - Indexes optimized for processor query patterns:
--     1. Pending notification queries (processed_at IS NULL)
--     2. Aggregate-specific queries (aggregate_type, aggregate_id)
--     3. Retry processing (high retry counts for monitoring)
--     4. Cleanup of old processed records
--   - All timestamps are timezone-aware (TIMESTAMPTZ)
--   - This schema is idempotent (IF NOT EXISTS used throughout)
--
-- Usage:
--   Execute this SQL file to create or update the outbox schema.
--   The schema is designed to be re-run safely (idempotent).
--
-- Related:
--   - TransitionNotificationOutbox: src/omnibase_infra/runtime/transition_notification_outbox.py
--   - ModelStateTransitionNotification: omnibase_core.models.notifications
--   - ProtocolTransitionNotificationPublisher: omnibase_core.protocols.notifications

-- =============================================================================
-- MAIN TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS transition_notification_outbox (
    -- Identity
    id BIGSERIAL PRIMARY KEY,

    -- Notification Payload (serialized ModelStateTransitionNotification)
    notification_data JSONB NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- processed_at encodes both "pending" and "processed" states in a single column:
    --   NULL = pending (not yet processed, eligible for processing)
    --   non-NULL = processed (successfully published, timestamp of completion)
    -- Note: No separate "processed_implies_no_pending" constraint is needed because
    -- the single-column semantic makes the states mutually exclusive by design -
    -- a row cannot be both pending (NULL) and processed (non-NULL) simultaneously.
    processed_at TIMESTAMPTZ,

    -- Retry Tracking
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT,  -- Most recent error message (sanitized, max 1000 chars)

    -- Aggregate Information (for queries and debugging)
    aggregate_type TEXT NOT NULL,
    aggregate_id UUID NOT NULL,

    -- Constraints
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0)
);

-- =============================================================================
-- INDEXES FOR OUTBOX PROCESSING
-- =============================================================================

-- Index for efficient pending notification queries (PRIMARY query pattern)
-- The processor uses: SELECT ... WHERE processed_at IS NULL ORDER BY created_at
-- Partial index: only index rows with NULL processed_at (pending notifications)
-- Query pattern: SELECT id, notification_data FROM transition_notification_outbox
--                WHERE processed_at IS NULL ORDER BY created_at LIMIT :batch_size
--                FOR UPDATE SKIP LOCKED
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON transition_notification_outbox (created_at)
    WHERE processed_at IS NULL;

-- Index for aggregate-specific queries (debugging and monitoring)
-- Enables queries like "show all notifications for entity X"
-- Query pattern: SELECT * FROM transition_notification_outbox
--                WHERE aggregate_type = :type AND aggregate_id = :id
CREATE INDEX IF NOT EXISTS idx_outbox_aggregate
    ON transition_notification_outbox (aggregate_type, aggregate_id);

-- Index for retry monitoring (identify stuck/failing notifications)
-- Partial index: only pending notifications with retries
-- Query pattern: SELECT * FROM transition_notification_outbox
--                WHERE processed_at IS NULL AND retry_count > :threshold
CREATE INDEX IF NOT EXISTS idx_outbox_retry_pending
    ON transition_notification_outbox (retry_count, created_at)
    WHERE processed_at IS NULL AND retry_count > 0;

-- Index for cleanup queries (delete old processed records)
-- Enables efficient deletion of processed records older than a threshold
-- Query pattern: DELETE FROM transition_notification_outbox
--                WHERE processed_at IS NOT NULL AND processed_at < :cutoff_time
CREATE INDEX IF NOT EXISTS idx_outbox_cleanup
    ON transition_notification_outbox (processed_at)
    WHERE processed_at IS NOT NULL;

-- Index for aggregate type filtering with pending status
-- Useful for monitoring specific aggregate types
-- Query pattern: SELECT COUNT(*) FROM transition_notification_outbox
--                WHERE aggregate_type = :type AND processed_at IS NULL
CREATE INDEX IF NOT EXISTS idx_outbox_aggregate_type_pending
    ON transition_notification_outbox (aggregate_type)
    WHERE processed_at IS NULL;

-- =============================================================================
-- TABLE AND COLUMN COMMENTS
-- =============================================================================

COMMENT ON TABLE transition_notification_outbox IS
    'Outbox pattern table for guaranteed state transition notification delivery (OMN-1139). '
    'Stores notifications in same transaction as projections for atomicity. '
    'Background processor publishes pending notifications asynchronously.';

COMMENT ON COLUMN transition_notification_outbox.id IS
    'Auto-incrementing primary key. Used for efficient row locking with FOR UPDATE SKIP LOCKED.';

COMMENT ON COLUMN transition_notification_outbox.notification_data IS
    'Serialized ModelStateTransitionNotification as JSONB. Contains all notification fields: '
    'aggregate_type, aggregate_id, transition (from_state, to_state, event_type), '
    'correlation_id, timestamp, and optional metadata.';

COMMENT ON COLUMN transition_notification_outbox.created_at IS
    'Timestamp when notification was stored in outbox. Used for FIFO ordering of pending notifications.';

COMMENT ON COLUMN transition_notification_outbox.processed_at IS
    'Timestamp when notification was successfully published. NULL indicates pending status.';

COMMENT ON COLUMN transition_notification_outbox.retry_count IS
    'Number of failed publish attempts. Incremented on each failure, not reset on success. '
    'Use for monitoring and alerting on stuck notifications.';

COMMENT ON COLUMN transition_notification_outbox.last_error IS
    'Most recent error message from failed publish attempt (sanitized, max 1000 chars). '
    'Updated on each failure for debugging. May contain value even when processed_at is set '
    'if earlier retries failed before final success.';

COMMENT ON COLUMN transition_notification_outbox.aggregate_type IS
    'Type of aggregate this notification is for (e.g., "registration", "workflow"). '
    'Denormalized from notification_data for efficient indexing and filtering.';

COMMENT ON COLUMN transition_notification_outbox.aggregate_id IS
    'UUID of the aggregate entity this notification is for. '
    'Denormalized from notification_data for efficient indexing and aggregate-specific queries.';

-- =============================================================================
-- INDEX STRATEGY DOCUMENTATION
-- =============================================================================
--
-- This schema defines multiple indexes optimized for different query patterns
-- used by the TransitionNotificationOutbox processor and operators.
--
-- PENDING NOTIFICATION PROCESSING:
-- --------------------------------
-- 1. idx_outbox_pending:
--    - Single-column index on created_at
--    - WHERE: processed_at IS NULL
--    - Use case: Main processor query - fetch pending notifications in FIFO order
--    - Critical for: process_pending() batch retrieval
--
-- The partial WHERE clause keeps this index small and fast by excluding
-- all processed records.
--
-- AGGREGATE QUERIES (Debugging):
-- ------------------------------
-- 2. idx_outbox_aggregate:
--    - Composite index on (aggregate_type, aggregate_id)
--    - Use case: Query notifications for a specific entity
--    - Example: "Show all notifications for registration entity X"
--
-- RETRY MONITORING:
-- -----------------
-- 3. idx_outbox_retry_pending:
--    - Composite index on (retry_count, created_at)
--    - WHERE: processed_at IS NULL AND retry_count > 0
--    - Use case: Identify stuck or failing notifications
--    - Example: Alert when retry_count > 5 for pending notifications
--
-- CLEANUP OPERATIONS:
-- -------------------
-- 4. idx_outbox_cleanup:
--    - Single-column index on processed_at
--    - WHERE: processed_at IS NOT NULL
--    - Use case: Efficient deletion of old processed records
--    - Example: DELETE ... WHERE processed_at < NOW() - INTERVAL '7 days'
--
-- TYPE-SPECIFIC MONITORING:
-- -------------------------
-- 5. idx_outbox_aggregate_type_pending:
--    - Single-column index on aggregate_type
--    - WHERE: processed_at IS NULL
--    - Use case: Count pending notifications by type for monitoring dashboards
--
-- =============================================================================
-- CONCURRENT PROCESSING PATTERN
-- =============================================================================
--
-- The outbox processor uses SELECT FOR UPDATE SKIP LOCKED to enable safe
-- concurrent processing by multiple instances:
--
--   SELECT id, notification_data
--   FROM transition_notification_outbox
--   WHERE processed_at IS NULL
--   ORDER BY created_at
--   LIMIT :batch_size
--   FOR UPDATE SKIP LOCKED;
--
-- SKIP LOCKED ensures that if another processor has locked a row, this query
-- skips it rather than blocking. This prevents:
-- - Duplicate processing
-- - Deadlocks between processors
-- - Head-of-line blocking
--
-- The id column (BIGSERIAL PRIMARY KEY) provides an efficient row lock target.
--
-- =============================================================================
-- CLEANUP RECOMMENDATIONS
-- =============================================================================
--
-- Processed records should be periodically deleted to prevent table bloat.
-- Recommended cleanup query (run periodically via cron or pg_cron):
--
--   DELETE FROM transition_notification_outbox
--   WHERE processed_at IS NOT NULL
--   AND processed_at < NOW() - INTERVAL '7 days';
--
-- Alternatively, implement as a PostgreSQL function:
--
--   CREATE OR REPLACE FUNCTION cleanup_outbox(retention_interval INTERVAL)
--   RETURNS BIGINT AS $$
--   DECLARE
--     deleted_count BIGINT;
--   BEGIN
--     DELETE FROM transition_notification_outbox
--     WHERE processed_at IS NOT NULL
--     AND processed_at < NOW() - retention_interval;
--     GET DIAGNOSTICS deleted_count = ROW_COUNT;
--     RETURN deleted_count;
--   END;
--   $$ LANGUAGE plpgsql;
--
-- Usage: SELECT cleanup_outbox(INTERVAL '7 days');
--
