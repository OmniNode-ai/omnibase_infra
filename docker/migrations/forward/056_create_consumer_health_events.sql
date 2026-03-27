-- =============================================================================
-- MIGRATION: Create consumer_health_events read-model projection table
-- =============================================================================
-- Ticket: OMN-6757 (Wire consumer-health topic with read-model projection)
-- Epic: OMN-5529 (Runtime Health Event Pipeline)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the consumer_health_events table for the read-model projection
--   consumer. This table stores raw consumer health events consumed from
--   onex.evt.omnibase-infra.consumer-health.v1 for omnidash /consumer-health
--   dashboard queries.
--
--   Separate from consumer_health_triage (migration 054) which stores
--   incident state for graduated triage response. This table stores the
--   raw event stream for historical analysis and dashboard display.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   DROP TABLE IF EXISTS consumer_health_events;
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.consumer_health_events (
    id                      BIGSERIAL       PRIMARY KEY,
    event_id                UUID            NOT NULL UNIQUE,
    correlation_id          UUID,
    consumer_identity       TEXT            NOT NULL,
    consumer_group          TEXT            NOT NULL,
    topic                   TEXT            NOT NULL,
    event_type              TEXT            NOT NULL,
    severity                TEXT            NOT NULL,
    fingerprint             TEXT            NOT NULL,
    rebalance_duration_ms   INT,
    partitions_assigned     INT,
    partitions_revoked      INT,
    error_message           TEXT            DEFAULT '',
    error_type              TEXT            DEFAULT '',
    hostname                TEXT            DEFAULT '',
    service_label           TEXT            DEFAULT '',
    emitted_at              TIMESTAMPTZ     NOT NULL,
    ingested_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Primary lookup: consumer identity + time range
CREATE INDEX IF NOT EXISTS idx_consumer_health_events_consumer_time
    ON consumer_health_events (consumer_identity, emitted_at DESC);

-- Dashboard filter: event type + severity
CREATE INDEX IF NOT EXISTS idx_consumer_health_events_type_severity
    ON consumer_health_events (event_type, severity);

-- Fingerprint lookup for correlation with triage table
CREATE INDEX IF NOT EXISTS idx_consumer_health_events_fingerprint
    ON consumer_health_events (fingerprint);

-- Time-range scans for dashboard
CREATE INDEX IF NOT EXISTS idx_consumer_health_events_emitted_at
    ON consumer_health_events (emitted_at DESC);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE consumer_health_events IS
    'Read-model projection of consumer health events (OMN-6757). '
    'Raw event stream from onex.evt.omnibase-infra.consumer-health.v1 '
    'for omnidash /consumer-health dashboard queries.';

COMMENT ON COLUMN consumer_health_events.event_id IS
    'Unique event identifier (UUID). Used for idempotent upserts.';

COMMENT ON COLUMN consumer_health_events.fingerprint IS
    'Stable hash of (consumer_identity, event_type, topic) for deduplication '
    'and correlation with consumer_health_triage table.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '056',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
