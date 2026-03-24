-- =============================================================================
-- MIGRATION: Create consumer_health_triage and consumer_restart_state tables
-- =============================================================================
-- Ticket: OMN-5512 (Task 7: Create migration for consumer_health_triage table)
-- Epic: OMN-5529 (Runtime Health Event Pipeline)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the consumer health triage tables for Layer 1 of the runtime
--   health event pipeline. consumer_health_triage tracks incidents by
--   fingerprint for graduated response. consumer_restart_state tracks
--   restart rate limiting by consumer identity.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_054_create_consumer_health_triage.sql
-- =============================================================================

-- Incident tracking by fingerprint for graduated response
CREATE TABLE IF NOT EXISTS public.consumer_health_triage (
    id                      BIGSERIAL       PRIMARY KEY,
    fingerprint             TEXT            NOT NULL,
    consumer_id             TEXT            NOT NULL,
    consumer_group          TEXT            NOT NULL,
    topic                   TEXT            NOT NULL,
    event_type              TEXT            NOT NULL,
    severity                TEXT            NOT NULL,
    incident_state          TEXT            NOT NULL DEFAULT 'open',
    occurrence_count        INT             NOT NULL DEFAULT 1,
    first_seen_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_seen_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    escalated_at            TIMESTAMPTZ,
    resolved_at             TIMESTAMPTZ,
    linear_ticket_id        TEXT,
    error_message           TEXT            DEFAULT '',
    service_name            TEXT            DEFAULT '',
    hostname                TEXT            DEFAULT '',
    correlation_id          TEXT
);

-- Restart rate limiting by consumer identity
CREATE TABLE IF NOT EXISTS public.consumer_restart_state (
    id                      BIGSERIAL       PRIMARY KEY,
    consumer_id             TEXT            NOT NULL,
    consumer_group          TEXT            NOT NULL,
    topic                   TEXT            NOT NULL,
    last_restart_at         TIMESTAMPTZ,
    restart_count_30min     INT             NOT NULL DEFAULT 0,
    restart_window_start    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_restart_success    BOOLEAN,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (consumer_id, consumer_group, topic)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Triage table indexes
CREATE INDEX IF NOT EXISTS idx_consumer_health_triage_fingerprint
    ON consumer_health_triage (fingerprint);

CREATE INDEX IF NOT EXISTS idx_consumer_health_triage_consumer
    ON consumer_health_triage (consumer_id, consumer_group);

CREATE INDEX IF NOT EXISTS idx_consumer_health_triage_state
    ON consumer_health_triage (incident_state);

CREATE INDEX IF NOT EXISTS idx_consumer_health_triage_last_seen
    ON consumer_health_triage (last_seen_at);

CREATE INDEX IF NOT EXISTS idx_consumer_health_triage_fingerprint_state
    ON consumer_health_triage (fingerprint, incident_state);

-- Restart state indexes
CREATE INDEX IF NOT EXISTS idx_consumer_restart_state_consumer
    ON consumer_restart_state (consumer_id, consumer_group, topic);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE consumer_health_triage IS
    'Incident tracking for consumer health events (OMN-5512). '
    'Each row tracks an incident by fingerprint for graduated response: '
    'Slack warning -> Slack repeated -> restart command -> Linear ticket.';

COMMENT ON COLUMN consumer_health_triage.fingerprint IS
    'Stable hash of (consumer_id, event_type, topic) for deduplication.';

COMMENT ON COLUMN consumer_health_triage.incident_state IS
    'Lifecycle state: open, acknowledged, restart_pending, restart_succeeded, '
    'restart_failed, ticketed, resolved.';

COMMENT ON COLUMN consumer_health_triage.occurrence_count IS
    'Number of times this fingerprint has been seen in the current incident.';

COMMENT ON TABLE consumer_restart_state IS
    'Restart rate limiting state per consumer identity (OMN-5512). '
    'Prevents restart storms by tracking restart frequency within a 30-min window.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '054',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
