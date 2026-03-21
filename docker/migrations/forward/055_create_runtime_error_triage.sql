-- =============================================================================
-- MIGRATION: Create runtime_error_triage table
-- =============================================================================
-- Ticket: OMN-5514 (Task 13: Create migration for runtime_error_triage table)
-- Epic: OMN-5529 (Runtime Health Event Pipeline)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the runtime error triage table for Layer 2 of the runtime
--   health event pipeline. Tracks runtime error incidents by fingerprint
--   for graduated triage response with cross-layer correlation to Layer 1
--   consumer health incidents.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_055_create_runtime_error_triage.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.runtime_error_triage (
    id                      BIGSERIAL       PRIMARY KEY,
    fingerprint             TEXT            NOT NULL,
    logger_name             TEXT            NOT NULL,
    error_category          TEXT            NOT NULL,
    severity                TEXT            NOT NULL,
    incident_state          TEXT            NOT NULL DEFAULT 'open',
    occurrence_count        INT             NOT NULL DEFAULT 1,
    message_template        TEXT            NOT NULL DEFAULT '',
    first_seen_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_seen_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    escalated_at            TIMESTAMPTZ,
    resolved_at             TIMESTAMPTZ,
    linear_ticket_id        TEXT,
    -- Cross-layer correlation: links to consumer_health_triage fingerprint
    correlated_consumer_fingerprint TEXT,
    service_name            TEXT            DEFAULT '',
    hostname                TEXT            DEFAULT '',
    correlation_id          TEXT
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_fingerprint
    ON runtime_error_triage (fingerprint);

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_category
    ON runtime_error_triage (error_category);

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_state
    ON runtime_error_triage (incident_state);

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_last_seen
    ON runtime_error_triage (last_seen_at);

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_fingerprint_state
    ON runtime_error_triage (fingerprint, incident_state);

CREATE INDEX IF NOT EXISTS idx_runtime_error_triage_correlation
    ON runtime_error_triage (correlated_consumer_fingerprint);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE runtime_error_triage IS
    'Incident tracking for runtime error events (OMN-5514). '
    'Each row tracks an error incident by fingerprint for graduated triage. '
    'Supports cross-layer correlation with consumer_health_triage via '
    'correlated_consumer_fingerprint.';

COMMENT ON COLUMN runtime_error_triage.fingerprint IS
    'Stable hash of (logger_name, message_template, error_category) for deduplication.';

COMMENT ON COLUMN runtime_error_triage.incident_state IS
    'Lifecycle state: open, acknowledged, suppressed, ticketed, resolved.';

COMMENT ON COLUMN runtime_error_triage.message_template IS
    'Templatized error message with variable parts replaced by placeholders.';

COMMENT ON COLUMN runtime_error_triage.correlated_consumer_fingerprint IS
    'Links to consumer_health_triage.fingerprint for cross-layer correlation. '
    'NULL if no consumer health correlation exists.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '055',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
