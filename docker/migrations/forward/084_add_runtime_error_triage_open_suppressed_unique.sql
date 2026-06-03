-- =============================================================================
-- MIGRATION: Add runtime_error_triage partial uniqueness for incident upsert
-- =============================================================================
-- Ticket: OMN-12618
-- Version: 1.0.0
--
-- PURPOSE:
--   The runtime error triage handler upserts active incidents with:
--     ON CONFLICT (fingerprint) WHERE incident_state IN ('open', 'suppressed')
--
--   PostgreSQL requires a matching unique or exclusion constraint for that
--   conflict target. Migration 055 created only non-unique indexes, causing
--   runtime triage writes to fail at deploy time.
--
-- IDEMPOTENCY:
--   - CREATE UNIQUE INDEX IF NOT EXISTS is safe to re-run.
--   - CONCURRENTLY avoids taking a long write lock on the live triage table.
--
-- ROLLBACK:
--   See rollback/rollback_084_add_runtime_error_triage_open_suppressed_unique.sql
-- =============================================================================

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_runtime_error_triage_active_fingerprint
    ON public.runtime_error_triage (fingerprint)
    WHERE incident_state IN ('open', 'suppressed');
