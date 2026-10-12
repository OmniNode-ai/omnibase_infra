-- =============================================================================
-- ROLLBACK 111: Restore the open/suppressed-only active fingerprint index
-- =============================================================================
-- Ticket: OMN-21029
--
-- WARNING:
--   Run this only with the pre-OMN-21029 handler deployed. Rows already in
--   'escalated' stay escalated but no longer take part in the uniqueness.
-- =============================================================================

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_runtime_error_triage_active_fingerprint
    ON public.runtime_error_triage (fingerprint)
    WHERE incident_state IN ('open', 'suppressed');

DROP INDEX CONCURRENTLY IF EXISTS public.uq_runtime_error_triage_active_fingerprint_v2;
