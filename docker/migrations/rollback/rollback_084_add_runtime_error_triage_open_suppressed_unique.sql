-- =============================================================================
-- ROLLBACK: Drop runtime_error_triage active fingerprint uniqueness
-- =============================================================================
-- Ticket: OMN-12618
--
-- WARNING:
--   Dropping this index makes the runtime error triage handler's active
--   incident upsert invalid again.
-- =============================================================================

DROP INDEX CONCURRENTLY IF EXISTS public.uq_runtime_error_triage_active_fingerprint;
