-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 111: Keep an escalated runtime error incident unique per fingerprint
-- =============================================================================
-- Ticket: OMN-21029
--
-- PURPOSE:
--   NodeRuntimeErrorTriageEffect now moves an incident that crosses the
--   contract-declared escalation rule to incident_state = 'escalated' and
--   publishes one escalation event. Later occurrences of the same fingerprint
--   must keep counting against that same row, never open a second incident
--   that would escalate (and publish) again. The handler's upsert therefore
--   targets:
--     ON CONFLICT (fingerprint)
--     WHERE incident_state IN ('open', 'suppressed', 'escalated')
--   which needs a unique index whose predicate covers 'escalated' too.
--
-- ORDER:
--   The new index is built first, then the migration-084 index is dropped, so
--   there is no moment without a matching unique index for the old handler.
--
-- ROLLBACK:
--   See rollback/rollback_111_widen_runtime_error_triage_active_unique_to_escalated.sql
-- =============================================================================

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_runtime_error_triage_active_fingerprint_v2
    ON public.runtime_error_triage (fingerprint)
    WHERE incident_state IN ('open', 'suppressed', 'escalated');

DROP INDEX CONCURRENTLY IF EXISTS public.uq_runtime_error_triage_active_fingerprint;
