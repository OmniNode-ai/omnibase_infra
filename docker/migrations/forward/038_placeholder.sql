-- =============================================================================
-- MIGRATION: Placeholder for intentionally skipped migration 038
-- =============================================================================
-- Ticket: OMN-4086 (Pipeline Audit Gap #8 — document missing 038)
-- Version: 1.0.0
--
-- PURPOSE:
--   This migration was intentionally skipped. Migration number 038 was never
--   created; the sequence jumped from 037 to 039 (OMN-3142, delta_bundles).
--
--   A gap in migration numbers is NOT an error — the migration runner applies
--   files in lexicographic order and does not require contiguous numbering.
--   However, the gap caused pipeline audit concern (OMN-4086) and this
--   placeholder exists to document the deliberate skip.
--
-- HISTORY:
--   037 — migration_sentinel (OMN-3737): Added migrations_complete sentinel
--   038 — (this placeholder): Documents the numbering gap
--   039 — create_delta_bundles (OMN-3142): delta bundle tracking table
--   040 — create_delta_metrics_by_model (OMN-3142): model metrics aggregates
--
-- ROLLBACK:
--   No-op — this migration performs no schema changes.
--   See rollback/rollback_038_placeholder.sql
-- =============================================================================

-- This migration intentionally performs no schema changes.
-- It exists solely to document that migration 038 was deliberately skipped.
-- The sequence jumped from 037 (migration_sentinel) to 039 (delta_bundles).

DO $$
BEGIN
  RAISE NOTICE 'Migration 038: placeholder — no schema changes (intentional numbering gap, OMN-4086)';
END;
$$;
