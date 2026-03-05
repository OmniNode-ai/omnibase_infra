-- =============================================================================
-- MIGRATION: Add migration-complete sentinel to db_metadata
-- =============================================================================
-- Ticket: OMN-3737 (Boot-Order Migration Sentinel)
-- Version: 1.0.0
--
-- PURPOSE:
--   Adds a migrations_complete flag to the db_metadata singleton row.
--   This sentinel is checked by runtime services via a docker-compose
--   healthcheck to ensure all forward migrations have been applied before
--   any runtime service starts. Prevents the race condition where
--   omniintelligence's validate_handshake auto-stamps a schema fingerprint
--   before tables like idempotency_records exist (Audit Gap #7).
--
-- IDEMPOTENCY:
--   - ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to re-run.
--   - UPDATE with WHERE clause is idempotent.
--
-- ROLLBACK:
--   See rollback/rollback_037_migration_sentinel.sql
-- =============================================================================

-- Add the sentinel column to the existing db_metadata singleton table.
ALTER TABLE public.db_metadata
    ADD COLUMN IF NOT EXISTS migrations_complete BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN public.db_metadata.migrations_complete IS
    'Set to TRUE by the final forward migration (037). Runtime services '
    'gate their startup on this flag via a docker-compose healthcheck '
    'to prevent schema fingerprint races (OMN-3737).';

-- Set the sentinel to TRUE. This migration is the last in the forward set,
-- so its execution proves all prior migrations have been applied.
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '037',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
