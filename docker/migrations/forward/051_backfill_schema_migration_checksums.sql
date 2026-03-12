-- =============================================================================
-- MIGRATION: Backfill NULL checksums and enforce NOT NULL on schema_migrations
-- =============================================================================
-- Ticket: OMN-4701 (OMN-4653 root cause fix)
-- Version: 1.0.0
--
-- PURPOSE:
--   The live omnibase_infra DB has 22 NULL checksum rows in schema_migrations.
--   These rows were applied before checksum tracking was implemented (the bash
--   runner used nullable DDL, contradicting the Python runner's NOT NULL spec).
--
--   This migration:
--   1. Backfills NULL checksums with a sentinel prefix so they are
--      identifiable as pre-checksum-era rows.
--   2. Enforces NOT NULL on the checksum column to prevent future nulls.
--
-- SAFETY:
--   Wrapped in a transaction. Idempotent: UPDATE only touches NULL rows.
--   ALTER TABLE SET NOT NULL is safe once all rows have a non-null checksum.
--
-- ROLLBACK:
--   See rollback/rollback_045_backfill_schema_migration_checksums.sql
-- =============================================================================

BEGIN;

-- Step 1: Backfill all NULL checksum rows with a stable sentinel value.
-- The prefix "backfilled:pre-checksum-era:" identifies rows patched by
-- this migration. The suffix is the version for traceability.
UPDATE public.schema_migrations
SET checksum = 'backfilled:pre-checksum-era:' || version
WHERE checksum IS NULL;

-- Step 2: Enforce NOT NULL now that all rows have a value.
ALTER TABLE public.schema_migrations
    ALTER COLUMN checksum SET NOT NULL;

-- Step 3: Document the column contract with a comment.
COMMENT ON COLUMN public.schema_migrations.checksum IS
    'SHA-256 of migration file at apply time. '
    'Prefix "backfilled:pre-checksum-era:" = applied before checksum tracking '
    '(2026-03-12 backfill, OMN-4653 / OMN-4701).';

COMMIT;
