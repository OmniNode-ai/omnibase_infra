-- =============================================================================
-- MIGRATION: Backfill NULL checksums and enforce NOT NULL on schema_migrations
-- =============================================================================
-- Ticket: OMN-4701 (OMN-4653 root cause fix)
-- Version: 1.1.0
--
-- PURPOSE:
--   The live omnibase_infra DB has NULL checksum rows in schema_migrations.
--   These rows were applied before checksum tracking was implemented (the bash
--   runner used nullable DDL, contradicting the Python runner's NOT NULL spec).
--
--   This migration:
--   1. Backfills NULL checksums with a sentinel prefix so they are
--      identifiable as pre-checksum-era rows.
--   2. Enforces NOT NULL on the checksum column to prevent future nulls.
--
-- SCHEMA COMPATIBILITY NOTE:
--   The live DB (created by the bash runner) uses column "version" as PK.
--   The CI test DB (which applies all migrations in order) uses "migration_id"
--   as PK, per migration 036_create_schema_migrations.sql.
--   This migration adapts to whichever column name is present.
--
-- SAFETY:
--   Wrapped in a transaction. Idempotent: UPDATE only touches NULL rows.
--   ALTER TABLE SET NOT NULL is safe once all rows have a non-null checksum.
--
-- ROLLBACK:
--   See rollback/rollback_051_backfill_schema_migration_checksums.sql
-- =============================================================================

BEGIN;

-- Step 1: Backfill all NULL checksum rows with a stable sentinel value.
-- The prefix "backfilled:pre-checksum-era:" identifies rows patched by
-- this migration. Handles both "version" (bash runner schema) and
-- "migration_id" (migration 036 schema) column names.
DO $$
DECLARE
  id_col TEXT;
BEGIN
  -- Detect which primary key column name this schema uses.
  SELECT column_name INTO id_col
  FROM information_schema.columns
  WHERE table_schema = 'public'
    AND table_name   = 'schema_migrations'
    AND column_name IN ('version', 'migration_id')
  LIMIT 1;

  IF id_col IS NULL THEN
    RAISE NOTICE 'schema_migrations has neither version nor migration_id column — skipping backfill';
  ELSE
    EXECUTE format(
      'UPDATE public.schema_migrations SET checksum = ''backfilled:pre-checksum-era:'' || %I WHERE checksum IS NULL',
      id_col
    );
    GET DIAGNOSTICS id_col = ROW_COUNT;
    RAISE NOTICE 'Backfilled % NULL checksum rows', id_col;
  END IF;
END;
$$;

-- Step 2: Enforce NOT NULL now that all rows have a value.
ALTER TABLE public.schema_migrations
    ALTER COLUMN checksum SET NOT NULL;

-- Step 3: Document the column contract with a comment.
COMMENT ON COLUMN public.schema_migrations.checksum IS
    'SHA-256 of migration file at apply time. '
    'Prefix "backfilled:pre-checksum-era:" = applied before checksum tracking '
    '(2026-03-12 backfill, OMN-4653 / OMN-4701).';

COMMIT;
