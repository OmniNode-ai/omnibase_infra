-- Rollback: 037_migration_sentinel
-- Removes the migrations_complete sentinel column from db_metadata.

ALTER TABLE public.db_metadata DROP COLUMN IF EXISTS migrations_complete;

UPDATE public.db_metadata
SET schema_version = '036', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
