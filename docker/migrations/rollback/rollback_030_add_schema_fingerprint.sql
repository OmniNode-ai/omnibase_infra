-- Rollback: 030_add_schema_fingerprint
-- Description: Remove schema fingerprint columns from db_metadata (OMN-2087)

ALTER TABLE public.db_metadata
    DROP COLUMN IF EXISTS expected_schema_fingerprint,
    DROP COLUMN IF EXISTS expected_schema_fingerprint_algo,
    DROP COLUMN IF EXISTS expected_schema_fingerprint_generated_at;

UPDATE public.db_metadata
SET schema_version = '029', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
