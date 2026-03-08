-- Rollback: 039_create_delta_bundles.sql
-- Ticket: OMN-3142

DROP TABLE IF EXISTS delta_bundles CASCADE;

UPDATE public.db_metadata
SET schema_version = '038',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
