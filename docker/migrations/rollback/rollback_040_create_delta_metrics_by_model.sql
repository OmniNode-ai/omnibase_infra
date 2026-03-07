-- Rollback: 040_create_delta_metrics_by_model.sql
-- Ticket: OMN-3142

DROP TABLE IF EXISTS delta_metrics_by_model CASCADE;

UPDATE public.db_metadata
SET schema_version = '039',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
