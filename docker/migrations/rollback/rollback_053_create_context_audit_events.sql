DROP TABLE IF EXISTS public.context_audit_events;

UPDATE public.db_metadata
SET schema_version = '052',
    migrations_complete = FALSE,
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
