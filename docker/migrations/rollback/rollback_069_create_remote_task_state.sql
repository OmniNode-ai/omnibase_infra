-- ROLLBACK: Drop remote_task_state table

DROP TABLE IF EXISTS remote_task_state;

UPDATE public.db_metadata
SET schema_version = '068',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
