-- =============================================================================
-- ROLLBACK: Drop remote_task_state table
-- =============================================================================
-- Ticket: OMN-9631

DROP TABLE IF EXISTS public.remote_task_state;

-- Revert migration sentinel
UPDATE public.db_metadata
SET schema_version = '068',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
