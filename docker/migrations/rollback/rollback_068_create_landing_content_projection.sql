-- =============================================================================
-- ROLLBACK: Drop landing_content_projection table
-- =============================================================================
-- Ticket: OMN-9661

DROP TABLE IF EXISTS landing_content_projection;

-- Revert migration sentinel
UPDATE public.db_metadata
SET schema_version = '067',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
