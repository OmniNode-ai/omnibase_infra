-- =============================================================================
-- ROLLBACK: Drop routing_outcomes and capability_scores tables
-- =============================================================================
-- Ticket: OMN-7277

DROP TABLE IF EXISTS public.capability_scores;
DROP TABLE IF EXISTS public.routing_outcomes;

-- Revert migration sentinel
UPDATE public.db_metadata
SET schema_version = '057',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
