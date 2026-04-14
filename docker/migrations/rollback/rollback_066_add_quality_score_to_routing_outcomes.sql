-- =============================================================================
-- ROLLBACK: Remove quality_score column from routing_outcomes
-- =============================================================================
-- Ticket: OMN-new

ALTER TABLE public.routing_outcomes
    DROP COLUMN IF EXISTS quality_score;

-- Revert migration sentinel
UPDATE public.db_metadata
SET schema_version = '065',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
