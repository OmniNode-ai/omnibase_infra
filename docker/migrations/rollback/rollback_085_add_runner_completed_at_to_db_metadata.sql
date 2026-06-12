-- =============================================================================
-- ROLLBACK: Remove runner_completed_at column from db_metadata (OMN-13062)
-- =============================================================================
-- Ticket: OMN-13062 (migration-gate vacuity fix — retro A-10)
-- Version: 1.0.0
-- =============================================================================

ALTER TABLE public.db_metadata
    DROP COLUMN IF EXISTS runner_completed_at;
