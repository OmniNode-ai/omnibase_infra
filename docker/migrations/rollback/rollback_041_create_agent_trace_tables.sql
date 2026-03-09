-- Rollback: 041_create_agent_trace_tables.sql
-- Ticket: OMN-4080
-- Drop in reverse dependency order

DROP TABLE IF EXISTS fix_transitions CASCADE;
DROP TABLE IF EXISTS frame_pr_association CASCADE;
DROP TABLE IF EXISTS pr_envelopes CASCADE;
DROP TABLE IF EXISTS change_frames CASCADE;
DROP TABLE IF EXISTS failure_signatures CASCADE;

UPDATE public.db_metadata
SET schema_version = '040',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
