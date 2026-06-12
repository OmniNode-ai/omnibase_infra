-- =============================================================================
-- MIGRATION: Add runner_completed_at column to db_metadata (OMN-13062)
-- =============================================================================
-- Ticket: OMN-13062 (migration-gate vacuity fix — retro A-10)
-- Recurrences: OMN-12885, OMN-12934
-- Version: 1.0.0
--
-- PURPOSE:
--   Extends db_metadata with runner_completed_at TIMESTAMPTZ.  The forward
--   migration runner (run-forward-migrations.sh) stamps this column as its
--   FINAL act after every migration in the infra and node sets succeeds.
--
--   This timestamp is the durable evidence that the runner reached
--   successful completion.  Combined with the migrations_complete=TRUE
--   sentinel (OMN-3737), the migration-gate healthcheck now has two
--   independent signals:
--     1. migrations_complete=TRUE  — the flag was set without error
--     2. runner_completed_at IS NOT NULL — the runner reached its final act
--
--   The runner clears migrations_complete to FALSE at the start of every run
--   and sets it (together with runner_completed_at) only after all migrations
--   succeed.  Any nonzero exit from any migration leaves the gate UNHEALTHY.
--
-- IDEMPOTENCY:
--   ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_085_add_runner_completed_at_to_db_metadata.sql
-- =============================================================================

ALTER TABLE public.db_metadata
    ADD COLUMN IF NOT EXISTS runner_completed_at TIMESTAMPTZ;

COMMENT ON COLUMN public.db_metadata.runner_completed_at IS
    'Stamped by run-forward-migrations.sh as its final act after all '
    'infra and node migrations succeed.  NULL means the runner never '
    'completed successfully (partial run, first boot, or mid-run failure). '
    'OMN-13062 (retro A-10).';
