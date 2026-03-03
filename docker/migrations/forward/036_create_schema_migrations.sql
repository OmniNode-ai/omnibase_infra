-- =============================================================================
-- MIGRATION: Create schema_migrations tracking table
-- =============================================================================
-- Ticket: OMN-3526 (DB Migration Reliability)
-- Version: 1.0.0
--
-- PURPOSE:
--   Per-migration history table for the omnibase_infra migration runner.
--   Tracks which migrations have been applied, when, and their file checksum.
--   Enables idempotent reruns, gap detection, and tamper detection.
--
-- ROLLBACK:
--   See rollback/rollback_036_create_schema_migrations.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.schema_migrations (
    migration_id    TEXT PRIMARY KEY,          -- e.g. "docker/035_create_skill_executions.sql"
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum        TEXT NOT NULL,             -- SHA-256 of file contents at apply time
    source_set      TEXT NOT NULL              -- "docker" or "src"
);
