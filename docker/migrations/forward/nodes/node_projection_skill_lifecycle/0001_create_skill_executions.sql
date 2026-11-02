-- Migration: 0001_create_skill_executions.sql
-- Purpose: the skill_executions table written by node_projection_skill_lifecycle.
--
-- The table is the one omnibase_infra's forward migration 048 creates for the
-- skill lifecycle consumer this node replaces; this file repeats that
-- definition with IF NOT EXISTS so a database that has the table is left
-- untouched and a fresh one gets the same shape.
--
--   * One row per skill-started or skill-completed event.
--   * event_id is the primary key and the UPSERT conflict target.
--   * run_id joins a started row to its completed row.
--   * Columns present on only one event type are nullable.

CREATE TABLE IF NOT EXISTS skill_executions (
    event_id                UUID            PRIMARY KEY,
    run_id                  UUID            NOT NULL,
    event_type              TEXT            NOT NULL CHECK (event_type IN ('started', 'completed')),
    skill_name              TEXT            NOT NULL,
    skill_id                TEXT,
    repo_id                 TEXT            NOT NULL,
    correlation_id          UUID            NOT NULL,
    args_count              INTEGER,
    status                  TEXT CHECK (status IN ('success', 'failed', 'partial')),
    duration_ms             INTEGER,
    error_type              TEXT,
    started_emit_failed     BOOLEAN         DEFAULT FALSE,
    session_id              TEXT,
    emitted_at              TIMESTAMPTZ     NOT NULL,
    received_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_executions_run_id
    ON skill_executions (run_id);
CREATE INDEX IF NOT EXISTS idx_skill_executions_skill_name
    ON skill_executions (skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_executions_emitted_at
    ON skill_executions (emitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_skill_executions_repo_id
    ON skill_executions (repo_id);
CREATE INDEX IF NOT EXISTS idx_skill_executions_skill_event_type
    ON skill_executions (skill_name, event_type);
