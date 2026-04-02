-- Migration: 006_create_skill_executions.sql
-- Purpose: Create skill_executions table for skill lifecycle observability
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-27
-- Ticket: OMN-2934
--
-- Design Decisions:
--
--   1. Single table, two event_type values:
--      "started" rows are emitted before skill dispatch.
--      "completed" rows are emitted after skill dispatch (success or failure).
--      Both share run_id as the join key, enabling duration calculation by
--      joining on run_id.
--
--   2. Idempotency via event_id primary key:
--      Both topic types include a UUID event_id. Consumers use
--      ON CONFLICT (event_id) DO NOTHING for at-least-once safety.
--
--   3. Nullable vs required columns:
--      Columns only present on one event type are nullable:
--        - skill_id: only in started events (repo-relative path)
--        - args_count: only in started events
--        - status, duration_ms, error_type, started_emit_failed: only in completed events
--      Columns present in both types are NOT NULL.
--
--   4. Partitioning not applied at v1:
--      Table is expected to be low-volume (one pair per skill invocation).
--      Add time-based partitioning in a future migration if needed.
--
--   5. Index on run_id supports efficient join between started/completed rows.
--      Index on skill_name supports omnidash skill monitoring queries.
--      Index on emitted_at supports time-range filtering.

-- =============================================================================
-- TABLE: skill_executions
-- =============================================================================
-- One row per skill-started or skill-completed event.
-- Pairs are joined via run_id for duration and outcome analysis.
-- =============================================================================

CREATE TABLE IF NOT EXISTS skill_executions (
    -- Unique event identifier (UUID from omniclaude event model)
    event_id                UUID            PRIMARY KEY,

    -- Join key linking a started event to its completed counterpart
    run_id                  UUID            NOT NULL,

    -- Discriminator: 'started' or 'completed'
    event_type              TEXT            NOT NULL CHECK (event_type IN ('started', 'completed')),

    -- Skill identity
    skill_name              TEXT            NOT NULL,   -- e.g. "pr-review"
    skill_id                TEXT,                       -- repo-relative path (started events only)
    repo_id                 TEXT            NOT NULL,   -- e.g. "omniclaude"

    -- Correlation
    correlation_id          UUID            NOT NULL,

    -- Started-event fields (NULL for completed events)
    args_count              INTEGER,                    -- Count of args provided (not values)

    -- Completed-event fields (NULL for started events)
    status                  TEXT CHECK (status IN ('success', 'failed', 'partial')),
    duration_ms             INTEGER,                    -- Wall-clock duration (perf_counter)
    error_type              TEXT,                       -- Exception class name if raised
    started_emit_failed     BOOLEAN         DEFAULT FALSE,  -- True if started emission failed

    -- Optional session context
    session_id              TEXT,                       -- Claude Code session identifier

    -- Timestamp from the event model (UTC)
    emitted_at              TIMESTAMPTZ     NOT NULL,

    -- Audit timestamp set by the consumer on insert
    received_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Support join: SELECT * FROM skill_executions WHERE run_id = $1
CREATE INDEX IF NOT EXISTS idx_skill_executions_run_id
    ON skill_executions (run_id);

-- Support omnidash per-skill analytics: WHERE skill_name = $1
CREATE INDEX IF NOT EXISTS idx_skill_executions_skill_name
    ON skill_executions (skill_name);

-- Support time-range queries: WHERE emitted_at BETWEEN $1 AND $2
CREATE INDEX IF NOT EXISTS idx_skill_executions_emitted_at
    ON skill_executions (emitted_at DESC);

-- Support repo-scoped queries: WHERE repo_id = $1
CREATE INDEX IF NOT EXISTS idx_skill_executions_repo_id
    ON skill_executions (repo_id);

-- Support per-skill status analysis: WHERE skill_name = $1 AND event_type = 'completed'
CREATE INDEX IF NOT EXISTS idx_skill_executions_skill_event_type
    ON skill_executions (skill_name, event_type);

COMMENT ON TABLE skill_executions IS
    'Skill lifecycle observability events (OMN-2934). '
    'One row per skill-started or skill-completed event emitted by omniclaude. '
    'Pairs share run_id as join key for duration and outcome analysis.';

COMMENT ON COLUMN skill_executions.event_id IS
    'UUID from ModelSkillStartedEvent or ModelSkillCompletedEvent. Primary key for idempotency.';
COMMENT ON COLUMN skill_executions.run_id IS
    'Join key shared between a started and its completed counterpart.';
COMMENT ON COLUMN skill_executions.event_type IS
    'Discriminator: started (before dispatch) or completed (after dispatch).';
COMMENT ON COLUMN skill_executions.skill_name IS
    'Human-readable skill identifier, e.g. "pr-review".';
COMMENT ON COLUMN skill_executions.skill_id IS
    'Repo-relative skill path, e.g. "plugins/onex/skills/pr-review". Only in started events.';
COMMENT ON COLUMN skill_executions.repo_id IS
    'Repository identifier, e.g. "omniclaude". Prevents cross-repo collisions.';
COMMENT ON COLUMN skill_executions.correlation_id IS
    'End-to-end correlation ID from the originating request.';
COMMENT ON COLUMN skill_executions.args_count IS
    'Count of args provided (not values — privacy). Only in started events.';
COMMENT ON COLUMN skill_executions.status IS
    'Outcome: success, failed, or partial. Only in completed events.';
COMMENT ON COLUMN skill_executions.duration_ms IS
    'Wall-clock duration from perf_counter(). NTP-immune. Only in completed events.';
COMMENT ON COLUMN skill_executions.error_type IS
    'Exception class name if task_dispatcher raised. Only in completed events.';
COMMENT ON COLUMN skill_executions.started_emit_failed IS
    'True if the skill.started emission itself failed. Consumers detect orphaned completed events.';
COMMENT ON COLUMN skill_executions.session_id IS
    'Optional Claude Code session identifier.';
COMMENT ON COLUMN skill_executions.emitted_at IS
    'UTC timestamp when the event was emitted by omniclaude.';
COMMENT ON COLUMN skill_executions.received_at IS
    'UTC timestamp when the event was received and persisted by omnibase_infra.';
